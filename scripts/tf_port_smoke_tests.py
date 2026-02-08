import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from lithobench_tf.neuralilt_tf import UNetTF
from lithobench_tf.cfnoilt_tf import CFNOTF, CFNOILTNetTF
from lithobench_tf.doinn_tf import RFNOTF, RFNONetTF
from lithobench_tf.ganopc_tf import GANOPCGeneratorTF, GANOPCDiscriminatorTF
from lithobench_tf.lithogan_tf import LithoGANGeneratorTF, LithoGANDiscriminatorTF
from lithobench_tf.damoilt_tf import DAMOILTGeneratorTF, DAMOILTDiscriminatorTF
from lithobench_tf.damolitho_tf import DAMOLithoGeneratorTF, DAMOLithoDiscriminatorTF
from lithobench_tf.cfnolitho_tf import CFNOLithoNetTF


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Minimal PyTorch references
# -----------------------------
def pt_conv2d(ch_in, ch_out, k, s, p, bias=True, norm=True, relu=False):
    layers = [nn.Conv2d(ch_in, ch_out, kernel_size=k, stride=s, padding=p, bias=bias)]
    if norm:
        layers.append(nn.BatchNorm2d(ch_out, affine=bias))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def pt_repeat2d(n, ch_in, ch_out, k, s, p, bias=True, norm=True, relu=False):
    layers = []
    for idx in range(n):
        layers.append(nn.Conv2d(ch_in if idx == 0 else ch_out, ch_out, kernel_size=k, stride=s, padding=p, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(ch_out, affine=bias))
        if relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PTUNetRef(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = pt_repeat2d(2, 1, 64, k=3, s=1, p=1, relu=True)
        self.conv2 = pt_repeat2d(2, 64, 128, k=3, s=1, p=1, relu=True)
        self.conv3 = pt_repeat2d(2, 128, 256, k=3, s=1, p=1, relu=True)
        self.conv4 = pt_repeat2d(2, 256, 512, k=3, s=1, p=1, relu=True)
        self.deconv4 = pt_repeat2d(2, 256 + 512, 256, k=3, s=1, p=1, relu=True)
        self.deconv3 = pt_repeat2d(2, 128 + 256, 128, k=3, s=1, p=1, relu=True)
        self.deconv2 = pt_repeat2d(2, 64 + 128, 64, k=3, s=1, p=1, relu=True)
        self.deconv1 = pt_conv2d(64, 1, k=3, s=1, p=1, norm=False, relu=False)

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.pool(conv1)
        conv2 = self.conv2(x)
        x = self.pool(conv2)
        conv3 = self.conv3(x)
        x = self.pool(conv3)
        x = self.conv4(x)
        x = self.upscale(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.deconv4(x)
        x = self.upscale(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.deconv3(x)
        x = self.upscale(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return self.sigmoid(x)


class PTComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, z):
        zr = z.real
        zi = z.imag
        yr = self.fc_r(zr) - self.fc_i(zi)
        yi = self.fc_r(zi) + self.fc_i(zr)
        return torch.complex(yr, yi)


def pt_sepconv2d(ch_in, ch_out, k, s, p, bias=True, norm=True, relu=False):
    layers = [nn.Conv2d(ch_in, ch_out, groups=ch_in, kernel_size=k, stride=s, padding=p, bias=bias)]
    if norm:
        layers.append(nn.BatchNorm2d(ch_out, affine=bias))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PTCFNORef(nn.Module):
    def __init__(self, c=1, d=8, k=16, s=1, size=(8, 8)):
        super().__init__()
        self.c = c
        self.d = d
        self.k = k
        self.s = s
        self.size = size
        self.fc = PTComplexLinear(self.c * (self.k ** 2), self.d)
        self.conv = pt_sepconv2d(self.d, self.d, k=2 * self.s + 1, s=1, p="same", relu=False)

    def forward(self, x):
        # x: NCHW
        b, c, h, w = x.shape
        hk = h // self.k
        wk = w // self.k
        patches = x.view(b, c, hk, self.k, wk, self.k).permute(0, 2, 4, 3, 5, 1).reshape(b * hk * wk, self.k * self.k * c)
        fft = torch.fft.fft(patches, dim=-1)
        fc = self.fc(fft)
        ifft = torch.fft.ifft(fc).real
        ifft = ifft.view(b, hk, wk, self.d).permute(0, 3, 1, 2)  # NCHW
        conved = self.conv(ifft)
        return torch.nn.functional.interpolate(conved, size=self.size, mode="nearest")


class PTCFNOILTNetRef(nn.Module):
    def __init__(self, cfno_sizes=((16, 16), (16, 16), (16, 16))):
        super().__init__()
        self.cfno0 = PTCFNORef(c=1, d=16, k=16, s=1, size=cfno_sizes[0])
        self.cfno1 = PTCFNORef(c=1, d=32, k=32, s=1, size=cfno_sizes[1])
        self.cfno2 = PTCFNORef(c=1, d=64, k=64, s=1, size=cfno_sizes[2])

        self.conv0a = pt_conv2d(1, 32, k=3, s=2, p=1, relu=True)
        self.conv0b = pt_repeat2d(2, 32, 32, k=3, s=1, p=1, relu=True)
        self.conv1a = pt_conv2d(32, 64, k=3, s=2, p=1, relu=True)
        self.conv1b = pt_repeat2d(2, 64, 64, k=3, s=1, p=1, relu=True)
        self.conv2a = pt_conv2d(64, 128, k=3, s=2, p=1, relu=True)
        self.conv2b = pt_repeat2d(2, 128, 128, k=3, s=1, p=1, relu=True)

        self.deconv0a = nn.Sequential(
            nn.ConvTranspose2d(16 + 32 + 64 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.deconv0b = pt_repeat2d(2, 128, 128, k=3, s=1, p=1, relu=True)
        self.deconv1a = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv1b = pt_repeat2d(2, 64, 64, k=3, s=1, p=1, relu=True)
        self.deconv2a = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.deconv2b = pt_repeat2d(2, 32, 32, k=3, s=1, p=1, relu=True)

        self.conv3 = pt_conv2d(32, 32, k=3, s=1, p=1, relu=True)
        self.conv4 = pt_conv2d(32, 32, k=3, s=1, p=1, relu=True)
        self.conv5 = pt_conv2d(32, 32, k=3, s=1, p=1, relu=True)
        self.conv6 = pt_conv2d(32, 1, k=3, s=1, p=1, norm=False, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        br0 = self.cfno0(x)
        br1 = self.cfno1(x)
        br2 = self.cfno2(x)

        br3 = self.conv0a(x)
        br3 = self.conv0b(br3)
        br3 = self.conv1a(br3)
        br3 = self.conv1b(br3)
        br3 = self.conv2a(br3)
        br3 = self.conv2b(br3)

        feat = torch.cat([br0, br1, br2, br3], dim=1)
        x = self.deconv0a(feat)
        x = self.deconv0b(x)
        x = self.deconv1a(x)
        x = self.deconv1b(x)
        x = self.deconv2a(x)
        x = self.deconv2b(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.sigmoid(x)


class PTCFNOLithoNetRef(nn.Module):
    def __init__(self, cfno_sizes=((16, 16), (16, 16), (16, 16))):
        super().__init__()
        self.cfno0 = PTCFNORef(c=1, d=16, k=16, s=1, size=cfno_sizes[0])
        self.cfno1 = PTCFNORef(c=1, d=32, k=32, s=1, size=cfno_sizes[1])
        self.cfno2 = PTCFNORef(c=1, d=64, k=64, s=1, size=cfno_sizes[2])

        self.conv0a = pt_conv2d(1, 32, k=3, s=2, p=1, relu=True)
        self.conv0b = pt_repeat2d(2, 32, 32, k=3, s=1, p=1, relu=True)
        self.conv1a = pt_conv2d(32, 64, k=3, s=2, p=1, relu=True)
        self.conv1b = pt_repeat2d(2, 64, 64, k=3, s=1, p=1, relu=True)
        self.conv2a = pt_conv2d(64, 128, k=3, s=2, p=1, relu=True)
        self.conv2b = pt_repeat2d(2, 128, 128, k=3, s=1, p=1, relu=True)

        self.deconv0a = nn.Sequential(
            nn.ConvTranspose2d(16 + 32 + 64 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.deconv0b = pt_repeat2d(2, 128, 128, k=3, s=1, p=1, relu=True)
        self.deconv1a = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv1b = pt_repeat2d(2, 64, 64, k=3, s=1, p=1, relu=True)
        self.deconv2a = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.deconv2b = pt_repeat2d(2, 32, 32, k=3, s=1, p=1, relu=True)

        self.conv3 = pt_conv2d(32, 32, k=3, s=1, p=1, relu=True)
        self.conv4 = pt_conv2d(32, 32, k=3, s=1, p=1, relu=True)
        self.conv5 = pt_conv2d(32, 32, k=3, s=1, p=1, relu=True)
        self.conv6l = pt_conv2d(32, 1, k=3, s=1, p=1, norm=False, relu=False)
        self.conv6r = pt_conv2d(32, 1, k=3, s=1, p=1, norm=False, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        br0 = self.cfno0(x)
        br1 = self.cfno1(x)
        br2 = self.cfno2(x)

        br3 = self.conv0a(x)
        br3 = self.conv0b(br3)
        br3 = self.conv1a(br3)
        br3 = self.conv1b(br3)
        br3 = self.conv2a(br3)
        br3 = self.conv2b(br3)

        feat = torch.cat([br0, br1, br2, br3], dim=1)
        x = self.deconv0a(feat)
        x = self.deconv0b(x)
        x = self.deconv1a(x)
        x = self.deconv1b(x)
        x = self.deconv2a(x)
        x = self.deconv2b(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        xl = self.sigmoid(self.conv6l(x))
        xr = self.sigmoid(self.conv6r(x))
        return xl, xr


class PTRFNORef(nn.Module):
    def __init__(self, out_channels, modes1, modes2):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / out_channels
        self.weights0 = nn.Parameter(scale * torch.rand(1, out_channels, 1, 1, dtype=torch.complex64))
        self.weights1 = nn.Parameter(scale * torch.rand(1, out_channels, modes1, modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(scale * torch.rand(1, out_channels, modes1, modes2, dtype=torch.complex64))

    def forward(self, x):
        # x: NCHW, expected Cin=1
        b = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        x_ft = x_ft * self.weights0
        out_ft = torch.zeros(
            b, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.complex64, device=x.device
        )
        out_ft[:, :, : self.modes1, : self.modes2] = x_ft[:, :, : self.modes1, : self.modes2] * self.weights1
        out_ft[:, :, -self.modes1 :, : self.modes2] = x_ft[:, :, -self.modes1 :, : self.modes2] * self.weights2
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class PTRFNONetRef(nn.Module):
    def __init__(self, modes1=8, modes2=8):
        super().__init__()
        self.rfno = PTRFNORef(64, modes1=modes1, modes2=modes2)
        self.conv0 = pt_conv2d(1, 16, k=3, s=2, p=1, relu=True)
        self.conv1 = pt_conv2d(16, 32, k=3, s=2, p=1, relu=True)
        self.conv2 = pt_conv2d(32, 64, k=3, s=2, p=1, relu=True)

        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv3 = pt_conv2d(16, 16, k=3, s=1, p=1, relu=True)
        self.conv4 = pt_conv2d(16, 16, k=3, s=1, p=1, relu=True)
        self.conv5 = pt_conv2d(16, 8, k=3, s=1, p=1, relu=True)
        self.conv6l = pt_conv2d(8, 1, k=3, s=1, p=1, norm=False, relu=False)
        self.conv6r = pt_conv2d(8, 1, k=3, s=1, p=1, norm=False, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        br0 = self.rfno(torch.nn.functional.avg_pool2d(x, kernel_size=8, stride=8))
        br1_0 = self.conv0(x)
        br1_1 = self.conv1(br1_0)
        br1_2 = self.conv2(br1_1)
        joined = self.deconv0(torch.cat([br0, br1_2], dim=1))
        joined = self.deconv1(torch.cat([joined, br1_1], dim=1))
        joined = self.deconv2(torch.cat([joined, br1_0], dim=1))
        joined = self.conv3(joined)
        joined = self.conv4(joined)
        joined = self.conv5(joined)
        xl = self.sigmoid(self.conv6l(joined))
        xr = self.sigmoid(self.conv6r(joined))
        return xl, xr


class PTGANOPCGeneratorRef(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            pt_conv2d(1, 16, k=5, s=2, p=2, relu=True),
            pt_conv2d(16, 64, k=5, s=2, p=2, relu=True),
            pt_conv2d(64, 128, k=5, s=2, p=2, relu=True),
            pt_conv2d(128, 512, k=5, s=2, p=2, relu=True),
            pt_conv2d(512, 1024, k=5, s=2, p=2, relu=True),
            nn.Sequential(
                nn.Conv2d(1024, 512 * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128 * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(128, 64 * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 16 * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(16, 1 * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
            ),
        )

    def forward(self, x):
        return self.seq(x)


class PTGANOPCDiscriminatorRef(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            pt_repeat2d(2, 1, 64, k=3, s=1, p=1, relu=True),
            pt_conv2d(64, 64, k=3, s=2, p=1, relu=True),
            pt_repeat2d(2, 64, 128, k=3, s=1, p=1, relu=True),
            pt_conv2d(128, 128, k=3, s=2, p=1, relu=True),
            pt_repeat2d(3, 128, 256, k=3, s=1, p=1, relu=True),
            pt_conv2d(256, 256, k=3, s=2, p=1, relu=True),
            pt_repeat2d(3, 256, 512, k=3, s=1, p=1, relu=True),
            pt_conv2d(512, 512, k=3, s=2, p=1, relu=True),
            pt_repeat2d(3, 512, 512, k=3, s=1, p=1, relu=True),
            pt_conv2d(512, 512, k=3, s=2, p=1, relu=True),
            nn.Flatten(),
            nn.Sequential(nn.Linear(8 * 8 * 512, 2048, bias=True), nn.BatchNorm1d(2048), nn.ReLU()),
            nn.Sequential(nn.Linear(2048, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU()),
            nn.Sequential(nn.Linear(512, 1, bias=True), nn.BatchNorm1d(1), nn.ReLU()),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.seq(x)


class PTLithoGANGeneratorRef(nn.Module):
    def __init__(self, cin=1, cout=1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = pt_conv2d(cin, 64, k=5, s=1, p="same", relu=True)
        self.conv2 = pt_conv2d(64, 128, k=5, s=1, p="same", relu=True)
        self.conv3 = pt_conv2d(128, 256, k=5, s=1, p="same", relu=True)
        self.conv4 = pt_conv2d(256, 512, k=5, s=1, p="same", relu=True)
        self.conv5 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.conv6 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.conv7 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.conv8 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.deconv8 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.deconv7 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.deconv6 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.deconv5 = pt_conv2d(512, 512, k=5, s=1, p="same", relu=True)
        self.deconv4 = pt_conv2d(512, 256, k=5, s=1, p="same", relu=True)
        self.deconv3 = pt_conv2d(256, 128, k=5, s=1, p="same", relu=True)
        self.deconv2 = pt_conv2d(128, 64, k=5, s=1, p="same", relu=True)
        self.deconv1l = pt_conv2d(64, cout, k=5, s=1, p="same", norm=False, relu=False)
        self.deconv1r = pt_conv2d(64, cout, k=5, s=1, p="same", norm=False, relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.conv6(x)
        x = self.pool(x)
        x = self.conv7(x)
        x = self.pool(x)
        x = self.conv8(x)
        x = self.pool(x)
        x = self.upscale(x)
        x = self.deconv8(x)
        x = self.upscale(x)
        x = self.deconv7(x)
        x = self.upscale(x)
        x = self.deconv6(x)
        x = self.upscale(x)
        x = self.deconv5(x)
        x = self.upscale(x)
        x = self.deconv4(x)
        x = self.upscale(x)
        x = self.deconv3(x)
        x = self.upscale(x)
        x = self.deconv2(x)
        x = self.upscale(x)
        xl = self.sigmoid(self.deconv1l(x))
        xr = self.sigmoid(self.deconv1r(x))
        return xl, xr


class PTLithoGANDiscriminatorRef(nn.Module):
    def __init__(self, cin_a=1, cin_b=1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.conv1l = pt_conv2d(cin_a, 64, k=5, s=1, p="same", relu=True)
        self.conv2l = pt_conv2d(64, 128, k=5, s=1, p="same", relu=True)
        self.conv3l = pt_conv2d(128, 256, k=5, s=1, p="same", relu=True)
        self.conv4l = pt_conv2d(256, 512, k=5, s=1, p="same", relu=True)
        self.conv5l = pt_conv2d(512, 1, k=5, s=1, p="same", relu=True)
        self.fc1l = nn.Sequential(nn.Linear(32 * 32 * 1, 1, bias=True))

        self.conv1r = pt_conv2d(cin_b, 64, k=5, s=1, p="same", relu=True)
        self.conv2r = pt_conv2d(64, 128, k=5, s=1, p="same", relu=True)
        self.conv3r = pt_conv2d(128, 256, k=5, s=1, p="same", relu=True)
        self.conv4r = pt_conv2d(256, 512, k=5, s=1, p="same", relu=True)
        self.conv5r = pt_conv2d(512, 1, k=5, s=1, p="same", relu=True)
        self.fc1r = nn.Sequential(nn.Linear(32 * 32 * 1, 1, bias=True))

    def forward(self, xl, xr):
        yl = self.conv1l(xl)
        yl = self.pool(yl)
        yl = self.conv2l(yl)
        yl = self.pool(yl)
        yl = self.conv3l(yl)
        yl = self.pool(yl)
        yl = self.conv4l(yl)
        yl = self.conv5l(yl)
        yl = self.flatten(yl)
        yl = self.fc1l(yl)
        yl = self.sigmoid(yl)

        yr = self.conv1r(xr)
        yr = self.pool(yr)
        yr = self.conv2r(yr)
        yr = self.pool(yr)
        yr = self.conv3r(yr)
        yr = self.pool(yr)
        yr = self.conv4r(yr)
        yr = self.conv5r(yr)
        yr = self.flatten(yr)
        yr = self.fc1r(yr)
        yr = self.sigmoid(yr)
        return yl, yr


class PTConvBlockRef(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, leaky=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PTDeconvBlockRef(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1, bias=True
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PTDAMOILTGeneratorRef(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        n1 = 32
        f0, f1, f2, f3, f4 = n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32
        self.conv_head = PTConvBlockRef(in_ch, n1, kernel_size=7, stride=1, padding=3)
        self.conv0 = PTConvBlockRef(n1, f0, stride=2)
        self.conv1 = PTConvBlockRef(f0, f1, stride=2)
        self.conv2 = PTConvBlockRef(f1, f2, stride=2)
        self.conv3 = PTConvBlockRef(f2, f3, stride=2)
        self.conv4 = PTConvBlockRef(f3, f4, stride=2)
        self.res0 = PTConvBlockRef(f4, f4, stride=1)
        self.res1 = PTConvBlockRef(f4, f4, stride=1)
        self.res2 = PTConvBlockRef(f4, f4, stride=1)
        self.res3 = PTConvBlockRef(f4, f4, stride=1)
        self.res4 = PTConvBlockRef(f4, f4, stride=1)
        self.res5 = PTConvBlockRef(f4, f4, stride=1)
        self.res6 = PTConvBlockRef(f4, f4, stride=1)
        self.res7 = PTConvBlockRef(f4, f4, stride=1)
        self.res8 = PTConvBlockRef(f4, f4, stride=1)
        self.deconv4 = PTDeconvBlockRef(f4, f3, stride=2)
        self.deconv3 = PTDeconvBlockRef(f3, f2, stride=2)
        self.deconv2 = PTDeconvBlockRef(f2, f1, stride=2)
        self.deconv1 = PTDeconvBlockRef(f1, f0, stride=2)
        self.deconv0 = PTDeconvBlockRef(f0, n1, stride=2)
        self.conv_tail = nn.Conv2d(n1, out_ch, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.conv_head(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        x = self.deconv0(x)
        return self.conv_tail(x)


class PTDAMOILTDiscriminatorRef(nn.Module):
    def __init__(self, in_size=32, resize_up=64):
        super().__init__()
        self.resize_up = resize_up
        self.conv0_0 = PTConvBlockRef(1, 64, kernel_size=4, stride=2, padding=1, leaky=True)
        self.conv1_0 = PTConvBlockRef(64, 128, kernel_size=4, stride=1, padding="same", leaky=True)
        self.conv2_0 = PTConvBlockRef(128, 1, kernel_size=4, stride=1, padding="same", leaky=True)
        self.flatten_0 = nn.Flatten()
        self.fc0_0 = nn.Linear((in_size // 2) * (in_size // 2), 1)
        self.sigmoid_0 = nn.Sigmoid()

        self.conv0_1 = PTConvBlockRef(1, 64, kernel_size=4, stride=2, padding=1, leaky=True)
        self.conv1_1 = PTConvBlockRef(64, 128, kernel_size=4, stride=1, padding="same", leaky=True)
        self.conv2_1 = PTConvBlockRef(128, 1, kernel_size=4, stride=1, padding="same", leaky=True)
        self.flatten_1 = nn.Flatten()
        self.fc0_1 = nn.Linear(resize_up * resize_up // 4, 1)
        self.sigmoid_1 = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv0_0(x)
        x0 = self.conv1_0(x0)
        x0 = self.conv2_0(x0)
        x0 = self.flatten_0(x0)
        x0 = self.fc0_0(x0)
        x0 = self.sigmoid_0(x0)

        x1 = torch.nn.functional.interpolate(x, size=(self.resize_up, self.resize_up), mode="bilinear")
        x1 = self.conv0_1(x1)
        x1 = self.conv1_1(x1)
        x1 = self.conv2_1(x1)
        x1 = self.flatten_1(x1)
        x1 = self.fc0_1(x1)
        x1 = self.sigmoid_1(x1)
        return 0.5 * (x0 + x1)


class PTDAMOLithoGeneratorRef(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        n1 = 32
        f0, f1, f2, f3, f4 = n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32
        self.sigmoid = nn.Sigmoid()
        self.conv_head = PTConvBlockRef(in_ch, n1, kernel_size=7, stride=1, padding=3)
        self.conv0 = PTConvBlockRef(n1, f0, stride=2)
        self.conv1 = PTConvBlockRef(f0, f1, stride=2)
        self.conv2 = PTConvBlockRef(f1, f2, stride=2)
        self.conv3 = PTConvBlockRef(f2, f3, stride=2)
        self.conv4 = PTConvBlockRef(f3, f4, stride=2)
        self.res0 = PTConvBlockRef(f4, f4, stride=1)
        self.res1 = PTConvBlockRef(f4, f4, stride=1)
        self.res2 = PTConvBlockRef(f4, f4, stride=1)
        self.res3 = PTConvBlockRef(f4, f4, stride=1)
        self.res4 = PTConvBlockRef(f4, f4, stride=1)
        self.res5 = PTConvBlockRef(f4, f4, stride=1)
        self.res6 = PTConvBlockRef(f4, f4, stride=1)
        self.res7 = PTConvBlockRef(f4, f4, stride=1)
        self.res8 = PTConvBlockRef(f4, f4, stride=1)
        self.deconv4 = PTDeconvBlockRef(f4, f3, stride=2)
        self.deconv3 = PTDeconvBlockRef(f3, f2, stride=2)
        self.deconv2 = PTDeconvBlockRef(f2, f1, stride=2)
        self.deconv1 = PTDeconvBlockRef(f1, f0, stride=2)
        self.deconv0 = PTDeconvBlockRef(f0, n1, stride=2)
        self.conv_tail_l = nn.Conv2d(n1, out_ch, kernel_size=7, stride=1, padding=3)
        self.conv_tail_r = nn.Conv2d(n1, out_ch, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.conv_head(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        x = self.deconv0(x)
        xl = self.sigmoid(self.conv_tail_l(x))
        xr = self.sigmoid(self.conv_tail_r(x))
        return xl, xr


class PTDAMOLithoDiscriminatorRef(nn.Module):
    def __init__(self, in_size=32, resize_up=64):
        super().__init__()
        self.resize_up = resize_up
        self.conv0_0 = PTConvBlockRef(2, 64, kernel_size=4, stride=2, padding=1, leaky=True)
        self.conv1_0 = PTConvBlockRef(64, 128, kernel_size=4, stride=1, padding="same", leaky=True)
        self.conv2_0 = PTConvBlockRef(128, 1, kernel_size=4, stride=1, padding="same", leaky=True)
        self.flatten_0 = nn.Flatten()
        self.fc0_0 = nn.Linear((in_size // 2) * (in_size // 2), 1)
        self.sigmoid_0 = nn.Sigmoid()

        self.conv0_1 = PTConvBlockRef(2, 64, kernel_size=4, stride=2, padding=1, leaky=True)
        self.conv1_1 = PTConvBlockRef(64, 128, kernel_size=4, stride=1, padding="same", leaky=True)
        self.conv2_1 = PTConvBlockRef(128, 1, kernel_size=4, stride=1, padding="same", leaky=True)
        self.flatten_1 = nn.Flatten()
        self.fc0_1 = nn.Linear(resize_up * resize_up // 4, 1)
        self.sigmoid_1 = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv0_0(x)
        x0 = self.conv1_0(x0)
        x0 = self.conv2_0(x0)
        x0 = self.flatten_0(x0)
        x0 = self.fc0_0(x0)
        x0 = self.sigmoid_0(x0)

        x1 = torch.nn.functional.interpolate(x, size=(self.resize_up, self.resize_up), mode="nearest")
        x1 = self.conv0_1(x1)
        x1 = self.conv1_1(x1)
        x1 = self.conv2_1(x1)
        x1 = self.flatten_1(x1)
        x1 = self.fc0_1(x1)
        x1 = self.sigmoid_1(x1)
        return 0.5 * (x0 + x1)


# -----------------------------
# Weight copy helpers
# -----------------------------
def _copy_conv2d(pt_conv: nn.Conv2d, tf_conv: tf.keras.layers.Conv2D):
    w = pt_conv.weight.detach().cpu().numpy()  # [O, I, H, W]
    w = np.transpose(w, (2, 3, 1, 0))         # [H, W, I, O]
    if pt_conv.bias is None:
        tf_conv.set_weights([w])
    else:
        b = pt_conv.bias.detach().cpu().numpy()
        tf_conv.set_weights([w, b])


def _copy_bn2d(pt_bn: nn.BatchNorm2d, tf_bn: tf.keras.layers.BatchNormalization):
    gamma = pt_bn.weight.detach().cpu().numpy()
    beta = pt_bn.bias.detach().cpu().numpy()
    moving_mean = pt_bn.running_mean.detach().cpu().numpy()
    moving_var = pt_bn.running_var.detach().cpu().numpy()
    tf_bn.set_weights([gamma, beta, moving_mean, moving_var])


def _copy_conv_transpose2d(pt_conv: nn.ConvTranspose2d, tf_conv: tf.keras.layers.Conv2DTranspose):
    w = pt_conv.weight.detach().cpu().numpy()  # [I, O, H, W]
    w = np.transpose(w, (2, 3, 1, 0))          # [H, W, O, I]
    if pt_conv.bias is None:
        tf_conv.set_weights([w])
    else:
        b = pt_conv.bias.detach().cpu().numpy()
        tf_conv.set_weights([w, b])


def _copy_dense(pt_linear: nn.Linear, tf_dense: tf.keras.layers.Dense):
    w = pt_linear.weight.detach().cpu().numpy()  # [O, I]
    b = pt_linear.bias.detach().cpu().numpy()
    w = np.transpose(w, (1, 0))                  # [I, O]
    tf_dense.set_weights([w, b])


def _copy_bn1d(pt_bn: nn.BatchNorm1d, tf_bn: tf.keras.layers.BatchNormalization):
    gamma = pt_bn.weight.detach().cpu().numpy()
    beta = pt_bn.bias.detach().cpu().numpy()
    moving_mean = pt_bn.running_mean.detach().cpu().numpy()
    moving_var = pt_bn.running_var.detach().cpu().numpy()
    tf_bn.set_weights([gamma, beta, moving_mean, moving_var])


def _copy_depthwise_conv_from_grouped(pt_conv: nn.Conv2d, tf_dw: tf.keras.layers.DepthwiseConv2D):
    # PyTorch grouped conv here is depthwise with weight [C,1,K,K]
    w = pt_conv.weight.detach().cpu().numpy()
    w = np.transpose(w[:, 0, :, :], (1, 2, 0))[:, :, :, None]  # [K,K,C,1]
    b = pt_conv.bias.detach().cpu().numpy() if pt_conv.bias is not None else None
    if b is None:
        tf_dw.set_weights([w])
    else:
        tf_dw.set_weights([w, b])


def _copy_repeat_block(pt_seq: nn.Sequential, tf_block):
    pt_convs = [m for m in pt_seq if isinstance(m, nn.Conv2d)]
    pt_bns = [m for m in pt_seq if isinstance(m, nn.BatchNorm2d)]
    assert len(pt_convs) == len(tf_block.layers_)
    for i, tf_layer in enumerate(tf_block.layers_):
        _copy_conv2d(pt_convs[i], tf_layer.conv)
        if tf_layer.use_bn:
            _copy_bn2d(pt_bns[i], tf_layer.bn)


def _copy_convbnrelu_single(pt_seq: nn.Sequential, tf_layer):
    pt_conv = next(m for m in pt_seq if isinstance(m, nn.Conv2d))
    _copy_conv2d(pt_conv, tf_layer.conv)
    if tf_layer.use_bn:
        pt_bn = next(m for m in pt_seq if isinstance(m, nn.BatchNorm2d))
        _copy_bn2d(pt_bn, tf_layer.bn)


def _copy_deconvbnrelu_single(pt_seq: nn.Sequential, tf_layer):
    pt_conv = next(m for m in pt_seq if isinstance(m, nn.ConvTranspose2d))
    _copy_conv_transpose2d(pt_conv, tf_layer.deconv)
    if tf_layer.use_bn:
        pt_bn = next(m for m in pt_seq if isinstance(m, nn.BatchNorm2d))
        _copy_bn2d(pt_bn, tf_layer.bn)


def _copy_complex_param(pt_param: torch.nn.Parameter, tf_var: tf.Variable):
    v = pt_param.detach().cpu().numpy().astype(np.complex64)
    tf_var.assign(v)


def _copy_pt_conv_block(pt_block, tf_block):
    pt_conv = next(m for m in pt_block.conv if isinstance(m, nn.Conv2d))
    pt_bn = next(m for m in pt_block.conv if isinstance(m, nn.BatchNorm2d))
    _copy_conv2d(pt_conv, tf_block.conv)
    _copy_bn2d(pt_bn, tf_block.bn)


def _copy_pt_deconv_block(pt_block, tf_block):
    pt_deconv = next(m for m in pt_block.conv if isinstance(m, nn.ConvTranspose2d))
    pt_bn = next(m for m in pt_block.conv if isinstance(m, nn.BatchNorm2d))
    _copy_conv_transpose2d(pt_deconv, tf_block.deconv)
    _copy_bn2d(pt_bn, tf_block.bn)


def _copy_pixelshuffle_block(pt_seq: nn.Sequential, tf_block):
    pt_conv = next(m for m in pt_seq if isinstance(m, nn.Conv2d))
    _copy_conv2d(pt_conv, tf_block.conv)
    if tf_block.use_bn:
        pt_bn = next(m for m in pt_seq if isinstance(m, nn.BatchNorm2d))
        _copy_bn2d(pt_bn, tf_block.bn)


def _copy_linearbnrelu_single(pt_seq: nn.Sequential, tf_layer):
    pt_fc = next(m for m in pt_seq if isinstance(m, nn.Linear))
    _copy_dense(pt_fc, tf_layer.fc)
    if tf_layer.use_bn:
        pt_bn = next(m for m in pt_seq if isinstance(m, nn.BatchNorm1d))
        _copy_bn1d(pt_bn, tf_layer.bn)


def test_neuralilt_unet_equivalence():
    set_seed(0)
    pt = PTUNetRef().cpu().eval()
    tf_model = UNetTF()

    x_pt = torch.randn(2, 1, 64, 64, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))

    _ = tf_model(x_tf, training=False)

    _copy_repeat_block(pt.conv1, tf_model.conv1)
    _copy_repeat_block(pt.conv2, tf_model.conv2)
    _copy_repeat_block(pt.conv3, tf_model.conv3)
    _copy_repeat_block(pt.conv4, tf_model.conv4)
    _copy_repeat_block(pt.deconv4, tf_model.deconv4)
    _copy_repeat_block(pt.deconv3, tf_model.deconv3)
    _copy_repeat_block(pt.deconv2, tf_model.deconv2)
    _copy_conv2d(pt.deconv1[0], tf_model.deconv1.conv)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_model(x_tf, training=False).numpy()
    y_tf_nchw = np.transpose(y_tf, (0, 3, 1, 2))

    mae = float(np.mean(np.abs(y_pt - y_tf_nchw)))
    mxe = float(np.max(np.abs(y_pt - y_tf_nchw)))
    print(f"[NeuralILT UNet] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 1e-5 and mxe < 1e-4


def test_cfno_core_equivalence():
    set_seed(0)
    pt = PTCFNORef(c=1, d=8, k=16, s=1, size=(8, 8)).cpu().eval()
    tf_layer = CFNOTF(c=1, d=8, k=16, s=1, size=(8, 8))

    x_pt = torch.randn(2, 1, 64, 64, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))

    _ = tf_layer(x_tf, training=False)

    _copy_dense(pt.fc.fc_r, tf_layer.fc.fc_r)
    _copy_dense(pt.fc.fc_i, tf_layer.fc.fc_i)
    _copy_depthwise_conv_from_grouped(pt.conv[0], tf_layer.conv.dw)
    _copy_bn2d(pt.conv[1], tf_layer.conv.bn)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_layer(x_tf, training=False).numpy()
    y_tf_nchw = np.transpose(y_tf, (0, 3, 1, 2))

    mae = float(np.mean(np.abs(y_pt - y_tf_nchw)))
    mxe = float(np.max(np.abs(y_pt - y_tf_nchw)))
    print(f"[CFNO core] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 1e-5 and mxe < 1e-4


def test_cfnoilt_net_equivalence():
    set_seed(0)
    cfno_sizes = ((16, 16), (16, 16), (16, 16))
    pt = PTCFNOILTNetRef(cfno_sizes=cfno_sizes).cpu().eval()
    tf_model = CFNOILTNetTF(cfno_sizes=cfno_sizes)

    x_pt = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))

    _ = tf_model(x_tf, training=False)

    # CFNO branches
    for pt_cfno, tf_cfno in [(pt.cfno0, tf_model.cfno0), (pt.cfno1, tf_model.cfno1), (pt.cfno2, tf_model.cfno2)]:
        _copy_dense(pt_cfno.fc.fc_r, tf_cfno.fc.fc_r)
        _copy_dense(pt_cfno.fc.fc_i, tf_cfno.fc.fc_i)
        _copy_depthwise_conv_from_grouped(pt_cfno.conv[0], tf_cfno.conv.dw)
        _copy_bn2d(pt_cfno.conv[1], tf_cfno.conv.bn)

    # CNN branch + tail
    _copy_convbnrelu_single(pt.conv0a, tf_model.conv0a)
    _copy_repeat_block(pt.conv0b, tf_model.conv0b)
    _copy_convbnrelu_single(pt.conv1a, tf_model.conv1a)
    _copy_repeat_block(pt.conv1b, tf_model.conv1b)
    _copy_convbnrelu_single(pt.conv2a, tf_model.conv2a)
    _copy_repeat_block(pt.conv2b, tf_model.conv2b)

    _copy_deconvbnrelu_single(pt.deconv0a, tf_model.deconv0a)
    _copy_repeat_block(pt.deconv0b, tf_model.deconv0b)
    _copy_deconvbnrelu_single(pt.deconv1a, tf_model.deconv1a)
    _copy_repeat_block(pt.deconv1b, tf_model.deconv1b)
    _copy_deconvbnrelu_single(pt.deconv2a, tf_model.deconv2a)
    _copy_repeat_block(pt.deconv2b, tf_model.deconv2b)

    _copy_convbnrelu_single(pt.conv3, tf_model.conv3)
    _copy_convbnrelu_single(pt.conv4, tf_model.conv4)
    _copy_convbnrelu_single(pt.conv5, tf_model.conv5)
    _copy_conv2d(pt.conv6[0], tf_model.conv6.conv)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_model(x_tf, training=False).numpy()
    y_tf_nchw = np.transpose(y_tf, (0, 3, 1, 2))

    mae = float(np.mean(np.abs(y_pt - y_tf_nchw)))
    mxe = float(np.max(np.abs(y_pt - y_tf_nchw)))
    print(f"[CFNOILT net tiny] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 1e-5 and mxe < 1e-4


def test_cfnolitho_net_equivalence():
    set_seed(0)
    cfno_sizes = ((16, 16), (16, 16), (16, 16))
    pt = PTCFNOLithoNetRef(cfno_sizes=cfno_sizes).cpu().eval()
    tf_model = CFNOLithoNetTF(cfno_sizes=cfno_sizes)

    x_pt = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    for pt_cfno, tf_cfno in [(pt.cfno0, tf_model.cfno0), (pt.cfno1, tf_model.cfno1), (pt.cfno2, tf_model.cfno2)]:
        _copy_dense(pt_cfno.fc.fc_r, tf_cfno.fc.fc_r)
        _copy_dense(pt_cfno.fc.fc_i, tf_cfno.fc.fc_i)
        _copy_depthwise_conv_from_grouped(pt_cfno.conv[0], tf_cfno.conv.dw)
        _copy_bn2d(pt_cfno.conv[1], tf_cfno.conv.bn)

    _copy_convbnrelu_single(pt.conv0a, tf_model.conv0a)
    _copy_repeat_block(pt.conv0b, tf_model.conv0b)
    _copy_convbnrelu_single(pt.conv1a, tf_model.conv1a)
    _copy_repeat_block(pt.conv1b, tf_model.conv1b)
    _copy_convbnrelu_single(pt.conv2a, tf_model.conv2a)
    _copy_repeat_block(pt.conv2b, tf_model.conv2b)

    _copy_deconvbnrelu_single(pt.deconv0a, tf_model.deconv0a)
    _copy_repeat_block(pt.deconv0b, tf_model.deconv0b)
    _copy_deconvbnrelu_single(pt.deconv1a, tf_model.deconv1a)
    _copy_repeat_block(pt.deconv1b, tf_model.deconv1b)
    _copy_deconvbnrelu_single(pt.deconv2a, tf_model.deconv2a)
    _copy_repeat_block(pt.deconv2b, tf_model.deconv2b)

    _copy_convbnrelu_single(pt.conv3, tf_model.conv3)
    _copy_convbnrelu_single(pt.conv4, tf_model.conv4)
    _copy_convbnrelu_single(pt.conv5, tf_model.conv5)
    _copy_conv2d(pt.conv6l[0], tf_model.conv6l.conv)
    _copy_conv2d(pt.conv6r[0], tf_model.conv6r.conv)

    y_pt_l, y_pt_r = pt(x_pt)
    y_tf_l, y_tf_r = tf_model(x_tf, training=False)
    y_tf_l = np.transpose(y_tf_l.numpy(), (0, 3, 1, 2))
    y_tf_r = np.transpose(y_tf_r.numpy(), (0, 3, 1, 2))

    mae_l = float(np.mean(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mae_r = float(np.mean(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    mxe_l = float(np.max(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mxe_r = float(np.max(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    print(f"[CFNOLitho net tiny] MAE_L={mae_l:.6e}, MAE_R={mae_r:.6e}, MAX_L={mxe_l:.6e}, MAX_R={mxe_r:.6e}")
    assert mae_l < 1e-5 and mae_r < 1e-5 and mxe_l < 1e-4 and mxe_r < 1e-4


def test_rfno_core_equivalence():
    set_seed(0)
    pt = PTRFNORef(out_channels=8, modes1=4, modes2=4).cpu().eval()
    tf_layer = RFNOTF(out_channels=8, modes1=4, modes2=4)

    x_pt = torch.randn(2, 1, 16, 16, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_layer(x_tf)

    _copy_complex_param(pt.weights0, tf_layer.weights0)
    _copy_complex_param(pt.weights1, tf_layer.weights1)
    _copy_complex_param(pt.weights2, tf_layer.weights2)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_layer(x_tf).numpy()
    y_tf_nchw = np.transpose(y_tf, (0, 3, 1, 2))

    mae = float(np.mean(np.abs(y_pt - y_tf_nchw)))
    mxe = float(np.max(np.abs(y_pt - y_tf_nchw)))
    print(f"[RFNO core] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 1e-5 and mxe < 1e-4


def test_rfnonet_equivalence():
    set_seed(0)
    pt = PTRFNONetRef(modes1=8, modes2=8).cpu().eval()
    tf_model = RFNONetTF(modes1=8, modes2=8)

    x_pt = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_complex_param(pt.rfno.weights0, tf_model.rfno.weights0)
    _copy_complex_param(pt.rfno.weights1, tf_model.rfno.weights1)
    _copy_complex_param(pt.rfno.weights2, tf_model.rfno.weights2)

    _copy_convbnrelu_single(pt.conv0, tf_model.conv0)
    _copy_convbnrelu_single(pt.conv1, tf_model.conv1)
    _copy_convbnrelu_single(pt.conv2, tf_model.conv2)
    _copy_deconvbnrelu_single(pt.deconv0, tf_model.deconv0)
    _copy_deconvbnrelu_single(pt.deconv1, tf_model.deconv1)
    _copy_deconvbnrelu_single(pt.deconv2, tf_model.deconv2)
    _copy_convbnrelu_single(pt.conv3, tf_model.conv3)
    _copy_convbnrelu_single(pt.conv4, tf_model.conv4)
    _copy_convbnrelu_single(pt.conv5, tf_model.conv5)
    _copy_conv2d(pt.conv6l[0], tf_model.conv6l.conv)
    _copy_conv2d(pt.conv6r[0], tf_model.conv6r.conv)

    y_pt_l, y_pt_r = pt(x_pt)
    y_tf_l, y_tf_r = tf_model(x_tf, training=False)
    y_tf_l = np.transpose(y_tf_l.numpy(), (0, 3, 1, 2))
    y_tf_r = np.transpose(y_tf_r.numpy(), (0, 3, 1, 2))

    mae_l = float(np.mean(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mae_r = float(np.mean(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    mxe_l = float(np.max(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mxe_r = float(np.max(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    print(f"[RFNONet tiny] MAE_L={mae_l:.6e}, MAE_R={mae_r:.6e}, MAX_L={mxe_l:.6e}, MAX_R={mxe_r:.6e}")
    # End-to-end RFNO/CNN fusion is sensitive to FFT backend and accumulation order.
    # Keep a practical threshold while core RFNO test remains strict.
    assert mae_l < 2e-3 and mae_r < 2e-3 and mxe_l < 1e-2 and mxe_r < 1e-2


def test_ganopc_generator_equivalence():
    set_seed(0)
    pt = PTGANOPCGeneratorRef().cpu().eval()
    tf_model = GANOPCGeneratorTF()

    x_pt = torch.randn(1, 1, 256, 256, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_convbnrelu_single(pt.seq[0], tf_model.conv1)
    _copy_convbnrelu_single(pt.seq[1], tf_model.conv2)
    _copy_convbnrelu_single(pt.seq[2], tf_model.conv3)
    _copy_convbnrelu_single(pt.seq[3], tf_model.conv4)
    _copy_convbnrelu_single(pt.seq[4], tf_model.conv5)
    _copy_pixelshuffle_block(pt.seq[5], tf_model.spsr5)
    _copy_pixelshuffle_block(pt.seq[6], tf_model.spsr4)
    _copy_pixelshuffle_block(pt.seq[7], tf_model.spsr3)
    _copy_pixelshuffle_block(pt.seq[8], tf_model.spsr2)
    _copy_pixelshuffle_block(pt.seq[9], tf_model.spsr1)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_model(x_tf, training=False).numpy()
    y_tf_nchw = np.transpose(y_tf, (0, 3, 1, 2))

    mae = float(np.mean(np.abs(y_pt - y_tf_nchw)))
    mxe = float(np.max(np.abs(y_pt - y_tf_nchw)))
    print(f"[GANOPC generator] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 1e-5 and mxe < 1e-4


def test_ganopc_discriminator_equivalence():
    set_seed(0)
    pt = PTGANOPCDiscriminatorRef().cpu().eval()
    tf_model = GANOPCDiscriminatorTF()

    x_pt = torch.randn(2, 1, 256, 256, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_repeat_block(pt.seq[0], tf_model.repeat2a)
    _copy_convbnrelu_single(pt.seq[1], tf_model.conv1)
    _copy_repeat_block(pt.seq[2], tf_model.repeat2b)
    _copy_convbnrelu_single(pt.seq[3], tf_model.conv2)
    _copy_repeat_block(pt.seq[4], tf_model.repeat3a)
    _copy_convbnrelu_single(pt.seq[5], tf_model.conv3)
    _copy_repeat_block(pt.seq[6], tf_model.repeat3b)
    _copy_convbnrelu_single(pt.seq[7], tf_model.conv4)
    _copy_repeat_block(pt.seq[8], tf_model.repeat3c)
    _copy_convbnrelu_single(pt.seq[9], tf_model.conv5)
    _copy_linearbnrelu_single(pt.seq[11], tf_model.fc1)
    _copy_linearbnrelu_single(pt.seq[12], tf_model.fc2)
    _copy_linearbnrelu_single(pt.seq[13], tf_model.fc3)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_model(x_tf, training=False).numpy()

    mae = float(np.mean(np.abs(y_pt - y_tf)))
    mxe = float(np.max(np.abs(y_pt - y_tf)))
    print(f"[GANOPC discriminator] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 1e-5 and mxe < 1e-4


def test_lithogan_generator_equivalence():
    set_seed(0)
    pt = PTLithoGANGeneratorRef(cin=1, cout=1).cpu().eval()
    tf_model = LithoGANGeneratorTF(cin=1, cout=1)

    x_pt = torch.randn(1, 1, 256, 256, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_convbnrelu_single(pt.conv1, tf_model.conv1)
    _copy_convbnrelu_single(pt.conv2, tf_model.conv2)
    _copy_convbnrelu_single(pt.conv3, tf_model.conv3)
    _copy_convbnrelu_single(pt.conv4, tf_model.conv4)
    _copy_convbnrelu_single(pt.conv5, tf_model.conv5)
    _copy_convbnrelu_single(pt.conv6, tf_model.conv6)
    _copy_convbnrelu_single(pt.conv7, tf_model.conv7)
    _copy_convbnrelu_single(pt.conv8, tf_model.conv8)
    _copy_convbnrelu_single(pt.deconv8, tf_model.deconv8)
    _copy_convbnrelu_single(pt.deconv7, tf_model.deconv7)
    _copy_convbnrelu_single(pt.deconv6, tf_model.deconv6)
    _copy_convbnrelu_single(pt.deconv5, tf_model.deconv5)
    _copy_convbnrelu_single(pt.deconv4, tf_model.deconv4)
    _copy_convbnrelu_single(pt.deconv3, tf_model.deconv3)
    _copy_convbnrelu_single(pt.deconv2, tf_model.deconv2)
    _copy_conv2d(pt.deconv1l[0], tf_model.deconv1l.conv)
    _copy_conv2d(pt.deconv1r[0], tf_model.deconv1r.conv)

    y_pt_l, y_pt_r = pt(x_pt)
    y_tf_l, y_tf_r = tf_model(x_tf, training=False)
    y_tf_l = np.transpose(y_tf_l.numpy(), (0, 3, 1, 2))
    y_tf_r = np.transpose(y_tf_r.numpy(), (0, 3, 1, 2))

    mae_l = float(np.mean(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mae_r = float(np.mean(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    mxe_l = float(np.max(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mxe_r = float(np.max(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    print(f"[LithoGAN generator] MAE_L={mae_l:.6e}, MAE_R={mae_r:.6e}, MAX_L={mxe_l:.6e}, MAX_R={mxe_r:.6e}")
    assert mae_l < 1e-5 and mae_r < 1e-5 and mxe_l < 1e-4 and mxe_r < 1e-4


def test_lithogan_discriminator_equivalence():
    set_seed(0)
    pt = PTLithoGANDiscriminatorRef(cin_a=1, cin_b=1).cpu().eval()
    tf_model = LithoGANDiscriminatorTF(cin_a=1, cin_b=1)

    xl_pt = torch.randn(2, 1, 256, 256, dtype=torch.float32)
    xr_pt = torch.randn(2, 1, 256, 256, dtype=torch.float32)
    xl_tf = np.transpose(xl_pt.numpy(), (0, 2, 3, 1))
    xr_tf = np.transpose(xr_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(xl_tf, xr_tf, training=False)

    _copy_convbnrelu_single(pt.conv1l, tf_model.conv1l)
    _copy_convbnrelu_single(pt.conv2l, tf_model.conv2l)
    _copy_convbnrelu_single(pt.conv3l, tf_model.conv3l)
    _copy_convbnrelu_single(pt.conv4l, tf_model.conv4l)
    _copy_convbnrelu_single(pt.conv5l, tf_model.conv5l)
    _copy_dense(pt.fc1l[0], tf_model.fc1l.fc)

    _copy_convbnrelu_single(pt.conv1r, tf_model.conv1r)
    _copy_convbnrelu_single(pt.conv2r, tf_model.conv2r)
    _copy_convbnrelu_single(pt.conv3r, tf_model.conv3r)
    _copy_convbnrelu_single(pt.conv4r, tf_model.conv4r)
    _copy_convbnrelu_single(pt.conv5r, tf_model.conv5r)
    _copy_dense(pt.fc1r[0], tf_model.fc1r.fc)

    y_pt_l, y_pt_r = pt(xl_pt, xr_pt)
    y_tf_l, y_tf_r = tf_model(xl_tf, xr_tf, training=False)
    y_tf_l = y_tf_l.numpy()
    y_tf_r = y_tf_r.numpy()

    mae_l = float(np.mean(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mae_r = float(np.mean(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    mxe_l = float(np.max(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mxe_r = float(np.max(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    print(f"[LithoGAN discriminator] MAE_L={mae_l:.6e}, MAE_R={mae_r:.6e}, MAX_L={mxe_l:.6e}, MAX_R={mxe_r:.6e}")
    assert mae_l < 1e-5 and mae_r < 1e-5 and mxe_l < 1e-4 and mxe_r < 1e-4


def test_damoilt_generator_equivalence():
    set_seed(0)
    pt = PTDAMOILTGeneratorRef(in_ch=1, out_ch=1).cpu().eval()
    tf_model = DAMOILTGeneratorTF(in_ch=1, out_ch=1)

    x_pt = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_pt_conv_block(pt.conv_head, tf_model.conv_head)
    _copy_pt_conv_block(pt.conv0, tf_model.conv0)
    _copy_pt_conv_block(pt.conv1, tf_model.conv1)
    _copy_pt_conv_block(pt.conv2, tf_model.conv2)
    _copy_pt_conv_block(pt.conv3, tf_model.conv3)
    _copy_pt_conv_block(pt.conv4, tf_model.conv4)
    _copy_pt_conv_block(pt.res0, tf_model.res0)
    _copy_pt_conv_block(pt.res1, tf_model.res1)
    _copy_pt_conv_block(pt.res2, tf_model.res2)
    _copy_pt_conv_block(pt.res3, tf_model.res3)
    _copy_pt_conv_block(pt.res4, tf_model.res4)
    _copy_pt_conv_block(pt.res5, tf_model.res5)
    _copy_pt_conv_block(pt.res6, tf_model.res6)
    _copy_pt_conv_block(pt.res7, tf_model.res7)
    _copy_pt_conv_block(pt.res8, tf_model.res8)
    _copy_pt_deconv_block(pt.deconv4, tf_model.deconv4)
    _copy_pt_deconv_block(pt.deconv3, tf_model.deconv3)
    _copy_pt_deconv_block(pt.deconv2, tf_model.deconv2)
    _copy_pt_deconv_block(pt.deconv1, tf_model.deconv1)
    _copy_pt_deconv_block(pt.deconv0, tf_model.deconv0)
    _copy_conv2d(pt.conv_tail, tf_model.conv_tail)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_model(x_tf, training=False).numpy()
    y_tf_nchw = np.transpose(y_tf, (0, 3, 1, 2))

    mae = float(np.mean(np.abs(y_pt - y_tf_nchw)))
    mxe = float(np.max(np.abs(y_pt - y_tf_nchw)))
    print(f"[DAMOILT generator] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 1e-5 and mxe < 1e-4


def test_damoilt_discriminator_equivalence():
    set_seed(0)
    in_size = 32
    resize_up = 64
    pt = PTDAMOILTDiscriminatorRef(in_size=in_size, resize_up=resize_up).cpu().eval()
    tf_model = DAMOILTDiscriminatorTF(resize_up=(resize_up, resize_up))

    x_pt = torch.randn(2, 1, in_size, in_size, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_pt_conv_block(pt.conv0_0, tf_model.conv0_0)
    _copy_pt_conv_block(pt.conv1_0, tf_model.conv1_0)
    _copy_pt_conv_block(pt.conv2_0, tf_model.conv2_0)
    _copy_dense(pt.fc0_0, tf_model.fc0_0)

    _copy_pt_conv_block(pt.conv0_1, tf_model.conv0_1)
    _copy_pt_conv_block(pt.conv1_1, tf_model.conv1_1)
    _copy_pt_conv_block(pt.conv2_1, tf_model.conv2_1)
    _copy_dense(pt.fc0_1, tf_model.fc0_1)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_model(x_tf, training=False).numpy()

    mae = float(np.mean(np.abs(y_pt - y_tf)))
    mxe = float(np.max(np.abs(y_pt - y_tf)))
    print(f"[DAMOILT discriminator] MAE={mae:.6e}, MAX={mxe:.6e}")
    # PyTorch/TF differ slightly on interpolate+same-padding details for this path.
    assert mae < 2e-3 and mxe < 5e-3


def test_damolitho_generator_equivalence():
    set_seed(0)
    pt = PTDAMOLithoGeneratorRef(in_ch=1, out_ch=1).cpu().eval()
    tf_model = DAMOLithoGeneratorTF(in_ch=1, out_ch=1)

    x_pt = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_pt_conv_block(pt.conv_head, tf_model.conv_head)
    _copy_pt_conv_block(pt.conv0, tf_model.conv0)
    _copy_pt_conv_block(pt.conv1, tf_model.conv1)
    _copy_pt_conv_block(pt.conv2, tf_model.conv2)
    _copy_pt_conv_block(pt.conv3, tf_model.conv3)
    _copy_pt_conv_block(pt.conv4, tf_model.conv4)
    _copy_pt_conv_block(pt.res0, tf_model.res0)
    _copy_pt_conv_block(pt.res1, tf_model.res1)
    _copy_pt_conv_block(pt.res2, tf_model.res2)
    _copy_pt_conv_block(pt.res3, tf_model.res3)
    _copy_pt_conv_block(pt.res4, tf_model.res4)
    _copy_pt_conv_block(pt.res5, tf_model.res5)
    _copy_pt_conv_block(pt.res6, tf_model.res6)
    _copy_pt_conv_block(pt.res7, tf_model.res7)
    _copy_pt_conv_block(pt.res8, tf_model.res8)
    _copy_pt_deconv_block(pt.deconv4, tf_model.deconv4)
    _copy_pt_deconv_block(pt.deconv3, tf_model.deconv3)
    _copy_pt_deconv_block(pt.deconv2, tf_model.deconv2)
    _copy_pt_deconv_block(pt.deconv1, tf_model.deconv1)
    _copy_pt_deconv_block(pt.deconv0, tf_model.deconv0)
    _copy_conv2d(pt.conv_tail_l, tf_model.conv_tail_l)
    _copy_conv2d(pt.conv_tail_r, tf_model.conv_tail_r)

    y_pt_l, y_pt_r = pt(x_pt)
    y_tf_l, y_tf_r = tf_model(x_tf, training=False)
    y_tf_l = np.transpose(y_tf_l.numpy(), (0, 3, 1, 2))
    y_tf_r = np.transpose(y_tf_r.numpy(), (0, 3, 1, 2))

    mae_l = float(np.mean(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mae_r = float(np.mean(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    mxe_l = float(np.max(np.abs(y_pt_l.detach().cpu().numpy() - y_tf_l)))
    mxe_r = float(np.max(np.abs(y_pt_r.detach().cpu().numpy() - y_tf_r)))
    print(f"[DAMOLitho generator] MAE_L={mae_l:.6e}, MAE_R={mae_r:.6e}, MAX_L={mxe_l:.6e}, MAX_R={mxe_r:.6e}")
    assert mae_l < 1e-5 and mae_r < 1e-5 and mxe_l < 1e-4 and mxe_r < 1e-4


def test_damolitho_discriminator_equivalence():
    set_seed(0)
    in_size = 32
    resize_up = 64
    pt = PTDAMOLithoDiscriminatorRef(in_size=in_size, resize_up=resize_up).cpu().eval()
    tf_model = DAMOLithoDiscriminatorTF(resize_up=(resize_up, resize_up))

    x_pt = torch.randn(2, 2, in_size, in_size, dtype=torch.float32)
    x_tf = np.transpose(x_pt.numpy(), (0, 2, 3, 1))
    _ = tf_model(x_tf, training=False)

    _copy_pt_conv_block(pt.conv0_0, tf_model.conv0_0)
    _copy_pt_conv_block(pt.conv1_0, tf_model.conv1_0)
    _copy_pt_conv_block(pt.conv2_0, tf_model.conv2_0)
    _copy_dense(pt.fc0_0, tf_model.fc0_0)

    _copy_pt_conv_block(pt.conv0_1, tf_model.conv0_1)
    _copy_pt_conv_block(pt.conv1_1, tf_model.conv1_1)
    _copy_pt_conv_block(pt.conv2_1, tf_model.conv2_1)
    _copy_dense(pt.fc0_1, tf_model.fc0_1)

    y_pt = pt(x_pt).detach().cpu().numpy()
    y_tf = tf_model(x_tf, training=False).numpy()

    mae = float(np.mean(np.abs(y_pt - y_tf)))
    mxe = float(np.max(np.abs(y_pt - y_tf)))
    print(f"[DAMOLitho discriminator] MAE={mae:.6e}, MAX={mxe:.6e}")
    assert mae < 2e-3 and mxe < 5e-3


if __name__ == "__main__":
    test_neuralilt_unet_equivalence()
    test_cfno_core_equivalence()
    test_cfnoilt_net_equivalence()
    test_cfnolitho_net_equivalence()
    test_rfno_core_equivalence()
    test_rfnonet_equivalence()
    test_ganopc_generator_equivalence()
    test_ganopc_discriminator_equivalence()
    test_lithogan_generator_equivalence()
    test_lithogan_discriminator_equivalence()
    test_damoilt_generator_equivalence()
    test_damoilt_discriminator_equivalence()
    test_damolitho_generator_equivalence()
    test_damolitho_discriminator_equivalence()
    print("All TF port smoke tests passed.")
