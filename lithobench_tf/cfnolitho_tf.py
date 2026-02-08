import tensorflow as tf

from .cfnoilt_tf import CFNOTF, ConvBnRelu, DeconvBnRelu, RepeatConvBlock


class CFNOLithoNetTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/litho/cfnolitho.py::CFNONet.
    Input/Output format: NHWC.
    Returns (xl, xr).
    """

    def __init__(self, cfno_sizes=((128, 128), (128, 128), (128, 128))):
        super().__init__()
        self.cfno0 = CFNOTF(c=1, d=16, k=16, s=1, size=cfno_sizes[0])
        self.cfno1 = CFNOTF(c=1, d=32, k=32, s=1, size=cfno_sizes[1])
        self.cfno2 = CFNOTF(c=1, d=64, k=64, s=1, size=cfno_sizes[2])

        self.conv0a = ConvBnRelu(32, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.conv0b = RepeatConvBlock(2, 32, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv1a = ConvBnRelu(64, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.conv1b = RepeatConvBlock(2, 64, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv2a = ConvBnRelu(128, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.conv2b = RepeatConvBlock(2, 128, kernel_size=3, strides=1, use_bn=True, use_relu=True)

        self.deconv0a = DeconvBnRelu(128, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.deconv0b = RepeatConvBlock(2, 128, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.deconv1a = DeconvBnRelu(64, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.deconv1b = RepeatConvBlock(2, 64, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.deconv2a = DeconvBnRelu(32, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.deconv2b = RepeatConvBlock(2, 32, kernel_size=3, strides=1, use_bn=True, use_relu=True)

        self.conv3 = ConvBnRelu(32, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv4 = ConvBnRelu(32, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv5 = ConvBnRelu(32, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv6l = ConvBnRelu(1, kernel_size=3, strides=1, use_bn=False, use_relu=False)
        self.conv6r = ConvBnRelu(1, kernel_size=3, strides=1, use_bn=False, use_relu=False)
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, x, training=False):
        br0 = self.cfno0(x, training=training)
        br1 = self.cfno1(x, training=training)
        br2 = self.cfno2(x, training=training)

        br3 = self.conv0a(x, training=training)
        br3 = self.conv0b(br3, training=training)
        br3 = self.conv1a(br3, training=training)
        br3 = self.conv1b(br3, training=training)
        br3 = self.conv2a(br3, training=training)
        br3 = self.conv2b(br3, training=training)

        feat = tf.concat([br0, br1, br2, br3], axis=-1)
        x = self.deconv0a(feat, training=training)
        x = self.deconv0b(x, training=training)
        x = self.deconv1a(x, training=training)
        x = self.deconv1b(x, training=training)
        x = self.deconv2a(x, training=training)
        x = self.deconv2b(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        xl = self.sigmoid(self.conv6l(x, training=training))
        xr = self.sigmoid(self.conv6r(x, training=training))
        return xl, xr
