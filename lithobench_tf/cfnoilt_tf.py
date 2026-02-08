import tensorflow as tf


class ComplexDense(tf.keras.layers.Layer):
    """
    Complex linear layer implemented with two real-valued dense layers:
      (Wr + iWi)(xr + ixi) = (Wr*xr - Wi*xi) + i(Wr*xi + Wi*xr)
    """

    def __init__(self, out_features):
        super().__init__()
        self.fc_r = tf.keras.layers.Dense(out_features, use_bias=True)
        self.fc_i = tf.keras.layers.Dense(out_features, use_bias=True)

    def call(self, x_complex):
        xr = tf.math.real(x_complex)
        xi = tf.math.imag(x_complex)
        yr = self.fc_r(xr) - self.fc_i(xi)
        yi = self.fc_r(xi) + self.fc_i(xr)
        return tf.complex(yr, yi)


class DepthwiseConvBn(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.dw = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            padding="same",
            depth_multiplier=1,
            use_bias=True,
        )
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)

    def call(self, x, training=False):
        x = self.dw(x)
        x = self.bn(x, training=training)
        return x


class CFNOTF(tf.keras.layers.Layer):
    """
    TensorFlow port of lithobench/ilt/cfnoilt.py::CFNO core layer.
    Input format: NHWC.
    """

    def __init__(self, c=1, d=16, k=16, s=1, size=(128, 128)):
        super().__init__()
        self.c = c
        self.d = d
        self.k = k
        self.s = s
        self.size = size
        self.fc = ComplexDense(out_features=d)
        self.conv = DepthwiseConvBn(channels=d, kernel_size=2 * s + 1)

    def _split_patches(self, x):
        # x: [B,H,W,C] -> [B*Hk*Wk, k*k*C]
        shape = tf.shape(x)
        b = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        hk = h // self.k
        wk = w // self.k
        x = tf.reshape(x, [b, hk, self.k, wk, self.k, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # [B, hk, wk, k, k, C]
        patches = tf.reshape(x, [b * hk * wk, self.k * self.k * c])
        return patches, b, hk, wk

    def call(self, x, training=False):
        patches, b, hk, wk = self._split_patches(x)
        patches = tf.cast(patches, tf.complex64)
        fft = tf.signal.fft(patches)
        fc = self.fc(fft)
        ifft = tf.math.real(tf.signal.ifft(fc))
        ifft = tf.reshape(ifft, [b, hk, wk, self.d])  # NHWC
        conved = self.conv(ifft, training=training)
        # PyTorch code uses default nearest in F.interpolate.
        return tf.image.resize(conved, size=self.size, method="nearest")


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, strides=1, use_bn=True, use_relu=False):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=True,
        )
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        if use_relu:
            self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_relu:
            x = self.relu(x)
        return x


class DeconvBnRelu(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, strides=2, use_bn=True, use_relu=False):
        super().__init__()
        # To match PyTorch ConvTranspose2d(k=3,s=2,p=1,output_padding=1),
        # use VALID and crop top-left by 1 after deconv.
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            use_bias=True,
        )
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        if use_relu:
            self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.deconv(x)
        x = x[:, 1:, 1:, :]
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_relu:
            x = self.relu(x)
        return x


class RepeatConvBlock(tf.keras.layers.Layer):
    def __init__(self, n, out_channels, kernel_size=3, strides=1, use_bn=True, use_relu=True):
        super().__init__()
        self.layers_ = [
            ConvBnRelu(
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                use_bn=use_bn,
                use_relu=use_relu,
            )
            for _ in range(n)
        ]

    def call(self, x, training=False):
        for layer in self.layers_:
            x = layer(x, training=training)
        return x


class CFNOILTNetTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/ilt/cfnoilt.py::CFNONet.
    Input/Output format: NHWC.
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
        self.conv6 = ConvBnRelu(1, kernel_size=3, strides=1, use_bn=False, use_relu=False)
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
        x = self.conv6(x, training=training)
        return self.sigmoid(x)
