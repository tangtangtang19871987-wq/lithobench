import tensorflow as tf


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
        # Match PyTorch ConvTranspose2d(k=3,s=2,p=1,output_padding=1):
        # VALID + crop top-left by 1 pixel.
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


class RFNOTF(tf.keras.layers.Layer):
    """
    TensorFlow port of lithobench/litho/doinn.py::RFNO.
    Uses NHWC input/output format.
    """

    def __init__(self, out_channels, modes1, modes2):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def build(self, input_shape):
        scale = 1.0 / float(self.out_channels)
        rnd = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
        self.weights0 = self.add_weight(
            name="weights0",
            shape=(1, self.out_channels, 1, 1),
            initializer=lambda shape, dtype=None: tf.cast(scale * rnd(shape, dtype=tf.float32), tf.complex64),
            dtype=tf.complex64,
            trainable=True,
        )
        self.weights1 = self.add_weight(
            name="weights1",
            shape=(1, self.out_channels, self.modes1, self.modes2),
            initializer=lambda shape, dtype=None: tf.cast(scale * rnd(shape, dtype=tf.float32), tf.complex64),
            dtype=tf.complex64,
            trainable=True,
        )
        self.weights2 = self.add_weight(
            name="weights2",
            shape=(1, self.out_channels, self.modes1, self.modes2),
            initializer=lambda shape, dtype=None: tf.cast(scale * rnd(shape, dtype=tf.float32), tf.complex64),
            dtype=tf.complex64,
            trainable=True,
        )

    def call(self, x):
        # x: NHWC -> NCHW
        x_nchw = tf.transpose(x, [0, 3, 1, 2])
        x_nchw = tf.cast(x_nchw, tf.float32)
        b = tf.shape(x_nchw)[0]
        h = tf.shape(x_nchw)[2]
        w = tf.shape(x_nchw)[3]
        wf = w // 2 + 1

        x_ft = tf.signal.rfft2d(x_nchw)  # [B, Cin, H, Wf], complex64
        x_ft = x_ft * self.weights0      # broadcast to [B, Cout, H, Wf], Cin expected 1

        # Low-frequency block (top)
        top_src = x_ft[:, :, : self.modes1, : self.modes2] * self.weights1
        top = tf.pad(top_src, [[0, 0], [0, 0], [0, h - self.modes1], [0, wf - self.modes2]])

        # Low-frequency block from negative rows (bottom)
        bot_src = x_ft[:, :, -self.modes1 :, : self.modes2] * self.weights2
        bottom = tf.pad(bot_src, [[0, 0], [0, 0], [h - self.modes1, 0], [0, wf - self.modes2]])

        out_ft = top + bottom
        y = tf.signal.irfft2d(out_ft, fft_length=tf.stack([h, w]))  # [B, Cout, H, W]
        return tf.transpose(y, [0, 2, 3, 1])  # NHWC


class RFNONetTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/litho/doinn.py::RFNONet.
    Returns (xl, xr), both NHWC.
    """

    def __init__(self, modes1=32, modes2=32):
        super().__init__()
        self.rfno = RFNOTF(64, modes1=modes1, modes2=modes2)

        self.conv0 = ConvBnRelu(16, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.conv1 = ConvBnRelu(32, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.conv2 = ConvBnRelu(64, kernel_size=3, strides=2, use_bn=True, use_relu=True)

        self.deconv0 = DeconvBnRelu(32, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.deconv1 = DeconvBnRelu(16, kernel_size=3, strides=2, use_bn=True, use_relu=True)
        self.deconv2 = DeconvBnRelu(16, kernel_size=3, strides=2, use_bn=True, use_relu=True)

        self.conv3 = ConvBnRelu(16, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv4 = ConvBnRelu(16, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv5 = ConvBnRelu(8, kernel_size=3, strides=1, use_bn=True, use_relu=True)
        self.conv6l = ConvBnRelu(1, kernel_size=3, strides=1, use_bn=False, use_relu=False)
        self.conv6r = ConvBnRelu(1, kernel_size=3, strides=1, use_bn=False, use_relu=False)
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, x, training=False):
        # Average pool by 8, keep NHWC
        br0_in = tf.nn.avg_pool2d(x, ksize=8, strides=8, padding="VALID")
        br0 = self.rfno(br0_in)

        br1_0 = self.conv0(x, training=training)
        br1_1 = self.conv1(br1_0, training=training)
        br1_2 = self.conv2(br1_1, training=training)

        joined = self.deconv0(tf.concat([br0, br1_2], axis=-1), training=training)
        joined = self.deconv1(tf.concat([joined, br1_1], axis=-1), training=training)
        joined = self.deconv2(tf.concat([joined, br1_0], axis=-1), training=training)

        joined = self.conv3(joined, training=training)
        joined = self.conv4(joined, training=training)
        joined = self.conv5(joined, training=training)
        xl = self.sigmoid(self.conv6l(joined, training=training))
        xr = self.sigmoid(self.conv6r(joined, training=training))
        return xl, xr
