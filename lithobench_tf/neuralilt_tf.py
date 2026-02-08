import tensorflow as tf


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, strides=1, use_bn=True, use_relu=True):
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
            # Match PyTorch BatchNorm2d defaults as closely as possible.
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


class BilinearUpsample2xAlignCorners(tf.keras.layers.Layer):
    def call(self, x):
        shape = tf.shape(x)
        h = shape[1] * 2
        w = shape[2] * 2
        return tf.raw_ops.ResizeBilinear(
            images=x,
            size=tf.stack([h, w]),
            align_corners=True,
            half_pixel_centers=False,
        )


class UNetTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/ilt/neuralilt.py::UNet.
    Input/Output format: NHWC.
    """

    def __init__(self):
        super().__init__()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
        self.upscale = BilinearUpsample2xAlignCorners()
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

        self.conv1 = RepeatConvBlock(2, 64, use_bn=True, use_relu=True)
        self.conv2 = RepeatConvBlock(2, 128, use_bn=True, use_relu=True)
        self.conv3 = RepeatConvBlock(2, 256, use_bn=True, use_relu=True)
        self.conv4 = RepeatConvBlock(2, 512, use_bn=True, use_relu=True)

        self.deconv4 = RepeatConvBlock(2, 256, use_bn=True, use_relu=True)
        self.deconv3 = RepeatConvBlock(2, 128, use_bn=True, use_relu=True)
        self.deconv2 = RepeatConvBlock(2, 64, use_bn=True, use_relu=True)
        self.deconv1 = ConvBnRelu(1, use_bn=False, use_relu=False)

    def call(self, x, training=False):
        conv1 = self.conv1(x, training=training)
        x = self.pool(conv1)

        conv2 = self.conv2(x, training=training)
        x = self.pool(conv2)

        conv3 = self.conv3(x, training=training)
        x = self.pool(conv3)

        x = self.conv4(x, training=training)
        x = self.upscale(x)
        x = tf.concat([x, conv3], axis=-1)

        x = self.deconv4(x, training=training)
        x = self.upscale(x)
        x = tf.concat([x, conv2], axis=-1)

        x = self.deconv3(x, training=training)
        x = self.upscale(x)
        x = tf.concat([x, conv1], axis=-1)

        x = self.deconv2(x, training=training)
        x = self.deconv1(x, training=training)
        x = self.sigmoid(x)
        return x

