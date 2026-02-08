import tensorflow as tf


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, strides=1, padding="same", use_bn=True, use_relu=False):
        super().__init__()
        self.pad = None
        conv_padding = padding
        if isinstance(padding, int):
            self.pad = tf.keras.layers.ZeroPadding2D(padding=((padding, padding), (padding, padding)))
            conv_padding = "valid"
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_padding,
            use_bias=True,
        )
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        if use_relu:
            self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_relu:
            x = self.relu(x)
        return x


class RepeatConvBlock(tf.keras.layers.Layer):
    def __init__(self, n, out_channels, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True):
        super().__init__()
        self.layers_ = [
            ConvBnRelu(
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bn=use_bn,
                use_relu=use_relu,
            )
            for _ in range(n)
        ]

    def call(self, x, training=False):
        for layer in self.layers_:
            x = layer(x, training=training)
        return x


class PixelShuffleBlock(tf.keras.layers.Layer):
    def __init__(self, upscale, out_channels, kernel_size=3, strides=1, padding="same", use_bn=True, use_relu=False):
        super().__init__()
        self.upscale = upscale
        self.out_channels = out_channels
        self.pad = None
        conv_padding = padding
        if isinstance(padding, int):
            self.pad = tf.keras.layers.ZeroPadding2D(padding=((padding, padding), (padding, padding)))
            conv_padding = "valid"
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels * (upscale ** 2),
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_padding,
            use_bias=True,
        )
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        if use_relu:
            self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        # Match PyTorch nn.PixelShuffle channel layout exactly.
        r = self.upscale
        x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
        shape = tf.shape(x)
        n = shape[0]
        h = shape[2]
        w = shape[3]
        x = tf.reshape(x, [n, self.out_channels, r, r, h, w])
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, [n, self.out_channels, h * r, w * r])
        x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_relu:
            x = self.relu(x)
        return x


class LinearBnRelu(tf.keras.layers.Layer):
    def __init__(self, out_features, use_bn=True, use_relu=False):
        super().__init__()
        self.fc = tf.keras.layers.Dense(out_features, use_bias=True)
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        if use_relu:
            self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.use_relu:
            x = self.relu(x)
        return x


class GANOPCGeneratorTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/ilt/ganopc.py::Generator.
    Input/Output format: NHWC.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnRelu(16, kernel_size=5, strides=2, padding=2, use_bn=True, use_relu=True)
        self.conv2 = ConvBnRelu(64, kernel_size=5, strides=2, padding=2, use_bn=True, use_relu=True)
        self.conv3 = ConvBnRelu(128, kernel_size=5, strides=2, padding=2, use_bn=True, use_relu=True)
        self.conv4 = ConvBnRelu(512, kernel_size=5, strides=2, padding=2, use_bn=True, use_relu=True)
        self.conv5 = ConvBnRelu(1024, kernel_size=5, strides=2, padding=2, use_bn=True, use_relu=True)

        self.spsr5 = PixelShuffleBlock(2, 512, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.spsr4 = PixelShuffleBlock(2, 128, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.spsr3 = PixelShuffleBlock(2, 64, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.spsr2 = PixelShuffleBlock(2, 16, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.spsr1 = PixelShuffleBlock(2, 1, kernel_size=3, strides=1, padding=1, use_bn=False, use_relu=False)

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.spsr5(x, training=training)
        x = self.spsr4(x, training=training)
        x = self.spsr3(x, training=training)
        x = self.spsr2(x, training=training)
        x = self.spsr1(x, training=training)
        return x


class GANOPCDiscriminatorTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/ilt/ganopc.py::Discriminator.
    Input/Output format: NHWC.
    """

    def __init__(self):
        super().__init__()
        self.repeat2a = RepeatConvBlock(2, 64, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.conv1 = ConvBnRelu(64, kernel_size=3, strides=2, padding=1, use_bn=True, use_relu=True)
        self.repeat2b = RepeatConvBlock(2, 128, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.conv2 = ConvBnRelu(128, kernel_size=3, strides=2, padding=1, use_bn=True, use_relu=True)
        self.repeat3a = RepeatConvBlock(3, 256, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.conv3 = ConvBnRelu(256, kernel_size=3, strides=2, padding=1, use_bn=True, use_relu=True)
        self.repeat3b = RepeatConvBlock(3, 512, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.conv4 = ConvBnRelu(512, kernel_size=3, strides=2, padding=1, use_bn=True, use_relu=True)
        self.repeat3c = RepeatConvBlock(3, 512, kernel_size=3, strides=1, padding=1, use_bn=True, use_relu=True)
        self.conv5 = ConvBnRelu(512, kernel_size=3, strides=2, padding=1, use_bn=True, use_relu=True)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = LinearBnRelu(2048, use_bn=True, use_relu=True)
        self.fc2 = LinearBnRelu(512, use_bn=True, use_relu=True)
        self.fc3 = LinearBnRelu(1, use_bn=True, use_relu=True)
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, x, training=False):
        x = self.repeat2a(x, training=training)
        x = self.conv1(x, training=training)
        x = self.repeat2b(x, training=training)
        x = self.conv2(x, training=training)
        x = self.repeat3a(x, training=training)
        x = self.conv3(x, training=training)
        x = self.repeat3b(x, training=training)
        x = self.conv4(x, training=training)
        x = self.repeat3c(x, training=training)
        x = self.conv5(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        return self.sigmoid(x)
