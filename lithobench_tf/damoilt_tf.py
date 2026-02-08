import tensorflow as tf


class ConvBlockTF(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, strides=1, padding="same", leaky=False):
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
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.act = tf.keras.layers.LeakyReLU(negative_slope=0.01) if leaky else tf.keras.layers.ReLU()

    def call(self, x, training=False):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)


class DeconvBlockTF(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, strides=2):
        super().__init__()
        # Match PyTorch ConvTranspose2d(k=3,s=2,p=1,output_padding=1)
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            use_bias=True,
        )
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.deconv(x)
        x = x[:, 1:, 1:, :]
        x = self.bn(x, training=training)
        return self.relu(x)


class DAMOILTGeneratorTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/ilt/damoilt.py::Generator.
    Input/Output format: NHWC.
    """

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        n1 = 32
        f0, f1, f2, f3, f4 = n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32

        self.conv_head = ConvBlockTF(n1, kernel_size=7, strides=1, padding="same", leaky=False)
        self.conv0 = ConvBlockTF(f0, kernel_size=3, strides=2, padding=1, leaky=False)
        self.conv1 = ConvBlockTF(f1, kernel_size=3, strides=2, padding=1, leaky=False)
        self.conv2 = ConvBlockTF(f2, kernel_size=3, strides=2, padding=1, leaky=False)
        self.conv3 = ConvBlockTF(f3, kernel_size=3, strides=2, padding=1, leaky=False)
        self.conv4 = ConvBlockTF(f4, kernel_size=3, strides=2, padding=1, leaky=False)

        self.res0 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res1 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res2 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res3 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res4 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res5 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res6 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res7 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)
        self.res8 = ConvBlockTF(f4, kernel_size=3, strides=1, padding="same", leaky=False)

        self.deconv4 = DeconvBlockTF(f3, kernel_size=3, strides=2)
        self.deconv3 = DeconvBlockTF(f2, kernel_size=3, strides=2)
        self.deconv2 = DeconvBlockTF(f1, kernel_size=3, strides=2)
        self.deconv1 = DeconvBlockTF(f0, kernel_size=3, strides=2)
        self.deconv0 = DeconvBlockTF(n1, kernel_size=3, strides=2)

        self.conv_tail = tf.keras.layers.Conv2D(out_ch, kernel_size=7, strides=1, padding="same", use_bias=True)

    def call(self, x, training=False):
        x = self.conv_head(x, training=training)
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)

        x = self.res0(x, training=training)
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        x = self.res4(x, training=training)
        x = self.res5(x, training=training)
        x = self.res6(x, training=training)
        x = self.res7(x, training=training)
        x = self.res8(x, training=training)

        x = self.deconv4(x, training=training)
        x = self.deconv3(x, training=training)
        x = self.deconv2(x, training=training)
        x = self.deconv1(x, training=training)
        x = self.deconv0(x, training=training)
        return self.conv_tail(x)


class DAMOILTDiscriminatorTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/ilt/damoilt.py::Discriminator.
    Input/Output format: NHWC.
    """

    def __init__(self, resize_up=(512, 512)):
        super().__init__()
        self.resize_up = resize_up

        self.conv0_0 = ConvBlockTF(64, kernel_size=4, strides=2, padding=1, leaky=True)
        self.conv1_0 = ConvBlockTF(128, kernel_size=4, strides=1, padding="same", leaky=True)
        self.conv2_0 = ConvBlockTF(1, kernel_size=4, strides=1, padding="same", leaky=True)
        self.flatten_0 = tf.keras.layers.Flatten()
        self.fc0_0 = tf.keras.layers.Dense(1, use_bias=True)
        self.sigmoid_0 = tf.keras.layers.Activation("sigmoid")

        self.conv0_1 = ConvBlockTF(64, kernel_size=4, strides=2, padding=1, leaky=True)
        self.conv1_1 = ConvBlockTF(128, kernel_size=4, strides=1, padding="same", leaky=True)
        self.conv2_1 = ConvBlockTF(1, kernel_size=4, strides=1, padding="same", leaky=True)
        self.flatten_1 = tf.keras.layers.Flatten()
        self.fc0_1 = tf.keras.layers.Dense(1, use_bias=True)
        self.sigmoid_1 = tf.keras.layers.Activation("sigmoid")

    def call(self, x, training=False):
        x0 = self.conv0_0(x, training=training)
        x0 = self.conv1_0(x0, training=training)
        x0 = self.conv2_0(x0, training=training)
        x0 = self.flatten_0(x0)
        x0 = self.fc0_0(x0)
        x0 = self.sigmoid_0(x0)

        x1 = tf.image.resize(x, size=self.resize_up, method="nearest")
        x1 = self.conv0_1(x1, training=training)
        x1 = self.conv1_1(x1, training=training)
        x1 = self.conv2_1(x1, training=training)
        x1 = self.flatten_1(x1)
        x1 = self.fc0_1(x1)
        x1 = self.sigmoid_1(x1)
        return 0.5 * (x0 + x1)
