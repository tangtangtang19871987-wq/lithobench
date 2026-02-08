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


class LithoGANGeneratorTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/litho/lithogan.py::Generator.
    Input/Output format: NHWC.
    Returns (xl, xr).
    """

    def __init__(self, cin=1, cout=1):
        super().__init__()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
        self.upscale = BilinearUpsample2xAlignCorners()
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

        self.conv1 = ConvBnRelu(64, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv2 = ConvBnRelu(128, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv3 = ConvBnRelu(256, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv4 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv5 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv6 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv7 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv8 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)

        self.deconv8 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.deconv7 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.deconv6 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.deconv5 = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.deconv4 = ConvBnRelu(256, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.deconv3 = ConvBnRelu(128, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.deconv2 = ConvBnRelu(64, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.deconv1l = ConvBnRelu(cout, kernel_size=5, strides=1, use_bn=False, use_relu=False)
        self.deconv1r = ConvBnRelu(cout, kernel_size=5, strides=1, use_bn=False, use_relu=False)

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.pool(x)
        x = self.conv2(x, training=training)
        x = self.pool(x)
        x = self.conv3(x, training=training)
        x = self.pool(x)
        x = self.conv4(x, training=training)
        x = self.pool(x)
        x = self.conv5(x, training=training)
        x = self.pool(x)
        x = self.conv6(x, training=training)
        x = self.pool(x)
        x = self.conv7(x, training=training)
        x = self.pool(x)
        x = self.conv8(x, training=training)
        x = self.pool(x)
        x = self.upscale(x)
        x = self.deconv8(x, training=training)
        x = self.upscale(x)
        x = self.deconv7(x, training=training)
        x = self.upscale(x)
        x = self.deconv6(x, training=training)
        x = self.upscale(x)
        x = self.deconv5(x, training=training)
        x = self.upscale(x)
        x = self.deconv4(x, training=training)
        x = self.upscale(x)
        x = self.deconv3(x, training=training)
        x = self.upscale(x)
        x = self.deconv2(x, training=training)
        x = self.upscale(x)
        xl = self.sigmoid(self.deconv1l(x, training=training))
        xr = self.sigmoid(self.deconv1r(x, training=training))
        return xl, xr


class LithoGANDiscriminatorTF(tf.keras.Model):
    """
    TensorFlow port of lithobench/litho/lithogan.py::Discriminator.
    Input/Output format: NHWC.
    Returns (left_score, right_score).
    """

    def __init__(self, cin_a=1, cin_b=1):
        super().__init__()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
        self.flatten = tf.keras.layers.Flatten()
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

        self.conv1l = ConvBnRelu(64, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv2l = ConvBnRelu(128, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv3l = ConvBnRelu(256, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv4l = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv5l = ConvBnRelu(1, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.fc1l = LinearBnRelu(1, use_bn=False, use_relu=False)

        self.conv1r = ConvBnRelu(64, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv2r = ConvBnRelu(128, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv3r = ConvBnRelu(256, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv4r = ConvBnRelu(512, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.conv5r = ConvBnRelu(1, kernel_size=5, strides=1, use_bn=True, use_relu=True)
        self.fc1r = LinearBnRelu(1, use_bn=False, use_relu=False)

    def call(self, xl, xr, training=False):
        yl = self.conv1l(xl, training=training)
        yl = self.pool(yl)
        yl = self.conv2l(yl, training=training)
        yl = self.pool(yl)
        yl = self.conv3l(yl, training=training)
        yl = self.pool(yl)
        yl = self.conv4l(yl, training=training)
        yl = self.conv5l(yl, training=training)
        yl = self.flatten(yl)
        yl = self.fc1l(yl, training=training)
        yl = self.sigmoid(yl)

        yr = self.conv1r(xr, training=training)
        yr = self.pool(yr)
        yr = self.conv2r(yr, training=training)
        yr = self.pool(yr)
        yr = self.conv3r(yr, training=training)
        yr = self.pool(yr)
        yr = self.conv4r(yr, training=training)
        yr = self.conv5r(yr, training=training)
        yr = self.flatten(yr)
        yr = self.fc1r(yr, training=training)
        yr = self.sigmoid(yr)

        return yl, yr
