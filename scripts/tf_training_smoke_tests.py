import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf

from lithobench_tf import (
    UNetTF,
    CFNOILTNetTF,
    CFNOLithoNetTF,
    RFNONetTF,
    GANOPCGeneratorTF,
    GANOPCDiscriminatorTF,
    LithoGANGeneratorTF,
    LithoGANDiscriminatorTF,
    DAMOILTGeneratorTF,
    DAMOILTDiscriminatorTF,
    DAMOLithoGeneratorTF,
    DAMOLithoDiscriminatorTF,
)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _assert_finite(grads):
    for g in grads:
        if g is None:
            continue
        if g.dtype.is_complex:
            tf.debugging.assert_all_finite(tf.math.real(g), "Found non-finite real gradient")
            tf.debugging.assert_all_finite(tf.math.imag(g), "Found non-finite imag gradient")
        else:
            tf.debugging.assert_all_finite(g, "Found non-finite gradient")


def _one_step_single(model, x_shape, y_shape, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    x = tf.random.normal(x_shape)
    y = tf.random.normal(y_shape)
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = tf.reduce_mean(tf.square(pred - y))
    grads = tape.gradient(loss, model.trainable_variables)
    _assert_finite(grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())


def _one_step_tuple(model, x_shape, y_shapes, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    x = tf.random.normal(x_shape)
    y0 = tf.random.normal(y_shapes[0])
    y1 = tf.random.normal(y_shapes[1])
    with tf.GradientTape() as tape:
        p0, p1 = model(x, training=True)
        loss = tf.reduce_mean(tf.square(p0 - y0)) + tf.reduce_mean(tf.square(p1 - y1))
    grads = tape.gradient(loss, model.trainable_variables)
    _assert_finite(grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())


def _one_step_disc(model, x_shape, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    x = tf.random.normal(x_shape)
    y = tf.random.uniform([x_shape[0], 1], minval=0.0, maxval=1.0)
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, pred))
    grads = tape.gradient(loss, model.trainable_variables)
    _assert_finite(grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())


def _one_step_disc_dual(model, xl_shape, xr_shape, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    xl = tf.random.normal(xl_shape)
    xr = tf.random.normal(xr_shape)
    y = tf.random.uniform([xl_shape[0], 1], minval=0.0, maxval=1.0)
    with tf.GradientTape() as tape:
        pl, pr = model(xl, xr, training=True)
        loss = 0.5 * (
            tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, pl))
            + tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, pr))
        )
    grads = tape.gradient(loss, model.trainable_variables)
    _assert_finite(grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())


def run():
    set_seed(0)
    results = {}

    results["UNetTF"] = _one_step_single(UNetTF(), [1, 64, 64, 1], [1, 64, 64, 1])
    cfno_sizes = ((16, 16), (16, 16), (16, 16))
    results["CFNOILTNetTF"] = _one_step_single(
        CFNOILTNetTF(cfno_sizes=cfno_sizes), [1, 128, 128, 1], [1, 128, 128, 1]
    )
    results["CFNOLithoNetTF"] = _one_step_tuple(
        CFNOLithoNetTF(cfno_sizes=cfno_sizes), [1, 128, 128, 1], ([1, 128, 128, 1], [1, 128, 128, 1])
    )
    results["RFNONetTF"] = _one_step_tuple(RFNONetTF(modes1=8, modes2=8), [1, 128, 128, 1], ([1, 128, 128, 1], [1, 128, 128, 1]))

    results["GANOPCGeneratorTF"] = _one_step_single(GANOPCGeneratorTF(), [1, 64, 64, 1], [1, 64, 64, 1])
    results["GANOPCDiscriminatorTF"] = _one_step_disc(GANOPCDiscriminatorTF(), [2, 64, 64, 1])

    results["LithoGANGeneratorTF"] = _one_step_tuple(
        LithoGANGeneratorTF(), [1, 256, 256, 1], ([1, 256, 256, 1], [1, 256, 256, 1])
    )
    results["LithoGANDiscriminatorTF"] = _one_step_disc_dual(
        LithoGANDiscriminatorTF(), [2, 256, 256, 1], [2, 256, 256, 1]
    )

    results["DAMOILTGeneratorTF"] = _one_step_single(DAMOILTGeneratorTF(), [1, 128, 128, 1], [1, 128, 128, 1])
    results["DAMOILTDiscriminatorTF"] = _one_step_disc(DAMOILTDiscriminatorTF(resize_up=(64, 64)), [2, 32, 32, 1])

    results["DAMOLithoGeneratorTF"] = _one_step_tuple(
        DAMOLithoGeneratorTF(), [1, 128, 128, 1], ([1, 128, 128, 1], [1, 128, 128, 1])
    )
    results["DAMOLithoDiscriminatorTF"] = _one_step_disc(
        DAMOLithoDiscriminatorTF(resize_up=(64, 64)), [2, 32, 32, 2]
    )

    for name, loss in results.items():
        print(f"[{name}] one-step loss={loss:.6f}")
    print("All TF training smoke tests passed.")


if __name__ == "__main__":
    run()
