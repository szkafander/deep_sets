# -*- coding: utf-8 -*-
"""
Deep Set examples

"""

import deep_set as ds
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import moment
import matplotlib.pyplot as pl


# dummy data generators --- these all generate sets of 1-vectors (scalars)
def gen_pop(num_points, num_batch, bias_spread=3, var_spread=2):
    x = np.random.rand(num_batch, num_points, 1)
    bias = np.random.rand(len(x)) * bias_spread - bias_spread / 2
    var = np.random.rand(len(x)) * var_spread
    x += np.tile(bias[..., np.newaxis], (1, num_points))[..., np.newaxis]
    x *= np.tile(var[..., np.newaxis], (1, num_points))[..., np.newaxis]
    return x


mean = lambda x: np.mean(x, axis=1)
std = lambda x: np.std(x, axis=1)
nth_moment = lambda x, o: moment(x, axis=1, moment=o) / np.std(x, axis=1) ** o
maximum = lambda x: np.max(x, axis=1)


def test_data(
        target_function,
        num_points,
        num_batch=50,
        bias_spread=3,
        var_spread=2
    ):
    x = gen_pop(num_points, num_batch, bias_spread, var_spread)
    y = target_function(x)
    return x, y


# compare against a simple MLP if you'd like
def mlp(num_nodes=5, num_layers=2, activation="relu"):
    def inner(x):
        out = layers.Flatten()(x)
        for i in range(num_layers):
            out = layers.Dense(num_nodes)(out)
            out = layers.Activation(activation)(out)
        out = layers.Dense(1)(out)
        return out
    return inner


target_function = maximum
genfunc = lambda c, num_batch: test_data(target_function, c, num_batch)
# train on sets with cardinality = c_train
c_train = 200
num_batch = 100
x, y = genfunc(c_train, num_batch)

inp = layers.Input(shape=(None, 1))
model = tf.keras.models.Model(
        inputs=[inp], 
        outputs=[ds.deep_set(10, 1, "relu", 10, 2, "relu", 1, 1, 1)(inp)])
model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
        loss="mae"
    )
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)

model.fit(x=x, y=y, batch_size=100, epochs=10000, callbacks=[early_stopping])

# test on sets with cardinality = num_pts_test
c_test = 50
num_test = 100
x_test, y_test = genfunc(c_test, num_test)
preds = model.predict(x_test)

pl.plot(y_test, preds, 'o')
pl.plot(y_test, y_test)

# repeat this with an MLP - it will fail when c_train != c_test
