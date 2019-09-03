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
    x = np.random.randn(num_batch, num_points, 1)
    bias = np.random.rand(len(x)) * bias_spread - bias_spread / 2
    var = np.random.rand(len(x)) * var_spread
    x += np.tile(bias[...,np.newaxis], (1, num_points))[...,np.newaxis]
    x *= np.tile(var[...,np.newaxis], (1, num_points))[...,np.newaxis]
    return x


def td_mean(num_points, num_batch=50, bias_spread=3, var_spread=2):
    """ Use this if the set mean is the target """
    x = gen_pop(num_points, num_batch, bias_spread, var_spread)
    y = np.mean(x, axis=1, keepdims=True)[:,:,0]
    return x, y


def td_std(num_points, num_batch=50, bias_spread=3, var_spread=2):
    """ Use this if the set std is the target """
    x = gen_pop(num_points, num_batch, bias_spread, var_spread)
    y = np.std(x, axis=1, keepdims=True)[:,:,0]
    return x, y


def td_skew(num_points, num_batch=50, bias_spread=3, var_spread=2, skew_spread=2):
    """ Use this if the set skewness is the target """
    x = np.zeros((num_batch, num_points, 1))
    y = np.zeros((num_batch, 1))
    for i_batch in range(num_batch):
        skew = np.random.rand() * skew_spread - skew_spread / 2
        pop = stats.skewnorm.rvs(skew, size=(num_points, 1))
        x[i_batch,...] = pop
        y[i_batch,:] = skew
    return x, y


def td_moment(num_points, num_batch=50, order=3, bias_spread=3, var_spread=2):
    """ Use this if an arbitrary moment is the target (does not make much sense
    above the 4th though) """
    x = gen_pop(num_points, num_batch, bias_spread, var_spread)
    std = np.std(x, axis=1)
    mu = moment(x, axis=1, moment=order)
    return x, mu / std**order


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


genfunc = td_std
# train on sets with cardinality = c_train
c_train = 200
num_batch = 10000
x, y = genfunc(c_train, num_batch)
inp = layers.Input(shape=(None, 1))

model = tf.keras.models.Model(
        inputs=[inp], 
        outputs=[ds.deep_set(10, 2, "relu", 10, 2, "relu", 1, 1, 1)(inp)])
model.compile("sgd", loss="mse")
model.summary()

model.fit(x=x, y=y, batch_size=100, epochs=100, validation_split=0.2)

# test on sets with cardinality = num_pts_test
c_test = 100
num_test = 1000
x_test, y_test = genfunc(c_test, num_test)
preds = model.predict(x_test)

pl.plot(y_test, preds, 'o')
pl.plot(y_test, y_test)

# repeat this with an MLP - it will fail when c_train != c_test
