# -*- coding: utf-8 -*-
"""
An implementation of a model from the Deep Sets paper 
(https://arxiv.org/abs/1703.06114). This can be used to do predictions
on sets.

Modify the global pooling op by dividing by the cardinality if you invariance
to set length.

All fully connected ops are conveniently implemented as Conv1D. The models
thus obtained can make predictions based on a set of n-vectors.

"""
import tensorflow as tf
from tensorflow.keras import layers


def phi(num_nodes=5, num_layers=2, activation="relu", n=0):
    def inner(x):
        for i in range(num_layers):
            if i == 0:
                out = layers.Conv1D(
                        num_nodes, 
                        kernel_size=1, 
                        padding="same",
                        name="phi_{}_nin_{}".format(n, i)
                    )(x)
            else:
                out = layers.Conv1D(
                        num_nodes,
                        kernel_size=1,
                        padding="same",
                        name="phi_{}_nin_{}".format(n, i)
                    )(out)
            out = layers.Activation(
                    activation,
                    name="phi_{}_nin_{}_{}".format(n, i, activation))(out)
            out = layers.BatchNormalization(
                        name="phi_{}_bn_{}".format(n, i)
                    )(out)
        return out
    return inner


def global_sum_pool(n=0):
    return layers.Lambda(
        lambda x: tf.keras.backend.mean(x, axis=1, keepdims=False),
        name="sum_{}".format(n)
    )


def rho(
        num_nodes=5, 
        num_layers=2, 
        num_out=1, 
        activation="relu", 
        expand_dims=False,
        n=0
    ):
    def inner(x):
        for i in range(num_layers):
            if i == 0:
                out = layers.Dense(
                        num_nodes, 
                        name="rho_{}_dense_{}".format(n, i)
                    )(x)
                out = layers.Activation(
                        activation,
                        name="rho_{}_dense_{}_{}".format(n, i, activation)
                    )(out)
                out = layers.BatchNormalization(
                        name="rho_{}_bn_{}".format(n, i)
                    )(out)
            elif i == num_layers-1:
                out = layers.Dense(
                        num_out,
                        name="rho_{}_dense_out".format(n))(out)
            else:
                out = layers.Dense(
                        num_nodes, 
                        name="rho_{}_dense_{}".format(n, i)
                    )(out)
                out = layers.Activation(
                        activation,
                        name="rho_{}_dense_{}_{}".format(n, i, activation)
                    )(out)
                out = layers.BatchNormalization(
                        name="rho_{}_bn_{}".format(n, i)
                    )(out)
        if expand_dims:
            out = layers.Reshape(
                    (num_out, 1), 
                    name="rho_{}_expand".format(n)
                )(out)
        return out
    return inner


def set_layer(
        phi_nodes=5, 
        phi_layers=2, 
        phi_activation="relu",
        rho_nodes=5, 
        rho_layers=2, 
        rho_num_out=1,
        rho_expand=False,
        rho_activation="relu",
        n=0
    ):
    def inner(x):
        out = phi(phi_nodes, phi_layers, phi_activation, n)(x)
        out = global_sum_pool(n)(out)
        out = rho(
                rho_nodes, 
                rho_layers, 
                rho_num_out, 
                rho_activation, 
                rho_expand,
                n
            )(out)
        return out
    return inner


def deep_set(
        phi_nodes=5, 
        phi_layers=2, 
        phi_activation="relu",
        rho_nodes=5, 
        rho_layers=2, 
        rho_activation="relu",
        rho_num_out=5,
        num_layers=2,
        set_num_out=1
    ):
    def inner(x):
        for i in range(num_layers):
            if i == 0:
                out = set_layer(
                        phi_nodes, 
                        phi_layers, 
                        phi_activation,
                        rho_nodes, 
                        rho_layers,
                        rho_num_out,
                        i!=num_layers-1,
                        rho_activation,
                        i
                    )(x)
            else:
                out = set_layer(
                        phi_nodes, 
                        phi_layers, 
                        phi_activation,
                        rho_nodes, 
                        rho_layers, 
                        rho_num_out if i!=num_layers-1 else set_num_out, 
                        i!=num_layers-1, 
                        rho_activation,
                        i
                    )(out)
        return out
    return inner