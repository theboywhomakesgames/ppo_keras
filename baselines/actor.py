import tensorflow as tf
import numpy as np

class actor(tf.Module):
    def __init__(self):
        print("build actor")
        super(actor, self).__init__()
        print("1")
        self.d1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        print("2")
        self.d2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        print("mu")
        self.mu = tf.keras.layers.Dense(4, activation=tf.nn.tanh)
        print("std")
        self.std = tf.keras.layers.Dense(4, activation=self.std_activation)

    def std_activation(self, input):
        phase1 = tf.nn.tanh(input)
        phase1 = (phase1 + 1) / 2
        return phase1

    def __call__(self, input):
        inner = self.d1(input)
        inner = self.d2(inner)
        mu = self.mu(inner)
        std = self.std(inner)

        # try:
        #     if np.isnan(mu.numpy()).any() or np.isnan(std.numpy()).any():
        #         print("nan value")
        # except:
        #     print("err")

        return mu, std
