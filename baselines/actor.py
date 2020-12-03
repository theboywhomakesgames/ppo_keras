from numpy.core.fromnumeric import shape
import tensorflow as tf
import numpy as np

class actor(tf.keras.Model):
    def __init__(self):
        print("build actor")
        super(actor, self).__init__()
        self.d1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.d2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.mu = tf.keras.layers.Dense(4, activation=tf.nn.tanh)
        self.std = tf.keras.layers.Dense(4, activation=self.std_activation)

    def std_activation(self, input):
        phase1 = tf.nn.sigmoid(input)
        phase2 = phase1 * 3 / 4
        return phase2

    def set_inputs(self, input):
        input = np.array(input)
        self._set_inputs(input.shape[0])

    def call(self, input):
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
