import tensorflow as tf
import numpy as np

class critic(tf.Module):
    def __init__(self):
        print("build critic")
        super(critic, self).__init__()
        self.d1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.d2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.v = tf.keras.layers.Dense(1)

    def custom_activation(self, input):
        inner = tf.nn.tanh(input / 2)
        return inner * 10

    def __call__(self, input):
        inner = self.d1(input)
        inner = self.d2(inner)
        v = self.v(inner)

        # try:
        #     if np.isnan(v.numpy()).any():
        #         print("nan value")
        # except:
            # print("err")

        return v
