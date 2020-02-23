from keras import backend as K
from keras.engine.topology import Layer
# from keras.layers import Dense
import numpy as np


class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape
