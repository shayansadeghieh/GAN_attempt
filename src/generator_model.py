import tensorflow as tf 
import numpy as np
import pandas as pd 

'''
The goal of the model is to create a generator that produces
N x E nodes and edges that are connected in a similar, but 
potentially not identical fashion to the graphs that the
discriminator is aware of(who knows if this
will work)

'''
class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(DenseLayer, self).__init__()
        self.num_outputs = num_outputs
    
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                shape=[int(input_shape[-1]),
                                        self.num_outputs])
    
    def call(self, input):
        return tf.matmul(input, self.kernel)
        
