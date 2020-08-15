import tensorflow as tf 
import numpy as np
import pandas as pd 

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
        
