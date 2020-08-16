import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.ops import rnn_cell_impl
import pandas as pd
import numpy as np

class DiscriminatorLSTMCell(rnn_cell_impl.BasicLSTMCell):
    def __init__(self, 
                num_units, 
                num_inputs
                activation,
                reuse = False,
                name,
                dtype,
                **kwargs
                ):
        '''
        Args:
        num_units: int, the number of units in the LSTM cell. 
        num_inputs: int, the number of inputs into the LSTM cell.


        '''
        super(DiscriminatorLSTMCell, self).__init__()
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.activation = activation
        
    
    # if reuse == False:
    #     self.w = self.BuildInputWeights()
    
    def BuildInputWeights(self):
        pass 



    
