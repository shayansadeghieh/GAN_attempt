import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.ops import rnn_cell_impl
import pandas as pd
import numpy as np

class DiscriminatorLSTMCell(rnn_cell_impl.BasicLSTMCell):
    def __init__(self, 
                num_units, 
                forget_bias,
                activation,
                reuse,
                name,
                dtype,
                **kwargs
                ):
        '''
        Args:
        num_units: int, The number of units in the LSTM cell. 
        forget_bias: We add forget_bias (default: 1) to the 
        biases of the forget gate in order to reduce the scale of 
        forgetting in the beginning of the training. 
        state_is_tuple:	If True, accepted and returned states are 
        2-tuples of the c_state and m_state. If False, they are 
        concatenated along the column axis. The latter behavior will soon be 
        deprecated.
        activation: Activation function of the inner states. Default: tanh
        reuse: (optional) Python boolean describing whether to reuse variables 
        in an existing scope.
        name: String, the name of the layer. Layers with the same name 
        will share weights, but to avoid mistakes we require reuse=True 
        in such cases.
        dtype: Default dtype of the layer 
        **kwargs: Dict, keyword named properties for common layer attributes
        '''
    super(DiscriminatorLSTMCell, self).__init__()
    
