import tensorflow.compat.v1 as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import pandas as pd
import numpy as np

#Import custom classes 
from linear_class import _Linear

class LSTMCell(rnn_cell_impl.RNNCell):
    '''
    Module implementing LSTM Cell. Child of tensorflow's
    RNNCell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.
    '''
    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
        num_units: int, The number of units in the LSTM cell.
        forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
        state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
        activation: Activation function of the inner states.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        When restoring from CudnnLSTM-trained checkpoints, must use
        CudnnCompatibleLSTMCell instead.
        """
        super(LSTMCell, self).__init__()
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None




if __name__ == "__main__":
    discriminator_tester = LSTMCell(10)

    print('ran discriminator successfully')


    
