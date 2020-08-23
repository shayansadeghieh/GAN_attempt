import tensorflow.compat.v1 as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import pandas as pd
import numpy as np

#Import custom classes 
from linear_class import _Linear

class DiscriminatorLSTMCell(rnn_cell_impl.RNNCell):
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
        super(DiscriminatorLSTMCell, self).__init__()
        if not state_is_tuple:
                logging.warn("%s: Using a concatenated state is slower and will soon be "
                        "deprecated.  Use state_is_tuple=True.", self)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None

    #Required because RNNCell is parent and 
    # is an abstract class. However, could be unused. 
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units 
    
    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
        inputs: `2-D` tensor with shape `[batch_size x input_size]`.
        state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.
        Returns:
        A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        sigmoid = math_ops.sigmoid

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)


if __name__ == "__main__":
    discriminator_tester = DiscriminatorLSTMCell(10)

    print('ran discriminator successfully')


    
