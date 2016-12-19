from __future__ import absolute_import, division, print_function

#testing the state touple class.
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from IPython.core.debugger import Tracer; debug_here = Tracer();

#create a tf style cell state tuple object to derive the actual tuple from.
_AttendAndSpellStateTouple = \
    collections.namedtuple(
        "AttendAndSpellStateTouple",
        "pre_context_states, post_context_states, one_hot_char, context_vector"
        )

class StateTouple(_AttendAndSpellStateTouple):
    """ Tuple used by Attend and spell cells for `state_size`,
     `zero_state`, and output state.
      Stores four elements:
      `(pre_context_states, post_context_states, one_hot_char,
            context_vector)`, in that order.
    """
    @property
    def dtype(self):
        """ Check if the all internal state variables have the same data-type
            if yes return that type. """
        for i in range(1, len(self)):
            if self[i-1].dtype != self[i].dtype:
                raise TypeError("Inconsistent internal state: %s vs %s" %
                                (str(self[i-1].dtype), str(self[i].dtype)))
        return self[0].dtype

    @property
    def _shape(self):
        """ Make shure tf.Tensor.get_shape(this) returns  the correct output.
        """
        return self.get_shape()

    def get_shape(self):
        """ Return the shapes of the elements contained in the state tuple. """
        flat_shapes = []
        flat_self = nest.flatten(self)
        for i in range(0, len(flat_self)):
            flat_shapes.append(tf.Tensor.get_shape(flat_self[i]))
        shapes = nest.pack_sequence_as(self, flat_shapes)
        return shapes

    def to_tensor(self):
        """ This op turns the touple into a tensor.
        Returns:
            state_tensor: A tensor with the list contents concatenated along one dimension.
            element_lengths: A tensor with the length of each element in the state_tensor.
        """
        with tf.variable_scope("StateTouple_to_StateTensor"):
            flat_self = nest.flatten(self)
            
            squeezed_tensors = []
            for i in range(0, len(flat_self)):
                # state touple elements have the shape [Dimension(1), Dimension(?)].
                # the ? depends on the network parameters. The zeroth dimension is not 
                # interesting.
                squeezed_element = tf.squeeze(flat_self[i], [0])
                squeezed_tensors.append(squeezed_element)
            state_tensor = tf.concat(0, squeezed_tensors)
            return state_tensor

    def get_element_lengths(self):
        """
        Get the length of individual state elements as found in a concatenated
        state Tensor.
        Returns:
            The element lengths of the single touple entries. 
        """
        flat_self = nest.flatten(self)            
        element_lengths = []
        for i in range(0, len(flat_self)):
            # state touple elements have the shape [Dimension(1), Dimension(?)].
            # the ? depends on the network parameters. The zeroth dimension is not 
            # interesting.
            squeezed_element = tf.squeeze(flat_self[i], [0])
            element_lengths.append(int(tf.Tensor.get_shape(squeezed_element)[0]))
        return element_lengths

    def to_list(self, state_tensor, element_lengths):
        """
        Take a state tensor and pack it into a list with the same
        structure of self.
        WARNING: The code assumes that the state_tensor input does indeed
                 fit into the state list and that tensor sizes in element_lengths
                 are correct.
                 This assumption is not checked, if it's violated strange
                 things will happen.
        Args:
            state_tensor: A concatinated attend and spell state tensor.
            element_lengths: The element lengths of the state tensor.
        Returns:
            A repacked AttendAndSpellStateTouple container object.
        """
        
        with tf.variable_scope("StateTensor_to_StateTouple"):
            flat_self = []
            start = 0
            for length in element_lengths:
                stop = start + length
                list_element = tf.gather(state_tensor, tf.range(start, stop))
                list_element = tf.reshape(list_element, [1, length])
                flat_self.append(list_element)
                start = stop
        return nest.pack_sequence_as(self, flat_self)


one = tf.reshape(tf.constant(np.array([1])), [1, 1])
two = tf.reshape(tf.constant(np.array([2, 3])), [1, 2])
three = tf.reshape(tf.constant(np.array([4, 5, 6])), [1, 3])
four = tf.reshape(tf.constant(np.array([7, 8, 9, 10])), [1, 4])

test = StateTouple(one, two, three, four)

one = tf.reshape(tf.constant(np.array([10])), [1, 1])
two = tf.reshape(tf.constant(np.array([9, 8])), [1, 2])
three = tf.reshape(tf.constant(np.array([7, 6, 5])), [1, 3])
four = tf.reshape(tf.constant(np.array([4, 3, 2, 1])), [1, 4])

test_two = StateTouple(one, two, three, four)

tensor_test_lengths = test.get_element_lengths()
tensor_test = test.to_tensor()
tensor_test2 = test_two.to_tensor()
print(tensor_test)
original = test.to_list(tensor_test, tensor_test_lengths)
load_other = test.to_list(tensor_test2, tensor_test_lengths)
other_tensor = load_other.to_tensor()




sess = tf.Session()
with sess.as_default():
    tf.initialize_all_variables().run()
    #print(char_prob.eval())
    print('tensor_test', tensor_test.eval())
    print('other_tensor', other_tensor.eval())










