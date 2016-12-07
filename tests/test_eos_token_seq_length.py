from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

from IPython.core.debugger import Tracer; debug_here = Tracer();

batch_size = 5
max_it = tf.constant(6)


char_mat_0 = [[0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.9, 0.0, 0.0]]

char_mat_1 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

char_mat_2 = [[0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

char_mat_3 = [[0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

char_mat_4 = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

char_lst = [char_mat_0, char_mat_1, char_mat_2,
            char_mat_3, char_mat_4]
np_char_tensor = np.array(char_lst)

char_prob = tf.constant(np.array(np_char_tensor), tf.float64)
char_prob = tf.transpose(char_prob, [1, 0, 2])
print(tf.Tensor.get_shape(char_prob))
sequence_length_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
sequence_length = tf.constant(sequence_length_lst)
done_mask = tf.cast(tf.zeros(batch_size), tf.bool)

for time in range(0, 4):
    print(time)
    current_date = char_prob[:, time, :]
    max_vals = tf.argmax(current_date, 1)
    mask = tf.equal(max_vals, tf.constant(0, tf.int64))

    current_mask = tf.logical_and(mask, tf.logical_not(done_mask))
    done_mask = tf.logical_or(mask, done_mask)

    time_vec = tf.ones(batch_size)*time
    sequence_length = tf.select(current_mask, time_vec, sequence_length, name=None)

    not_done_no = tf.reduce_sum(tf.cast(tf.logical_not(done_mask), tf.int32))
    all_eos = tf.equal(not_done_no, tf.constant(0))
    stop_loop = tf.logical_or(all_eos, tf.greater(time, max_it))

sess = tf.Session()
with sess.as_default():
    tf.initialize_all_variables().run()
    #print(char_prob.eval())
    print(max_vals.eval())
    print(mask.eval())
    print(done_mask.eval())
    print(sequence_length.eval())
    print(stop_loop.eval())
