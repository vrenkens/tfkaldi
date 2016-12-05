from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

batch_size = 5

char_mat_0 = [[0.0, 0.0, 0.2, 0.3, 0.9, 0.0],
              [0.0, 0.1, 0.2, 0.3, 0.9, 0.0],
              [0.0, 0.1, 0.4, 0.0, 0.9, 0.0],
              [0.1, 0.1, 0.1, 0.1, 0.9, 0.0],
              [0.1, 0.1, 0.0, 0.0, 0.9, 0.0]]

char_mat_1 = [[1.0, 0.0, 0.2, 0.3, 0.4, 0.0],
              [0.0, 1.0, 0.2, 0.3, 0.4, 0.0],
              [0.0, 0.1, 0.4, 0.0, 1.4, 0.0],
              [0.1, 0.1, 0.1, 0.1, 1.1, 0.0],
              [0.1, 0.1, 0.0, 0.0, 1.0, 0.0]]

char_mat_2 = [[0.0, 0.0, 0.2, 0.3, 0.4, 0.0],
              [0.0, 0.1, 0.2, 0.3, 0.4, 0.0],
              [0.0, 0.1, 0.4, 0.0, 0.4, 0.0],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
              [0.1, 0.1, 0.0, 0.0, 0.0, 0.0]]

char_lst = [char_mat_0, char_mat_1, char_mat_2]
np_char_tensor = np.array(char_lst)

char_prob = tf.constant(np.array(np_char_tensor), tf.float64)
char_prob = tf.reshape(char_prob, [5, 3, 6])
char_prob_exp = tf.expand_dim(char_prob)
char_prob_test = tf.squeeze(char_prob_exp)

sequence_length_lst = [0.0, 0.0, 0.0, 0.0, 0.0]
sequence_length = tf.constant(sequence_length_lst)
done_mask = tf.cast(tf.zeros(batch_size), tf.bool)

for time in range(0, 1):
    print(time)
    current_date = char_prob[:, time, :]
    max_vals = tf.argmax(current_date, 1)
    mask = tf.equal(max_vals, tf.constant(1, tf.int64))

    current_mask = tf.logical_and(mask, done_mask)
    done_mask = tf.logical_or(mask, done_mask)

    time_vec = tf.ones(batch_size)*time
    sequence_length = tf.select(current_mask, sequence_length, time_vec, name=None)


sess = tf.Session()
with sess.as_default():
    tf.initialize_all_variables().run()
    print(char_prob.eval())
    print(max_vals.eval())
    print(mask.eval())
    print(sequence_length.eval())

