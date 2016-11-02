import numpy as np
import tensorflow as tf

#A time minor tensor [batch_size, time, input_size]

#shape (2,3) =>two time steps, three chars.
utt_one = np.array([[1, 2, 3], [4, 5, 6]])
utt_two = np.array([[7, 8, 9], [10, 11, 12]])

input_tensor = np.array([utt_one, utt_two])

zero_init = tf.constant_initializer(0)
tensor_init = tf.constant_initializer(input_tensor)
vec_lst = []

inputs = tf.get_variable('input', [2, 2, 3], dtype=tf.float32,
                         initializer=tensor_init)

concat = tf.concat(1, [inputs[:, 0, :], inputs[:, 1, :]])

init_op = tf.initialize_all_variables()

sess = tf.Session()
with sess.as_default():
    sess.run(init_op)

    concat_out = concat.eval()
    print(concat_out)
    print(concat_out.shape)