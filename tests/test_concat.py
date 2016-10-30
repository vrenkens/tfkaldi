import numpy as np
import tensorflow as tf

char_mat = np.zeros([5, 6])
char_vec = [1.0, 1.0, 0.0, 0.5, 0.5]

zero_init = tf.constant_initializer(0)
vec_init = tf.constant_initializer(char_vec)
mat_init = tf.constant_initializer(char_mat)
vec_lst = []

i = tf.get_variable('i', [], dtype=tf.int32, initializer=zero_init)
i = i + 1

char_vec_tf_old = tf.get_variable('test_old', [1, 5], dtype=tf.float32,
                                  initializer=vec_init)
char_vec_tf = tf.get_variable('test', [1, 5], dtype=tf.float32,
                              initializer=vec_init)

concat = tf.concat(0, [char_vec_tf_old, char_vec_tf])
scaled_vec = tf.nn.softmax(char_vec_tf)
prob_zero = tf.exp(char_vec_tf[:, 0]) / tf.reduce_sum(tf.exp(char_vec_tf))

init_op = tf.initialize_all_variables()
print(tf.Tensor.get_shape(i))

sess = tf.Session()
with sess.as_default():
    sess.run(init_op)
    print(i.eval())
    print(concat.eval())
    print(scaled_vec.eval())
    print(prob_zero.eval())
