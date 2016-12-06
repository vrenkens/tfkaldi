import numpy as np
import tensorflow as tf

batches = 5
labels = 32

shape = tf.constant([batches, labels], dtype=tf.int64)
values = tf.constant([1]*batches, dtype=tf.int64)

max_pos = [0, 1, 2, 3, 4]
idx = []
for batch_no, char_no in enumerate(max_pos):
    idx.append([batch_no, char_no])
idx = tf.convert_to_tensor(idx, dtype=tf.int64)

one_hot_char = tf.SparseTensor(idx, values, shape)
one_hot_char = tf.sparse_tensor_to_dense(one_hot_char)

init_op = tf.initialize_all_variables()

np_shape = tf.contrib.util.constant_value(shape)
print(np_shape)

#self.zero_chars =
sess = tf.Session()
with sess.as_default():
    sess.run(init_op)
    print(one_hot_char.eval())
