from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

from IPython.core.debugger import Tracer; debug_here = Tracer();


#5 labels <eos>, <sos>, a, b, c#
start = tf.constant(5)
max_length = 15
test = tf.range(0, max_length)
testgather1 = tf.gather(test, tf.range(test[5], max_length))

test1 = tf.range(1, max_length + 1)
test2 = tf.range(2, max_length + 2)
test3 = tf.range(3, max_length + 3)

expand_test = tf.expand_dims(test, 1)
expand_test1 = tf.expand_dims(test1, 1)
expand_test2 = tf.expand_dims(test2, 1)
expand_test3 = tf.expand_dims(test3, 1)
test_mat = tf.concat(1, [expand_test,
                         expand_test1,
                         expand_test2,
                         expand_test3])
print(tf.Tensor.get_shape(test_mat))
debug_here()
testgather2 = tf.gather(test_mat, [0, 0, 2])
flat_mat = tf.reshape(test_mat, [-1])


sess = tf.Session()
with sess.as_default():
    print(test.eval())
    print(testgather1.eval())
    print(expand_test.eval())
    print(test_mat.eval())
    print(testgather2.eval())
    print(flat_mat.eval())
