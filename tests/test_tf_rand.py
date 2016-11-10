#pylint: disable=C0111
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.device('/gpu:1'):
    zero = tf.constant(0.0)
    one = tf.constant(1.0)

    def pick_ground_trouth():
        return zero
    def pick_last_output():
        return one
    def pred():
        return tf.greater(rand_val, 0.2)

    val_lst = []
    for _ in range(0, 1000):
        rand_val = tf.random_uniform(shape=(), minval=0.0, maxval=1.0,
                                     dtype=tf.float32, seed=None,
                                     name='random_uniform')
        selected = tf.cond(pred(), pick_ground_trouth, pick_last_output,
                           name='truth_or_output_sel')
        val_lst.append(selected)
    vals = tf.convert_to_tensor(val_lst)


#self.zero_chars =
sess = tf.Session()
with sess.as_default():
    #tf.initialize_all_variables()
    np_values = vals.eval()
    print(np.sum(np_values)/len(np_values))

# approx 0.2 is a good output, because zero is the ground trough value here,
# which is picked 0.8 of the time.
