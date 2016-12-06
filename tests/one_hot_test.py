import numpy as np
import tensorflow as tf



class one_hot_test(object):
    """
    Testing the one hot encoding function..
    """
    def __init__(self, target_label_no):
        self.target_label_no = target_label_no

    def encode_targets_one_hot(self, targets):
        """
        Transforn the targets into one hot encoded targets.
        Args:
            targets: Tensor of shape [batch_size, max_target_time, 1]
        Returns:
            one_hot_targets: [batch_size, max_target_time, label_no]
        """
        with tf.variable_scope("one_hot_encoding"):
            target_one_hot = tf.one_hot(targets,
                                        self.target_label_no,
                                        axis=2)
            #one hot encoding adds an extra dimension we don't want.
            #squeeze it out.
            target_one_hot = tf.squeeze(target_one_hot, squeeze_dims=[3])
            print("train targets shape: ", tf.Tensor.get_shape(target_one_hot))
        return target_one_hot

one_hot_encoder = one_hot_test(10)

test_target = [[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9], 
               [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]

np_targets = np.array(test_target)
targets = tf.constant(np_targets)
targets = tf.expand_dims(targets, 2)
print(tf.Tensor.get_shape(targets))

one_hot_targets = one_hot_encoder.encode_targets_one_hot(targets)

init_op = tf.initialize_all_variables()

sess = tf.Session()
with sess.as_default():
    sess.run(init_op)
    print(one_hot_targets.eval())




