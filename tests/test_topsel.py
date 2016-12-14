import numpy as np
import tensorflow as tf

beam_width = 2
target_label_no = 4

char_probs = tf.constant([0.1, 0.2, 0.3, 0.4], tf.float32)

top_char_probs, top_char_ind = tf.nn.top_k(char_probs, k=beam_width)
top_chars_hot = tf.one_hot(top_char_ind[1], target_label_no)
#top_probs = char_probs[top_char_pos]

sess = tf.Session()
with sess.as_default():
    print(char_probs.eval())
    print(top_char_ind.eval())
    print(top_chars_hot.eval())
