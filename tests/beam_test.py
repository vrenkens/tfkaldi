from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

from IPython.core.debugger import Tracer; debug_here = Tracer();

beam_width = 4
max_it = tf.constant(6)

char_mat_1 = [[0.0, 0.0, 0.5, 0.6, 0.0, 0.0],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.11],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.11],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.11]]

char_mat_2 = [[0.0, 0.0, 0.5, 0.6, 0.0, 0.0],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.11],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.11],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.11]]

char_mat_3 = [[0.1, 0.0, 0.9, 0.0, 0.0, 0.0],
              [0.0, 0.1, 0.2, 0.0, 0.0, 0.0],
              [0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.3, 0.2, 0.1, 0.1]]

char_mat_4 = [[0.0, 0.0, 0.1, 0.9, 0.0, 0.0],
              [0.2, 0.2, 0.5, 0.1, 0.0, 0.0],
              [0.4, 0.3, 0.2, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.1, 0.1, 0.1, 0.2]]

char_mat_5 = [[0.9, 0.0, 0.1, 0.0, 0.0, 0.0],
              [0.5, 0.1, 0.1, 0.3, 0.0, 0.0],
              [0.5, 0.1, 0.1, 0.3, 0.0, 0.0],
              [0.5, 0.1, 0.1, 0.3, 0.0, 0.0]]

#expected output: [5, 2, 4, 5, 4]

char_lst = [char_mat_1, char_mat_2, char_mat_3,
            char_mat_4, char_mat_5]
np_char_tensor = np.array(char_lst)

char_prob = tf.constant(np.array(np_char_tensor), tf.float32)
#char_prob = tf.transpose(char_prob, [1, 0, 2])
print(tf.Tensor.get_shape(char_prob))
sequence_length_lst = [1]*beam_width
sequence_length = tf.constant(sequence_length_lst)
done_mask = tf.cast(tf.zeros(beam_width), tf.bool)
beam_probs = tf.ones(beam_width)
selected = tf.expand_dims(tf.ones(beam_width, tf.int32), 1)
#selected = tf.expand_dims(tf.constant([0, 1, 1, 1]), 1)

for time in range(0, 5):
    print(time)

    #beam expansion step.
    expanded_beam_probs_lst = []
    expanded_selected_lst = []
    beam_pos_lst = []
    for beam_no in range(0, beam_width):
        full_probs = tf.nn.softmax(char_prob[time, beam_no, :])
        best_probs, selected_new = tf.nn.top_k(full_probs, k=beam_width,
                                               sorted=True)
        new_beam_prob = tf.sqrt(best_probs * beam_probs[beam_no])
        expanded_beam_probs_lst.append(new_beam_prob)
        expanded_selected_lst.append(selected_new)
        beam_pos_lst.append(tf.ones(beam_width)*beam_no)

    prob_tensor = tf.concat(0, expanded_beam_probs_lst)
    sel_tensor = tf.concat(0, expanded_selected_lst)
    beam_pos_tensor = tf.cast(tf.concat(0, beam_pos_lst), tf.int32)
    best_probs, stay_indices = tf.nn.top_k(prob_tensor, k=beam_width,
                                           sorted=True)

    new_selected = tf.gather(sel_tensor, stay_indices)
    beam_pos_selected = tf.gather(beam_pos_tensor, stay_indices)
    old_selected = tf.gather(selected, beam_pos_selected)
    selected = tf.concat(1, [old_selected, tf.expand_dims(new_selected, 1)])

    mask = tf.equal(new_selected, tf.constant(0, tf.int32))
    current_mask = tf.logical_and(mask, tf.logical_not(done_mask))
    done_mask = tf.logical_or(mask, done_mask)
    time_vec = tf.ones(beam_width, tf.int32)*(time+2)
    sequence_length = tf.select(done_mask,
                                sequence_length,
                                time_vec,
                                name=None)

    not_done_no = tf.reduce_sum(tf.cast(tf.logical_not(done_mask), tf.int32))
    all_eos = tf.equal(not_done_no, tf.constant(0))
    stop_loop = tf.logical_or(all_eos, tf.greater(time, max_it))
    keep_working = tf.logical_not(stop_loop)

sess = tf.Session()
with sess.as_default():
    tf.initialize_all_variables().run()
    #print(char_prob.eval())
    print('Prob tensor', prob_tensor.eval())
    print('Sel tensor', sel_tensor.eval())
    print(new_selected.eval())
    print(beam_pos_selected.eval())
    print(best_probs.eval())
    print(stay_indices.eval())
    print(selected.eval())
    print(mask.eval())
    print(done_mask.eval())
    print(sequence_length.eval())
    print(keep_working.eval())
