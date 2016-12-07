import tensorflow as tf



eos_prob_vec = [1.0, 1.0, 1.0, 1.0, 0.2]

prob_init = tf.constant_initializer(eos_prob_vec)
i = 99
max_target_length = 100
eos_treshold = 0.8

eos_prob = tf.get_variable('test', shape=[5, 1], dtype=tf.float32,
                           initializer=prob_init)

#loop_continue_conditions = tf.logical_and(tf.less(eos_prob, eos_treshold),
#                                          tf.less(i, max_target_length))

sequence_length_init = [100, 50, 10, 20, 40]
seq_init = tf.constant_initializer(sequence_length_init)
sequence_length = tf.get_variable('test2', dtype=tf.int32,
                                  initializer=sequence_length_init)
time = tf.constant(30)


elements_finished = (time >= sequence_length)
finished = tf.reduce_all(elements_finished)


loop_continue_conditions = tf.less(eos_prob, eos_treshold)

loop_continue_counter = tf.reduce_sum(tf.to_int32(loop_continue_conditions))
keep_working = tf.not_equal(loop_continue_counter, 0)
finished_eos = tf.logical_not(keep_working)

max_time_tensr = tf.reduce_max(sequence_length)

init_op = tf.initialize_all_variables()

sess = tf.Session()
with sess.as_default():
    sess.run(init_op)
    print(loop_continue_counter.eval())
    print(finished_eos.eval())
    print(finished.eval())
    print(max_time_tensr.eval())
