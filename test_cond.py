import tensorflow as tf

eos_prob_vec = [1.0, 1.0, 0.0, 0.5, 0.5]

prob_init = tf.constant_initializer(eos_prob_vec)
i = 99
max_target_length = 100
eos_treshold = 0.8

eos_prob = tf.get_variable('test', [5, 1], dtype=tf.float32,
                           initilaizer=prob_init)

loop_continue_conditions = tf.logical_and(tf.less(eos_prob, eos_treshold),
                                          tf.less(i, max_target_length))

loop_continue_counter = tf.reduce_sum(tf.to_int32(loop_continue_conditions))
keep_working = tf.not_equal(loop_continue_counter, 0)


init_op = tf.initialize_all_variables()

sess = tf.Session()
with sess.as_default():
    sess.run(init_op)
    print(loop_continue_counter.eval())
    print(keep_working.eval())
