from __future__ import absolute_import, division, print_function

import collections
import tensorflow as tf
from tensorflow.python.util import nest

from neuralnetworks.las_elements import AttendAndSpellCell
from neuralnetworks.las_elements import DecodingTouple
from neuralnetworks.las_elements import StateTouple
from IPython.core.debugger import Tracer; debug_here = Tracer();


class BeamList(list):
    """ A list, which is supposed to hold the beam variables.
    """
    @property
    def dtype(self):
        """Check if the all internal state variables have the same data-type
           if yes return that type. """
        for i in range(1, len(self)):
            if self[i-1].dtype != self[i].dtype:
                raise TypeError("Inconsistent internal state: %s vs %s" %
                                (str(self[i-1].dtype), str(self[i].dtype)))
        return self[0].dtype

    @property
    def _shape(self):
        """ Make shure tf.Tensor.get_shape(this) returns  the correct output.
        """
        return self.get_shape()

    def get_shape(self):
        """ Return the shapes of the elements contained in the state tuple. """
        flat_shapes = []
        flat_self = nest.flatten(self)
        for i in range(0, len(flat_self)):
            flat_shapes.append(tf.Tensor.get_shape(flat_self[i]))
        shapes = nest.pack_sequence_as(self, flat_shapes)
        return shapes


class BeamSearchSpeller(object):
    """
    The speller takes high level features and implements an attention based
    transducer to find the desired sequence labeling.
    """

    def __init__(self, as_cell_settings, batch_size, dtype, target_label_no,
                 max_decoding_steps, beam_width):
        """ Initialize the listener.

        Arguments:
            A speller settings object containing:
            decoder_state_size: The size of the decoder RNN
            feedforward_hidden_units: The number of hidden units in the ff nets.
            feedforward_hidden_layers: The number of hidden layers
                                       for the ff nets.
            net_out_prob: The network output reuse probability during training.
            type: If true a post context RNN is added to the tree.
        """
        self.as_set = as_cell_settings
        self.batch_size = batch_size
        self.dtype = dtype
        self.target_label_no = target_label_no
        self.max_decoding_steps = max_decoding_steps
        self.beam_width = beam_width
        self.attend_and_spell_cell = AttendAndSpellCell(
            self, self.as_set.decoder_state_size,
            self.as_set.feedforward_hidden_units,
            self.as_set.feedforward_hidden_layers,
            self.as_set.net_out_prob,
            self.as_set.type)

    def __call__(self, high_level_features, feature_seq_length, target_one_hot,
                 target_seq_length, decoding):
        """
        Arguments:
            high_level_features: The output from the listener
                                 [batch_size, max_input_time, listen_out]
            feature_seq_length: The feature sequence lengths [batch_size]
            target_one_hot: The one hot encoded targets
                                 [batch_size, max_target_time, label_no]
            target_seq_length: Target sequence length vector [batch_size]
            decoding: Flag indicating if a decoding graph must be set up.
        Returns:
            if decoding is not True
                logits: The output logits [batch_size, decoding_time, label_no]
                logits_sequence_length: The logit sequence lengths [batch_size]
            else:
                decoded_sequence: A vector containing the the decoded sequence
                decoded_sequence_length: a scalar indicating the length of that seq.
        """

        if decoding is not True:
            print('Adding training attend and spell computations ...')
            #training mode
            self.attend_and_spell_cell.set_features(high_level_features,
                                                    feature_seq_length)
            zero_state = self.attend_and_spell_cell.zero_state(
                self.batch_size, self.dtype)
            logits, _ = tf.nn.dynamic_rnn(cell=self.attend_and_spell_cell,
                                          inputs=target_one_hot,
                                          initial_state=zero_state,
                                          sequence_length=target_seq_length,
                                          scope='attend_and_spell')
            logits_sequence_length = target_seq_length
            return logits, logits_sequence_length
        else:
            print('Adding beam search attend and spell computations ...')

            assert self.batch_size == 1, "beam search batch_size must be one."
            self.attend_and_spell_cell.set_features(high_level_features,
                                                    feature_seq_length)
            cell_state = self.attend_and_spell_cell.zero_state(
                self.batch_size, self.dtype)

            #beam_search_attend_and_spell.
            with tf.variable_scope('attend_and_spell'):

                time = tf.constant(0, tf.int32, shape=[])

                probs = tf.ones(self.beam_width, tf.float32)
                selected = tf.ones([self.beam_width, 1], tf.int32)
                sequence_length = tf.ones(self.beam_width, tf.int32)
                
                #TODO:FIXME.
                states = BeamList()
                for _ in range(self.beam_width):
                    states.append(cell_state.to_tensor())
                cell_state.load_tensor(states[0][0],states[0][1])                    
                
                done_mask = tf.cast(tf.zeros(self.beam_width), tf.bool)                
                loop_vars = BeamList([probs, selected, states,
                                      time, sequence_length, done_mask])

                #set up the shape invariants for the while loop.
                shape_invariants = loop_vars.get_shape()
                flat_invariants = nest.flatten(shape_invariants)
                flat_invariants[1] = tf.TensorShape([self.beam_width,
                                                     None])
                shape_invariants = nest.pack_sequence_as(shape_invariants,
                                                         flat_invariants)
                result = tf.while_loop(
                    self.cond, self.body, loop_vars=[loop_vars],
                    shape_invariants=[shape_invariants])
                probs, selected, states, \
                    time, sequence_length, done_mask = result[0]

                # Select the beam with the largest probability here.
                # The beam is sorted according to probabilities in descending
                # order. 
                # The most likely sequence is therefore located at position zero.
                selected = tf.Print(selected, [selected[0]],
                                    message='Top beam',
                                    summarize=5)
                selected = tf.Print(selected, [selected[1]],
                                    message='Second beam',
                                    summarize=5)
                selected = tf.Print(selected, [probs],
                                    message='top beam probabilities',
                                    summarize=self.beam_width)
                return selected[0], sequence_length[0]
                
        

    def cond(self, loop_vars):
        """ Condition in charge of the attend and spell decoding
            while loop. It checks if all the spellers in the current batch
            are confident of having found an <eos> token or if a maximum time
            has been exceeded.
        Arguments:
            loop_vars: The loop variables.
        Returns:
            keep_working, true if the loop should continue.
        """
        _, _, _, time, _, done_mask = loop_vars

        #the encoding table has the eos token ">" placed at position 0.
        #i.e. ">", "<", ...
        not_done_no = tf.reduce_sum(tf.cast(tf.logical_not(done_mask),
                                            tf.int32))
        all_eos = tf.equal(not_done_no, tf.constant(0))
        stop_loop = tf.logical_or(all_eos, tf.greater(time,
                                                      self.max_decoding_steps))
        keep_working = tf.logical_not(stop_loop)
        #keep_working = tf.Print(keep_working, [keep_working, sequence_length])
        return keep_working


    def get_sequence_lengths(self, time, max_vals, done_mask,
                             logits_sequence_length):
        """
        Determine the sequence length of the decoded logits based on the
        greedy decoded end of sentence token probability, the current time and
        a done mask, which keeps track of the first appearance of an end of
        sentence token.

        Arguments:
            time: The current time step [].
            max_vals: The max_vals labels numbers used during beam search
                    [beam_size, label_no].
            done_mask: A boolean mask vector of size [batch_size]
            logits_sequence_length: An integer vector with [batch_size] entries.
        Return:
            Updated versions of the logits_sequence_length and mask
            vectors with unchanged sizes.
        """
        with tf.variable_scope("get_sequence_lengths"):
            mask = tf.equal(max_vals, tf.constant(0, tf.int32))
            #current_mask = tf.logical_and(mask, tf.logical_not(done_mask))

            time_vec = tf.ones(self.beam_width, tf.int32)*(time+1)
            logits_sequence_length = tf.select(done_mask,
                                               logits_sequence_length,
                                               time_vec)
            done_mask = tf.logical_or(mask, done_mask)
        return done_mask, logits_sequence_length

    def body(self, loop_vars):
        ''' The body of the decoding while loop. Contains a manual enrolling
            of the attend and spell computations.
        Arguments:
            The loop variables from the previous iteration.
        Returns:
            The loop variables as computed during the current iteration.
        '''

        beam_probs, selected, states, \
            time, sequence_length, done_mask = loop_vars
        time = time + 1

        #expand the beam
        expanded_beam_probs_lst = BeamList()
        expanded_selected_lst = BeamList()
        beam_pos_lst = BeamList()
        states_new = BeamList()

        for beam_no, cell_state in enumerate(states):
            logits, cell_state = \
                self.attend_and_spell_cell(None, cell_state)
            states_new.append(cell_state)
            full_probs = tf.nn.softmax(tf.squeeze(logits))
            best_probs, selected_new = tf.nn.top_k(full_probs,
                                                   k=self.beam_width,
                                                   sorted=True)
            new_beam_prob = tf.sqrt(best_probs * beam_probs[beam_no])
            expanded_beam_probs_lst.append(new_beam_prob)
            expanded_selected_lst.append(selected_new)
            beam_pos_lst.append(tf.ones(self.beam_width)*beam_no)


        #prune away the unlikely expansions
        prob_tensor = tf.concat(0, expanded_beam_probs_lst)
        sel_tensor = tf.concat(0, expanded_selected_lst)
        beam_pos_tensor = tf.cast(tf.concat(0, beam_pos_lst), tf.int32)
        best_probs, stay_indices = tf.nn.top_k(prob_tensor,
                                               k=self.beam_width,
                                               sorted=True)

        new_selected = tf.gather(sel_tensor, stay_indices)
        beam_pos_selected = tf.gather(beam_pos_tensor, stay_indices)
        old_selected = tf.gather(selected, beam_pos_selected)

        selected = tf.concat(1, [old_selected,
                                 tf.expand_dims(new_selected, 1)])
        selected.set_shape([self.beam_width, None])

        #update the sequence lengths.
        done_mask, sequence_length = self.get_sequence_lengths(
            time, new_selected, done_mask, sequence_length)

        #update the states
        #TODO: A tensor cannot be a list index. Find another way to 
        # do this?!?!?!?!.
        pos_lst = tf.unpack(beam_pos_selected)
        states = BeamList()
        for sel_no, pos in enumerate(pos_lst):
            state = StateTouple(states_new[pos].pre_context_states,
                                states_new[pos].post_context_states,
                                tf.expand_dims(tf.one_hot(new_selected[sel_no],
                                                          self.target_label_no),
                                               0),
                                states_new[pos].context_vector)
            states.append(state)

        out_vars = BeamList([best_probs, selected, states,
                             time, sequence_length, done_mask])
        return [out_vars]
