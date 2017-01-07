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
                 max_decoding_steps, beam_width, dropout_settings):
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
            dropout_settings)

        #Intoduce the zero_state variables, which are used during decoding.
        self.zero_state = None
        self.zero_state_lengths = None

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

            self.zero_state = cell_state
            # get the attend and spell state variable lengths 
            # and store them to facilitate later list to tensor conversions.
            self.zero_state_lengths = cell_state.get_element_lengths()

            #beam_search_attend_and_spell.
            with tf.variable_scope('attend_and_spell'):

                time = tf.constant(0, tf.int32, shape=[])

                probs = tf.log(tf.ones([self.beam_width], tf.float32))
                selected = tf.ones([self.beam_width, 1], tf.int32)
                sequence_length = tf.ones(self.beam_width, tf.int32)
                states = BeamList()
                for _ in range(self.beam_width):
                    states.append(cell_state)

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
                #selected = tf.Print(selected, [selected[0], selected[1],
                #                               selected[2],
                #                               selected[self.beam_width-1]],
                #                    message='        Beams',
                #                    summarize=self.max_decoding_steps)
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


    def expand_beam(self, time,  states, probs, done_mask, sequence_length):
        """
        Expand the beam considering beam width options per beam.
        To expand all options set the beam width to self.target_label_no.
        Args:
            time: The decoding time scalar [].
            states: The state list used by the attend and spell cell.
            probs: The beam element log-probability sum vector [self.beam_width].
            done_mask: Mask true if corresponding beam element selection contains
                       an <eos> token.
            sequence_length: Vector indicating the number of labels until the
                             first <eos> token of the corresponding
                             beam elememnt. The length is equal to time, if 
                             the corresponding done mask element is false.

        Returns:
            prob_tensor: Log probability sum vector of current and past labels
                         [beam_width*beam_width]
            new_sel_tensor: Vector containing new selected labels 
                         [beam_width*beam_width]
            beam_pos_tensor: Vector containing the beam of origin for each label
                             and probability [beam_width*beam_width]. 
            states_new: List containing the attend and spell cell state. 
                        The one_hot_char entry in the list containts the selected
                        label. It must be updatet after pruning.
        """
        #------------------ expand ------------------------------#
        expanded_probs = tf.TensorArray(dtype=tf.float32,
                                        size=self.beam_width,
                                        name='expanded_current_probs')
        expanded_selected = tf.TensorArray(dtype=tf.int32,
                                           size=self.beam_width,
                                           name='expanded_selected')
        beam_pos = tf.TensorArray(dtype=tf.int32,
                                  size=self.beam_width,
                                  name='beam_pos')
        states_new = BeamList()

        for beam_no, cell_state in enumerate(states):
            logits, cell_state = \
                self.attend_and_spell_cell(None, cell_state)
            states_new.append(cell_state)

            full_probs = tf.nn.softmax(tf.squeeze(logits))
            best_probs, selected_new = tf.nn.top_k(full_probs,
                                                   k=self.beam_width,
                                                   sorted=True)

            length = tf.cast(sequence_length[beam_no], tf.float32)
            #pylint: disable=W0640
            def update():
                """ Compute the beam probability using the newly found
                    label probs """
                update_val = (probs + tf.log(best_probs)) / length
                return update_val 

            def const(): 
                """ The probability for a finished beam is 
                    sum(log(probs))/length """
                update_val = (probs + tf.zeros(self.beam_width) / length)
                return update_val

            new_beam_prob = tf.cond(done_mask[beam_no], const, update)            
            expanded_probs = expanded_probs.write(beam_no,
                                                  new_beam_prob)
            expanded_selected = expanded_selected.write(beam_no,
                                                        selected_new)
            beam_pos = beam_pos.write(beam_no, tf.ones(
                self.beam_width, tf.int32)*beam_no)


        # make shure while time <beam_width 
        # only relevant beams are considered
        # if time < beam_width then time is selected.
        beam_count = tf.select(time < self.beam_width, 
                               time,
                               self.beam_width)
        expanded_probs_tensor = expanded_probs.gather(
            tf.range(0, beam_count))            
        expanded_selected_tensor = expanded_selected.gather(
            tf.range(0, beam_count))
        beam_pos_tensor = beam_pos.gather(
            tf.range(0, beam_count))
        prob_tensor = tf.reshape(expanded_probs_tensor, [-1])
        new_sel_tensor = tf.reshape(expanded_selected_tensor, [-1])
        beam_pos_tensor = tf.reshape(beam_pos_tensor, [-1])

        return prob_tensor, new_sel_tensor, beam_pos_tensor, states_new

    def prune_beam(self, prob_tensor, new_sel_tensor, old_selected,
                   beam_pos_tensor, done_mask, sequence_length):
        """
        Prune the beam retaining only the beam width most probable options.

        Args:
            prob_tensor: Log probability sum vector of current and past labels
                         [beam_width*beam_width]
            new_sel_tensor: Vector containing new selected labels 
                         [beam_width*beam_width]
            old_selected: [beam_width, time-1] tensor containing the selected labels
                          for each beam element until time-1.
            beam_pos_tensor: Vector containing the beam of origin for each label.
            done_mask: Mask true if corresponding beam element selection contains
                       an <eos> token.
            sequence_length: Vector indicating the number of labels until the
                             first <eos> token of the corresponding
                             beam elememnt. The length is equal to time, if 
                             the corresponding done mask element is false.

        Returns:
            Pruned or reshuffeled versions of very input vector. 
        """

        probs, stay_indices = tf.nn.top_k(prob_tensor,
                                          k=self.beam_width,
                                          sorted=True)

        # use the stay_indices to gather from expanded tensors, this produces reduced size
        # output vectors.
        new_selected = tf.gather(new_sel_tensor, stay_indices)

        beam_pos_selected = tf.gather(beam_pos_tensor, stay_indices)
        # use the beam_pos to gather from old beam data. These operations do not change the vector
        # sizes.
        old_selected = tf.gather(old_selected, beam_pos_selected)
        probs = tf.gather(probs, beam_pos_selected)
        done_mask = tf.gather(done_mask, beam_pos_selected)
        sequence_length = tf.gather(sequence_length, beam_pos_selected)

        return probs, new_selected, old_selected, beam_pos_selected, done_mask, sequence_length 


    def body(self, loop_vars):
        ''' The body of the decoding while loop. Contains a manual enrolling
            of the attend and spell computations.
        Arguments:
            The loop variables from the previous iteration.
        Returns:
            The loop variables as computed during the current iteration.
        '''

        probs, old_selected, states, \
            time, sequence_length, done_mask = loop_vars
        time = time + 1

        #beam expansion.
        prob_expanded, sel_expanded, beam_element_positions, states_new = \
            self.expand_beam(time, states, probs, done_mask, sequence_length)            

        #beam pruning.
        probs, new_selected, old_selected, beam_pos_selected, done_mask, sequence_length = \
            self.prune_beam(prob_expanded, sel_expanded, old_selected,
                            beam_element_positions, done_mask, sequence_length)

        #update the sequence lengths.
        done_mask, sequence_length = self.get_sequence_lengths(
            time, new_selected, done_mask, sequence_length)

        #append the new selections.
        add_selected = tf.expand_dims(new_selected, 1)
        selected = tf.concat(1, [old_selected,
                                 add_selected])
        selected.set_shape([self.beam_width, None])

        #update the states
        #create a cell state tensor.
        state_tensor = tf.pack([state.to_tensor() for state in states_new])
        pos_lst = tf.unpack(beam_pos_selected)
        states = BeamList()
        for sel_no, pos in enumerate(pos_lst):
            new_state_tensor = state_tensor[pos, :]
            new_state = self.zero_state.to_list(new_state_tensor,
                                                self.zero_state_lengths)
            state = StateTouple(new_state.pre_context_states,
                                tf.expand_dims(tf.one_hot(new_selected[sel_no],
                                                          self.target_label_no),
                                               0),
                                new_state.context_vector,
                                new_state.alpha)
            states.append(state)
        out_vars = BeamList([probs, selected, states,
                             time, sequence_length, done_mask])
        return [out_vars]


