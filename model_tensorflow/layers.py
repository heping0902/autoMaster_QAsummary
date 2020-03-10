import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Encoder(tf.keras.Model):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 word_embedding,
                 enc_units,
                 cell_type='GRU'):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.cell_type = cell_type
        # define embedding layer whose input is batch's single word
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[word_embedding], trainable=False)
        # define gru layers whose input is output of embedding layer
        self.gru_layer = tf.keras.layers.GRU(units=enc_units,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')
        self.lstm_layer = tf.keras.layers.LSTM(units=enc_units,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        """
        :param x: (batch_size, input_length) for embedding layer
        :param hidden: (batch_size, hidden_layer_output_units) for initial state of GRU or LSTM
        :return: output, state_h (or and state_c) of encoder layer
        """
        # print("+++++++++++++++")
        # print(x)
        # print(x.shape)
        # print("+++++++++++++++")
        inputs = self.embedding_layer(x)
        if 'GRU' == self.cell_type:
            # print(">>>>>>>>>>>>>>")
            # print(inputs.shape)
            # print(hidden.shape)
            # print("<<<<<<<<<<<<<<<")
            output, state_h = self.gru_layer(inputs, initial_state=hidden)
            return output, state_h

        elif 'LSTM' == self.cell_type:
            output, state_h, state_c = self.lstm_layer(inputs, initial_state=hidden)
            state = (state_h, state_c)
            return output, state

    def initialize_hidden_state(self, batch_size):
        if 'GRU' == self.cell_type:
            return tf.zeros((batch_size, self.enc_units))
        elif 'LSTM' == self.cell_type:
            return tf.zeros((batch_size, self.enc_units)), tf.zeros((batch_size, self.enc_units))


class Decoder(tf.keras.Model):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 word_embedding,
                 dec_units,
                 cell_type='GRU'):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.cell_type = cell_type
        # define embedding layer whose input is batch's single word
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[word_embedding], trainable=False)
        # define gru layers whose input is output of embedding layer
        self.gru_layer = tf.keras.layers.GRU(units=dec_units,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')
        self.lstm_layer = tf.keras.layers.LSTM(units=dec_units,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        """
        :param x: (batch_size, 1) for embedding layer (just input <start> word)
        :param hidden: (batch_size, hidden_layer_output_units) for initial state of GRU or LSTM
        :return: output, state (state_h, state_c when using LSTM) of decoder layer, shape of output is
        (batch_size, 1, hidden_layer_output_units), hidden_layer_output_units is often vocab_size.
        """
        # print(x)
        # print(x.shape)
        inputs = self.embedding_layer(x)
        # print("=============")
        # print(inputs)
        # print(inputs.shape)
        if 'GRU' == self.cell_type:
            output, state_h = self.gru_layer(inputs, initial_state=hidden)
            return output, state_h

        elif 'LSTM' == self.cell_type:
            output, state_h, state_c = self.lstm_layer(inputs, initial_state=hidden)
            state = (state_h, state_c)
            return output, state


class BahdananAttention(tf.keras.Model):

    def __init__(self, dense_units):
        super(BahdananAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(dense_units)
        self.W2 = tf.keras.layers.Dense(dense_units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, ht, hs):
        """
        #:param ht: Decoder hidden
        #:param hs: Encoder outputs
        #:return: context_vectors, attention_weights
        """
        w1_hs = self.W1(hs)
        w2_ht = self.W2(ht)
        tanh_w1_w2 = tf.nn.tanh(w1_hs + w2_ht)
        score = self.V(tanh_w1_w2)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * hs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = tf.expand_dims(context_vector, axis=1)
        return context_vector, attention_weights


class PointerGenerationNetwork(tf.keras.Model):

    def __init__(self):
        self.Wh = tf.keras.layers.Dense(1)
        self.Ws = tf.keras.layers.Dense(1)
        self.Wx = tf.keras.layers.Dense(1)

    def __call__(self, h, s, x):
        """
        :param h: context vector
        :param s: decoder state
        :param x: decoder input
        :return: Pgen, a soft switch to choose between generating a word from the vocab
        ot copying a word from the source input sequence.
        """
        Pgen = tf.nn.sigmoid(self.Wh(h) + self.Ws(s) + self.Wx(x))
        return Pgen


class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 word_embedding,
                 word2index,
                 units,
                 checkpoint_path,
                 with_attention=True,
                 cell_type='GRU',
                 print_batch=True):
        super(Seq2Seq, self).__init__()
        self.word_embedding = word_embedding
        self.word2index = word2index
        self.units = units
        self.checkpoint_path = checkpoint_path
        self.with_attention = with_attention
        self.cell_type = cell_type
        self.print_batch = print_batch

        self.gru_layer = tf.keras.layers.GRU(self.units,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')
        self.lstm_layer = tf.keras.layers.LSTM(units=units,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')
        self.optimizer = tf.keras.optimizers.Adam()
        self.calculate_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        """
        self.start = '<start>'
        self.end = '<end>'

        self.word2index[self.start] = len(self.word2index.keys())
        self.word2index[self.end] = len(self.word2index.keys())

        self.word_embedding = np.vstack((self.word_embedding, np.zeros(len(self.word_embedding[0])) + 0.1))
        self.word_embedding = np.vstack((self.word_embedding, np.zeros(len(self.word_embedding[0])) + 0.9))
        """
        self.embedding_dim = self.word_embedding.shape[1]
        self.vocab_size = self.word_embedding.shape[0]

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, weights=[word_embedding], trainable=False)

        self.encoder = Encoder(self.vocab_size,
                               self.embedding_dim,
                               self.word_embedding,
                               self.units,
                               self.cell_type)

        self.decoder = Decoder(self.vocab_size,
                               self.embedding_dim,
                               self.word_embedding,
                               self.units,
                               self.cell_type)

        self.attention = BahdananAttention(dense_units=units)
        self.pgn = PointerGenerationNetwork()
        self.dense = tf.keras.layers.Dense(units=self.vocab_size)

        self.checkpoint_prefix = os.path.join(checkpoint_path, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                              decoder=self.decoder,
                                              attention=self.attention,
                                              dense=self.dense,
                                              optimizer=self.optimizer)

    # 该loss函数计算的时候，是一个时间步一个时间步的计算，GPU利用率非常低，我们在train的时候，
    # 采取另一个办法，即先一次性计算完所有的时间步（即decoder的max_len），然后再一次性计算
    # 所有时间步的loss。该函数写在了LossFunction
    @staticmethod
    def LossFunction_step_by_step(y, y_hat, word2index):
        """
        calculate loss.
        :param y_hat: predict output (calculated output)
        :param y: real output (real output in fact, target word)
        :param word2index: word2index vocab
        :return: calculated result
        """
        y0 = y
        y_hat0 = y_hat
        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                              reduction='none')
        # convert y to a tensor
        mask = tf.logical_not(tf.math.equal(y, 0))
        # calculate loss with y and predictions y_hat.
        # described as docs, loss_object just can calculate that y is a single floating point
        # and y_hat is #classes floating point, so we need parameter mask to calculate multi
        # classes.
        loss = loss_object(y, y_hat0)
        # print(loss)

        # sum all loss, because the output of decoder may be many classes in some applications
        mask = tf.cast(mask, loss.dtype)  # convert the type of mask to be same with loss
        # print(mask)
        mask = tf.reshape(mask, [-1])
        loss1 = loss * mask
        return loss1  # return the mean of all losses

    @staticmethod
    def LossFunction(y, y_hat, word2index):
        """
        calculate loss.
        :param y_hat: predict output (calculated output)
        :param y: real output (real output in fact, target word)
        :param word2index: word2index vocab
        :return: calculated result
        """
        # 这个时候，y's shape is (batch_size, time_step, 1)
        y0 = y
        # y_hat是decoder中所有的时间步的输出，然后再输入到dense层
        # y_hat进入dense层之前的 shape is (batch_size, step_time T (也就是max_len), hidden_units of decoder)
        # 进入dense之后，输出的维度为(batch_size, target_max_len, vocab_size)
        y_hat0 = y_hat

        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                              reduction='none')
        # convert y to a tensor
        pad_index = word2index['<PAD>']
        # mask是将target sequence中所有的'<PAD>'定义为False,而所有非'<PAD>'的词为True，为之后清楚'<PAD>'奠定基础
        mask = tf.logical_not(tf.math.equal(y0, pad_index))

        # 这个时候计算decoder_length的目的就是把mask中所有的 1 加起来，结果的shape为（batch_size,）每个元素表示该batch个sequence中有效词(非'<PAD>')的个数
        decoder_length = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)

        # calculate loss with y and predictions y_hat.
        # described as docs, loss_object just can calculate that y is a single floating point
        # and y_hat is #classes floating point, so we need parameter mask to calculate multi
        # classes.
        loss = loss_object(y0, y_hat0)
        # convert the type of mask to be same as loss
        mask = tf.cast(mask, loss.dtype)
        # 除去<pad>
        loss1 = loss * mask
        loss2 = tf.reduce_sum(loss1, axis=-1)
        # 这里除以decoder_length就是为了计算除去<pad>之后剩下的词的loss的合（不计算<pad>的loss）
        loss3 = loss2 / decoder_length
        loss4 = tf.reduce_mean(loss3, axis=-1)
        return loss4  # return the mean of all losses

    # baseline seq2seq model + attention mechanism
    def train_one_step(self, input_sequence, target_sequence, batch_size):
        """
        :param: input_sequence: input sequence, shape is (batch_size, input_max_length)
        :param: target_sequence: target sequence, shape is (batch_size, target_max_len), but target word is just a word
        :return: batch_loss: average loss in one batch
        """
        """
        print("////////////////////")
        print(target_sequence)
        print(target_sequence[:, 2])
        print(target_sequence[:, 2].shape)
        print(target_sequence.shape)
        print(target_sequence)
        """
        all_decoder_output_list = []
        with tf.GradientTape() as tape:
            encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
            encoder_output, encoder_hidden = self.encoder(input_sequence, encoder_hidden)

            previous_decoder_hidden = encoder_hidden
            target_word = np.expand_dims(target_sequence[:, 0], axis=-1)
            for target_word_index in range(1, target_sequence.shape[1]):
                decoder_output, previous_decoder_hidden = self.decoder(target_word, previous_decoder_hidden)
                target_word = np.expand_dims(target_sequence[:, target_word_index], axis=-1)
                context_vector, attention_weights = self.attention(decoder_output, encoder_output)
                attention_vector = tf.concat([context_vector, decoder_output], axis=-1)
                # print(attention_vector)
                # shape of attention_vector is (batch_size, 1, output_units),
                # save every time step attention_vector.
                all_decoder_output_list.append(attention_vector)
            # average batch loss in this batch training
            all_decoder_output = tf.stack(all_decoder_output_list)
            all_decoder_output1 = tf.transpose(all_decoder_output, [1, 0, 2, 3])
            all_decoder_output2 = tf.squeeze(all_decoder_output1, [2])
            y_hat = self.dense(all_decoder_output2)
            loss = Seq2Seq.LossFunction(target_sequence[:, 1:], y_hat, self.word2index)
            accuracy = self.calculate_accuracy(target_sequence[:, 1:], y_hat)
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables\
                        + self.attention.trainable_variables + self.dense.trainable_variables
            gradients = tape.gradient(loss, variables)
            # a error: no gradients provide for any variable
            # resolution: need put all forward layer into GradientTape(), then the gradients will
            # be gotten.
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss, self.calculate_accuracy.result()

    # Add Point-Generation network to seq2seq model and attention mechanism
    def train_one_step_add_PGN(self, input_sequence, target_sequence, batch_size):
        """
        :param: input_sequence: input sequence, shape is (batch_size, input_max_length)
        :param: target_sequence: target sequence, shape is (batch_size, target_max_len), but target word is just a word
        :return: batch_loss: average loss in one batch
        """
        """
        print("////////////////////")
        print(target_sequence)
        print(target_sequence[:, 2])
        print(target_sequence[:, 2].shape)
        print(target_sequence.shape)
        print(target_sequence)
        """
        all_decoder_output_list = []
        with tf.GradientTape() as tape:
            encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
            encoder_output, encoder_hidden = self.encoder(input_sequence, encoder_hidden)

            previous_decoder_hidden = encoder_hidden
            target_word = np.expand_dims(target_sequence[:, 0], axis=-1)
            for target_word_index in range(1, target_sequence.shape[1]):
                decoder_output, previous_decoder_hidden = self.decoder(target_word, previous_decoder_hidden)
                target_word = np.expand_dims(target_sequence[:, target_word_index], axis=-1)
                context_vector, attention_weights = self.attention(decoder_output, encoder_output)
                attention_vector = tf.concat([context_vector, decoder_output], axis=-1)
                # print(attention_vector)
                # shape of attention_vector is (batch_size, 1, output_units),
                # save every time step attention_vector.
                all_decoder_output_list.append(attention_vector)
            # average batch loss in this batch training
            all_decoder_output = tf.stack(all_decoder_output_list)
            all_decoder_output1 = tf.transpose(all_decoder_output, [1, 0, 2, 3])
            all_decoder_output2 = tf.squeeze(all_decoder_output1, [2])
            y_hat = self.dense(all_decoder_output2)
            loss = Seq2Seq.LossFunction(target_sequence[:, 1:], y_hat, self.word2index)
            accuracy = self.calculate_accuracy(target_sequence[:, 1:], y_hat)
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables\
                        + self.attention.trainable_variables + self.dense.trainable_variables
            gradients = tape.gradient(loss, variables)
            # a error: no gradients provide for any variable
            # resolution: need put all forward layer into GradientTape(), then the gradients will
            # be gotten.
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss, self.calculate_accuracy.result()

    def validation_one_step(self, input_sequence, target_sequence, batch_size):
        """
        validate seq2seq model, calculate accuracy
        :param: input_sequence: input sequence, shape is (batch_size, input_max_length)
        :param: target_sequence: target sequence, shape is (batch_size, target_max_len), but target word is just a word
        :return: batch_loss: average loss in one batch
        """
        all_decoder_output_list = []
        encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
        encoder_output, encoder_hidden = self.encoder(input_sequence, encoder_hidden)

        previous_decoder_hidden = encoder_hidden
        target_word = np.expand_dims(target_sequence[:, 0], axis=-1)
        for target_word_index in range(1, target_sequence.shape[1]):
            decoder_output, previous_decoder_hidden = self.decoder(target_word, previous_decoder_hidden)
            target_word = np.expand_dims(target_sequence[:, target_word_index], axis=-1)
            context_vector, attention_weights = self.attention(decoder_output, encoder_output)
            attention_vector = tf.concat([context_vector, decoder_output], axis=-1)
            # print(attention_vector)
            # shape of attention_vector is (batch_size, 1, output_units),
            # save every time step attention_vector.
            all_decoder_output_list.append(attention_vector)
        # average batch loss in this batch training
        all_decoder_output = tf.stack(all_decoder_output_list)
        all_decoder_output1 = tf.transpose(all_decoder_output, [1, 0, 2, 3])
        all_decoder_output2 = tf.squeeze(all_decoder_output1, [2])
        y_hat = self.dense(all_decoder_output2)
        accuracy = self.calculate_accuracy(target_sequence[:, 1:], y_hat)
        return self.calculate_accuracy.result()

    def train_one_step_step_by_step(self, input_sequence, target_sequence, batch_size):
        """
        :param: input_sequence: input sequence, shape is (batch_size, input_length)
        :param: target_sequence: target sequence, shape is (batch_size, output_length), output length is max_length
        :return: batch_loss: average loss in one batch
        """
        loss = 0
        # print(self.word2index)
        # target_sequence = np.insert(target_sequence, 0, self.word2index[self.start], axis=1)
        # target_sequence = np.hstack((target_sequence, np.array([[self.word2index[self.end]]] * target_sequence.shape[0])))
        """
        print("////////////////////")
        print(target_sequence)
        print(target_sequence[:, 2])
        print(target_sequence[:, 2].shape)
        print(target_sequence.shape)
        print(target_sequence)
        """
        with tf.GradientTape() as tape:
            encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
            encoder_output, encoder_hidden = self.encoder(input_sequence, encoder_hidden)

            previous_decoder_hidden = encoder_hidden
            target_word = np.expand_dims(target_sequence[:, 0], axis=-1)
            for target_word_index in range(1, target_sequence.shape[1]):
                decoder_output, previous_decoder_hidden = self.decoder(target_word, previous_decoder_hidden)
                target_word = np.expand_dims(target_sequence[:, target_word_index], axis=-1)
                context_vector, attention_weights = self.attention(decoder_output, encoder_output)
                attention_vector = tf.concat([context_vector, decoder_output], axis=-1)
                # shape of x and y_hat is (batch_size, 1, output_units)
                # so need to reshape attention_vector to (batch_size, output_units) for being dense layer's input
                dense_input = tf.reshape(attention_vector, (-1, attention_vector.shape[2]))
                # get y_hat, whose shape is (batch_size, vocab_size) to calculate loss
                y_hat = self.dense(dense_input)
                loss += Seq2Seq.LossFunction_step_by_step(target_word, y_hat, self.word2index)
                # print(loss)
                # print(y_hat)
            # average batch loss in this batch training
            batch_loss = (loss / int(target_sequence.shape[1]))
            # print(batch_loss)
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables\
                        + self.attention.trainable_variables + self.dense.trainable_variables
            # print(variables)
            gradients = tape.gradient(loss, variables)
            # print(gradients)
            # a error: no gradients provide for any variable
            # resolution: need put all forward layer into GradientTape(), then the gradients will
            # be gotten.
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

    def train_one_step_without_attention_step_by_step(self, input_sequence, target_sequence, batch_size):
        """
        :param: input_sequence: input sequence, shape is (batch_size, input_length)
        :param: target_sequence: target sequence, shape is (batch_size, output_length)
        :return: batch_loss: average loss in one batch
        """
        loss = 0
        # target_sequence = np.insert(target_sequence, 0, self.word2index[self.start], axis=1)
        # target_sequence = np.hstack(
        #    (target_sequence, np.array([[self.word2index[self.end]]] * target_sequence.shape[0])))
        # print("////////////////////")
        # print(target_sequence[:, 2])
        # print(target_sequence[:, 2].shape)
        # print(target_sequence.shape)
        # print(target_sequence)

        with tf.GradientTape() as tape:
            encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
            encoder_output, encoder_hidden = self.encoder(input_sequence, encoder_hidden)

            previous_decoder_hidden = encoder_hidden
            target_word = np.expand_dims(target_sequence[:, 0], axis=-1)
            for target_word_index in range(1, target_sequence.shape[1]):
                decoder_output, previous_decoder_hidden = self.decoder(target_word, previous_decoder_hidden)
                target_word = np.expand_dims(target_sequence[:, target_word_index], axis=-1)
                # shape of output and y_hat is (batch_size, 1, vocab_size)
                # so need to reshape output to (batch_size, vocab_size) for being dense layer's input
                dense_input = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
                y_hat = self.dense(dense_input)
                loss += Seq2Seq.LossFunction_step_by_step(target_word, y_hat, self.word2index)
        # average batch loss in this batch training
        batch_loss = loss / int(target_sequence.shape[1])
        # print(batch_loss)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables + self.dense.trainable_variables
        # print(variables)
        gradients = tape.gradient(loss, variables)
        # print(gradients)
        # a error: no gradients provide for any variable
        # resolution: need put all forward layer into GradientTape(), then the gradients will
        # be gotten.
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def validation_one_step_without_attention_step_by_step(self, input_sequence, target_sequence, batch_size):
        """
        :param: input_sequence: input sequence, shape is (batch_size, input_length)
        :param: target_sequence: target sequence, shape is (batch_size, output_length)
        :return: batch_loss: average loss in one batch
        """
        encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
        encoder_output, encoder_hidden = self.encoder(input_sequence, encoder_hidden)
        previous_decoder_hidden = encoder_hidden
        target_word = np.expand_dims(target_sequence[:, 0], axis=-1)
        for target_word_index in range(1, target_sequence.shape[1]):
            decoder_output, previous_decoder_hidden = self.decoder(target_word, previous_decoder_hidden)
            target_word = np.expand_dims(target_sequence[:, target_word_index], axis=-1)
            # shape of output and y_hat is (batch_size, 1, vocab_size)
            # so need to reshape output to (batch_size, vocab_size) for being dense layer's input
            dense_input = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
            y_hat = self.dense(dense_input)
        self.calculate_accuracy(target_word, y_hat)
        return self.calculate_accuracy.result()

    def predict_one_sequence(self, input_sequence, word2index, output_max_len):
        """
        predict one sequence using greedy search algorithm
        :param input_sequence: input sequences, shape is (1, input_length)
        :param output_max_len: output word max length
        :param word2index: word2index vocab
        :return: output sequences, it's list, length is (output_length,)
        """
        encoder_initial_hidden = self.encoder.initialize_hidden_state(batch_size=1)
        encoder_output, encoder_state = self.encoder(input_sequence, encoder_initial_hidden)
        decoder_input_x = tf.expand_dims([word2index['<START>']], 0)
        # print("------")
        # print(decoder_input_x.shape)
        # print("----------")
        previous_decoder_hidden = encoder_state
        output_word_indexes = []
        attention_weights_array = np.zeros((output_max_len, input_sequence.shape[1]))
        for output_index in range(output_max_len):
            # decoder_input_x is a word index, whose shape is (1, 1), (batch_size=1, one_word_index=1)
            decoder_output, previous_decoder_hidden = self.decoder(decoder_input_x, previous_decoder_hidden)
            context_vector, attention_weights = self.attention(decoder_output, encoder_output)
            attention_weights = tf.reshape(attention_weights, (-1,))
            # print(">>>>>>>>>>>++++++++++++")
            # print(attention_weights)
            # print(attention_weights.shape)
            # print(attention_weights_array)
            # print("++++++++++<<<<<<<<<<<<<")
            attention_weights_array[output_index] = attention_weights.numpy()
            attention_vector = tf.concat([context_vector, decoder_output], axis=-1)
            # now shape of output is (1, units)
            dense_input = tf.reshape(attention_vector, (-1, attention_vector.shape[2]))
            # shape of y_hat is (1, vocab_size)
            y_hat = self.dense(dense_input)
            # output_word_index is a predict word index,
            # witch can be convert to word by index2word_model
            # and please notice that it's just a index
            # print(y_hat)
            # actually method used as follows is greedy algorithm
            output_word_index = tf.math.argmax(y_hat, axis=1)
            output_word_index = tf.keras.backend.eval(output_word_index)
            output_word_indexes.append(output_word_index[0])
            # print(output_word_indexes)
            # print("=============")
            # print(word2index['<END>'])
            # print(output_word_index)
            # print("=============")
            if word2index['<END>'] == output_word_index:
                return output_word_indexes, attention_weights_array

            decoder_input_x = tf.expand_dims(output_word_index, 0)
        return output_word_indexes, attention_weights_array

    def predict_one_sequence_beam_search(self, input_sequence, word2index, output_max_len, k):
        """
        predict sentence using beam search algorithm.
        beam search的大概思想就是和贪婪算法不同，贪婪算法在每个时间步中在当前预测的vocab个
        词中选取概率最大的词，其他的词全部扔掉。但是beam search不同的是，每次选取k个概率最大的词，然后再将预测的这几个词分别输入到下一个时间步中，
        再预测top k个词，依次类推。然后，这样会得到k^max_len个预测句子，那么这么多句子我们该如何取舍呢？我们先设定一个初始值1.0，然后每次预测下
        一个词的时候，用1.0乘当前预测词的概率，这样最后会得到一个完整句子的选择概率，再根据这个概率值排序，选取概率最大的句子。
        :param input_sequence: input sequences, shape is (1, input_length)
        :param output_max_len: output word max length
        :param word2index: word2index vocab
        :param k: 每个时间步中要选取的概率最大的词的个数
        :return: output sequences, it's a list, length is (output_length,)
        """
        encoder_initial_hidden = self.encoder.initialize_hidden_state(batch_size=1)
        encoder_output, encoder_state = self.encoder(input_sequence, encoder_initial_hidden)
        decoder_input_x = tf.expand_dims([word2index['<START>']], 0)
        previous_decoder_hidden = encoder_state
        attention_weights_array = np.zeros((output_max_len, input_sequence.shape[1]))
        attention_weights_array_vocab = {}
        k_beam = [(0, [])]

        for output_index in range(output_max_len):
            all_k_beams = []
            # use bigger prob word first
            for prob, predict_sent in k_beam:
                if len(predict_sent) is not 0:
                    decoder_input_x = tf.expand_dims([predict_sent[-1]], 0)
                # decoder_input_x is a word index, whose shape is (1, 1), (batch_size=1, one_word_index=1)
                decoder_output, previous_decoder_hidden = self.decoder(decoder_input_x, previous_decoder_hidden)
                context_vector, attention_weights = self.attention(decoder_output, encoder_output)
                attention_weights = tf.reshape(attention_weights, (-1,))
                attention_weights_array[output_index] = attention_weights.numpy()
                attention_weights_array_vocab[str(decoder_input_x.numpy()[0][0])] = attention_weights_array
                attention_vector = tf.concat([context_vector, decoder_output], axis=-1)
                # now shape of dense_input is (1, units)
                dense_input = tf.reshape(attention_vector, (-1, attention_vector.shape[2]))
                # shape of y_hat is (1, vocab_size)
                y_hat = self.dense(dense_input)
                # add softmax layer
                y_hat = tf.nn.softmax(y_hat)
                # convert y_hat to array to invork argsort()
                y_hat = tf.keras.backend.eval(y_hat)
                # shape of output_beam is (beam_size,), and k_beam[-1]'s prob is biggest, but it's all right
                output_beam = y_hat[0].argsort()[-k:]
                # convert output_words_index_beam to array, shape is (beam_size,)
                for k_index in range(k):
                    all_k_beams += [(
                        prob + np.log(y_hat[0][output_beam[k_index]]),
                        predict_sent + [output_beam[k_index]]
                    )]
            # sort all k_beam to get k's biggest prob predict sequences
            k_beam = sorted(all_k_beams)[-k:]
            # if the most prob prediction meet '<END>', then return
            if (int(k_beam[-1][-1][-1]) - int(word2index['<END>'])) is 0:
                # d = k_beam
                # print("d: {}".format(d))
                # a = k_beam[-1][-1][:-1]
                # print("a: {}".format(a))
                if k_beam[-1][-1][-1] is not k_beam[-1][-1][0]:
                    return k_beam[-1][-1][:-1], attention_weights_array_vocab[str(k_beam[-1][-1][-2])], k_beam
                else:
                    return k_beam[-2][-1], attention_weights_array_vocab[str(decoder_input_x.numpy()[0][0])], k_beam
            else:
                # else let's remove other prediction ended with '<END>', because continue to predict them are useless.
                sub_k_beam = k_beam[:-1]
                while sub_k_beam:
                    sub_prob, sub_predict_sent = sub_k_beam.pop()
                    index = len(sub_k_beam)
                    if (int(sub_predict_sent[-1]) - int(word2index['<END>'])) is 0:
                        k_beam.pop(index)

        # output_word must be one biggest prob predict sequences selected from k_beam
        output_word_indexes = sorted(k_beam)[-1][-1]
        # print(output_word_indexes)
        return output_word_indexes, attention_weights_array_vocab[str(output_word_indexes[-2])], k_beam

    def load_trained_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_path))

    def save_training_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    @staticmethod
    def plot_attention_weights(attention_weights, input_sentence, predict_sentence, font, attention_path):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention_weights, cmap='viridis')

        font_dict = {'fontsize': 14, 'fontproperties': font}

        ax.set_xticklabels(['']+input_sentence, fontdict=font_dict, rotation=90)
        ax.set_yticklabels(['']+predict_sentence, fontdict=font_dict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()
        fig.savefig(attention_path)

"""
if __name__ == "__main__":
    batch = 2
    voc = 6
    em_dim = 5
    u = 4
    inp = 3

    we = np.array([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6]
    ])

    w2i = {
        '<START>': 3,
        "a": 0,
        "b": 1,
        "c": 2,
        '<END>': 4,
        '<PAD>': 5
    }

    sample_two_batch = np.array([
        [3, 1, 2, 4, 5, 5],
        [3, 2, 4, 5, 5, 5]
])
    sample_one_batch = np.array([
        [0, 0, 0]
    ])
    sample2_two_batch = np.array([
        [0, 1, 2],
        [1, 1, 1],
    ])
    sample2_one_batch = np.array([
        [0, 0, 0]
    ])

    print(sample_one_batch.shape)
    print(sample2_one_batch.shape)
    seq2seq = Seq2Seq(word_embedding=we,
                      word2index=w2i,
                      units=u,
                      checkpoint_path='./temp',
                      with_attention=True,
                      cell_type='GRU',
                      print_batch=True)
    ba_lo = seq2seq.train_one_step(sample_two_batch, sample2_two_batch, batch_size=2)
    print(ba_lo)

    ou_seq, wgs = seq2seq.evaluate(sample_one_batch, output_max_len=3, word2index=w2i)
    print("1")
    print(ou_seq)
    print("2")
    print(wgs)

    from matplotlib import font_manager
    sample2_one_batch_words = ['a', 'a', 'a']
    ou_seq_words = ['<START>', '<START>', 'START']
    font_g = font_manager.FontProperties(fname='/home/heping/pycharm/aistudio/project1_auto_master_qa/resource/font/font_sample.ttf')
    Seq2Seq.plot_attention_weights(wgs, sample2_one_batch_words, ou_seq_words, font_g, './sample2.png')

    #plt.plot([i for i in range(10)])
    #plt.show()
"""
