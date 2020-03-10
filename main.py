from myUtils.myDataProcessing import MyDataProcessing
from model_tensorflow.layers import Seq2Seq
import numpy as np
import pandas as pd
import csv
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from multiprocessing import cpu_count
import logging
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.manifold import TSNE
import time
import os
import math
import copy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# -------------------------------------------------------
# path parameters
train_source_path = './dataset/AutoMaster_TrainSet.csv'
test_source_path = './dataset/AutoMaster_TestSet.csv'
cleaned_train_source_path = './dataset/train_cleaned.csv'
cleaned_test_source_path = './dataset/test_cleaned.csv'
merged_train_test_path = './dataset/merged_train_test.csv'
padded_cleaned_train_source_path = './dataset/train_padded_cleaned.csv'
padded_cleaned_test_source_path = './dataset/test_padded_cleaned.csv'
no_padded_cleaned_test_source_path = './dataset/test_cleaned_no_pad.csv'
word_freq_path = './dataset/word_freq.csv'
word2index_path = './dataset/word2index.csv'
index2word_path = './dataset/index2word.csv'
last_word2vec_model_path = './dataset/word2vec.model'
word2vec_model_path = './dataset/word2vec.model'
update_word2vec_model_path = './dataset/word2vec_updated.model'
word_embedding_path = './dataset/word_embedding.csv'
t_SNE_path = './dataset/t-SNE.png'
train_sequences_path = './dataset/train_sequences.csv'
test_sequences_path = './dataset/test_sequences.csv'
attention_path = './dataset/attention_weight.png'
checkpoint_path = './dataset/check_point'
prediction_result_path = './dataset/result.csv'
# -------------------------------------------------------
# word2vec model parameters
freq_threshold = 5
embedding_dim = 300
word2vec_windows = 10
workers = cpu_count() * 2
train_epoches = 100
# -------------------------------------------------------
# seq2seq model parameters
batch_size = 32
with_attention = True
cell_type = 'GRU'
print_batch = True
dense_units = 512
epoch_counts = 100
train_validation_scale = 0.95
beam_size = 5
# -------------------------------------------------------
# other global variables
font_global = font_manager.FontProperties(fname='./resource/font/font_sample.ttf')
# -------------------------------------------------------
# global class
myDataProcess = MyDataProcessing(stop_word_file_path='./resource/stopwords/哈工大停用词表.txt',
                                 special_words=r'[\s+\-\|\!\/\[\]\{\}_$%^*(+\"\')]+|[+——()'
                                               r'【】“”~@#￥%……&*]+',
                                 all_lower_case=False)


# combine two column and remove NaN
def combine(sentence):
    if isinstance(sentence['Dialogue'], str) and isinstance(sentence['Question'], str):
        return ' '.join(sentence)
    elif isinstance(sentence['Dialogue'], str):
        return sentence['Dialogue']
    elif isinstance(sentence['Question'], str):
        return sentence['Question']


# remove NaN value in train dataset or directly use dropna method in dataframe
def remove_nan(sentence):
    if isinstance(sentence, str) and isinstance(sentence, str):
        return sentence


def clean_dataframe_process(dataframe):
    for column in dataframe.columns:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(myDataProcess.clean_sentence_process)
    return dataframe


def process_original_dataset_and_save2file():
    original_train_df = pd.read_csv(train_source_path)
    original_test_df = pd.read_csv(test_source_path)

    # remove specific columns
    original_train_df.drop(columns=['QID', 'Brand', 'Model'], axis=1, inplace=True)
    original_test_df.drop(columns=['QID', 'Brand', 'Model'], axis=1, inplace=True)

    # clean None value in specific columns
    original_train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    original_test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

    # clean dataframe using multiCPU
    train_df = MyDataProcessing.parallelize_process(clean_dataframe_process, original_train_df)
    test_df = MyDataProcessing.parallelize_process(clean_dataframe_process, original_test_df)

    train_df.to_csv(cleaned_train_source_path, index=None, header=True)
    test_df.to_csv(cleaned_test_source_path, index=None, header=True)

    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # concatenate train and test need use train_df[['merged']], because
    # it will return matrix, but train_df['merged'] just return an object
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    merged_df.to_csv(merged_train_test_path, index=None, header=True)

    # Get all words for corpus
    words = []
    for sentence in merged_df['merged']:
        words += sentence.split(' ')

    # Get all words's frequency
    # and remove lower freq words (do not need)
    cleaned_words = []
    with open(word_freq_path, 'w') as writer:
        for word, freq in collections.Counter(words).items():
            writer.writelines([word, ',', str(freq), '\n'])
            if freq >= freq_threshold and '' != word:
                cleaned_words.append(word)
        writer.close()

    word2index_writer = csv.writer(open(word2index_path, 'w'))
    index2word_writer = csv.writer(open(index2word_path, 'w'))
    for index, word in enumerate(cleaned_words):
        word2index_writer.writerow([word, index])
        index2word_writer.writerow([index, word])


def reprocess_dataset_and_file_saved(word2index):
    train_df = pd.read_csv(cleaned_train_source_path, dtype=str)
    test_df = pd.read_csv(cleaned_test_source_path, dtype=str)

    train_df['Input'] = train_df[['Question', 'Dialogue']].apply(lambda x: combine(x), axis=1)
    test_df['Input'] = test_df[['Question', 'Dialogue']].apply(lambda x: combine(x), axis=1)
    test_df_no_pad = test_df.copy()

    train_df.rename(columns={'Report': 'Target'}, inplace=True)

    train_df = train_df.dropna()
    train_df = train_df.reset_index(drop=True)

    train_max_len = MyDataProcessing.get_max_len(train_df['Input'])
    test_max_len = MyDataProcessing.get_max_len(test_df['Input'])
    input_max_len = max(train_max_len, test_max_len)
    if input_max_len > 250:
        input_max_len = 250

    target_max_len = MyDataProcessing.get_max_len(train_df['Target'])
    if target_max_len > 40:
        target_max_len = 40

    train_df['Input'] = train_df['Input'].apply(
        lambda x: MyDataProcessing.pad_sentences(x, max_len=input_max_len, vocab=word2index))
    train_df['Target'] = train_df['Target'].apply(
        lambda x: MyDataProcessing.pad_sentences(x, max_len=target_max_len, vocab=word2index))
    test_df['Input'] = test_df['Input'].apply(
        lambda x: MyDataProcessing.pad_sentences(x, max_len=input_max_len, vocab=word2index))
    test_df_no_pad['Input'] = test_df_no_pad['Input'].apply(
        lambda x: MyDataProcessing.add_start_and_end_to_sentence(x, word2index=word2index))

    train_df.to_csv(padded_cleaned_train_source_path, columns=['Input', 'Target'], index=None, header=True)
    test_df.to_csv(padded_cleaned_test_source_path, columns=['Input'], index=None, header=True)
    test_df_no_pad.to_csv(no_padded_cleaned_test_source_path, columns=['Input'], index=None, header=True)


def convert_train_data_to_sequences_and_save_to_file(train_data_path, saved_path, word2index):
    train_data = pd.read_csv(train_data_path, sep=',')
    input_sentences, target_sentences = train_data['Input'], train_data['Target']

    sample_size = len(input_sentences)
    input_length = len(input_sentences[0].split(' '))
    target_length = len(target_sentences[0].split(' '))

    input_sequences = np.zeros((sample_size, input_length))
    target_sequences = np.zeros((sample_size, target_length))

    input_sequences_list = []
    target_sequences_list = []
    for i in range(0, sample_size):
        input_sequence = []
        target_sequence = []
        if isinstance(input_sentences[i], str) and isinstance(target_sentences[i], str):
            input_sequence = [int(word2index[input_word]) if input_word in word2index.keys() else '<UNK>' for input_word in input_sentences[i].split(' ')]
            target_sequence = [int(word2index[target_word]) if target_word in word2index.keys() else '<UNK>' for target_word in target_sentences[i].split(' ')]
        if len(input_sequence) and len(target_sequence) is not 0:
            input_sequences[i, :] = input_sequence
            target_sequences[i, :] = target_sequence
            input_sequences_list.append(input_sequence)
            target_sequences_list.append(target_sequence)
    df = pd.DataFrame({'input_sequences': input_sequences_list,
                       'target_sequences': target_sequences_list})
    df.to_csv(saved_path, index=False)
    return input_sequences, target_sequences


def convert_test_data_to_sequences_and_save_to_file(test_data_path, saved_path, word2index):
    test_data = pd.read_csv(test_data_path, sep=',')
    input_sentences = test_data['Input']
    sample_size = len(input_sentences)
    input_sequences_list = []
    input_sequences_array_list = []
    for i in range(0, sample_size):
        input_sequence = []
        if isinstance(input_sentences[i], str):
            input_sequence = [int(word2index[input_word]) if input_word in word2index.keys() else '<UNK>' for input_word in input_sentences[i].split(' ')]
            input_sequence_array = np.expand_dims(input_sequence, 0)
        if len(input_sequence) is not 0:
            input_sequences_list.append(input_sequence)
            input_sequences_array_list.append(input_sequence_array)
    df = pd.DataFrame({'input_sequences': input_sequences_list})
    df.to_csv(saved_path, index=False)
    return input_sequences_array_list


"""
def get_train_sequences_from_file(file_path):
    df = pd.read_csv(file_path)
    input_sequences_str = df['input_sequences']
    target_sequences_str = df['target_sequences']
    input_sequences_int = []
    target_sequences_int = []
    for line in input_sequences_str:
        sequence = []
        for index in line.strip().strip('[').rstrip(']').strip().split(','):
            sequence.append(int(index.strip()))
        if len(sequence) is not 0:
            input_sequences_int.append(sequence)
    for line in target_sequences_str:
        sequence = []
        for index in line.strip().strip('[').rstrip(']').strip().split(','):
            sequence.append(int(index.strip()))
        if len(sequence) is not 0:
            target_sequences_int.append(sequence)

    target_sequences_int = np.asarray(target_sequences_int)
    input_sequences_int = np.asarray(input_sequences_int)
    return input_sequences_int, target_sequences_int
"""


def train_word2vec_model_and_save_to(word2vec_model_file_path, use_train_data_path):
    model = word2vec.Word2Vec(LineSentence(use_train_data_path),
                              size=embedding_dim,
                              window=word2vec_windows,
                              min_count=freq_threshold,
                              workers=workers,
                              iter=train_epoches)
    model.save(word2vec_model_file_path)
    return model


def update_word2vec_model(word2vec_model, use_train_data_path):
    word2vec_model.build_vocab(LineSentence(use_train_data_path), update=True)
    word2vec_model.train(LineSentence(use_train_data_path),
                         epochs=train_epoches,
                         total_examples=word2vec_model.corpus_count)
    word2vec_model.save(update_word2vec_model_path)
    return word2vec_model


def draw_word2vec_by_tSNE(word_embedding, index2word, visual_size, fig_size, font, t_sne_path):
    tSNE = TSNE()
    embedding_tSNE = tSNE.fit_transform(word_embedding[:visual_size, :])
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    for i in range(visual_size):
        ax.scatter(*embedding_tSNE[i, :])
        ax.annotate(index2word[i],
                    (embedding_tSNE[i, 0], embedding_tSNE[i, 1]),
                    alpha=0.7,
                    fontproperties=font)
    fig.savefig(t_sne_path)


def get_word2vec_model_from_file(word2vec_model_file_path):
    return word2vec.Word2Vec.load(word2vec_model_file_path)


def get_word2index_from_word2vec_model(word2vec_model):
    return {word: index for index, word in enumerate(word2vec_model.wv.index2word)}


def get_index2word_from_word2vec_model(word2vec_model):
    return {index: word for index, word in enumerate(word2vec_model.wv.index2word)}


def save_word_embedding_to_file(word2index, word2vectors, word_embedding_file_path):
    word_embedding = []
    for word, index in word2index.items():
        word_embedding.append(word2vectors.wv.get_vector(word))
    word_embedding = np.asarray(word_embedding)
    np.savetxt(word_embedding_file_path, word_embedding, fmt='%s', delimiter=',')
    return word_embedding


# Return an float data array, whose shape is (vocab_size, vector_size)
def get_word_embedding_from_file(word_embedding_file_path):
    word_embedding = []
    with open(word_embedding_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        for row in csv_reader:
            word_embedding.append([float(s) for s in row[0].split(',')])
    word_embedding = np.asarray(word_embedding)
    print("word_embedding' shape is {}".format(word_embedding.shape))
    return word_embedding


"""
# Write before
def calculate_accuracy(input_sequences, target_sequences, seq2seq_model, word2index, word2embedding, predict_max_len):
    all_length = 0
    same_length = 0
    for index in range(input_sequences.shape[0]):
        input_sequence = input_sequences[index, :]
        input_sequence = np.expand_dims(input_sequence, 0)
        predict_sequence, _ = seq2seq_model.predict_one_sequence(input_sequence=input_sequence, word2index=word2index, output_max_len=predict_max_len)
        cleaned_target_sequence = MyDataProcessing.remove_pad_and_start(target_sequences[index, :], word2index)

        compare_length = min(len(predict_sequence), len(cleaned_target_sequence))
        all_length += compare_length
        for l in range(compare_length):
            if predict_sequence[l] is cleaned_target_sequence[l]:
                same_length += 1
    accuracy = same_length / float(all_length)
    return accuracy
"""


def train_seq2seq(seq2seq_model, input_sequences, target_sequences, validation_input_sequences, validation_target_sequences, epoch_numbers):
    for epoch in range(epoch_numbers):
        total_loss = 0
        train_accuracy = 0
        validation_accuracy_sum = 0
        start_time = time.time()

        # shuffle train dataset in every epoch
        shuffled_permutation = np.random.permutation(input_sequences.shape[0])
        input_sequences = input_sequences[shuffled_permutation, :]
        target_sequences = target_sequences[shuffled_permutation, :]

        if int(input_sequences.shape[0] % batch_size) is 0:
            batch_numbers = int(input_sequences.shape[0] / batch_size)
        else:
            batch_numbers = int(input_sequences.shape[0] / batch_size) + 1
        for batch_number in range(batch_numbers):
            if input_sequences.shape[0] < (batch_number + 1) * batch_size:
                input_sequence = input_sequences[batch_number * batch_size:]
                target_sequence = target_sequences[batch_number * batch_size:]
                final_batch_size = input_sequences.shape[0] - batch_number * batch_size
            else:
                input_sequence = input_sequences[batch_number * batch_size: (batch_number + 1) * batch_size]
                target_sequence = target_sequences[batch_number * batch_size: (batch_number + 1) * batch_size]
                final_batch_size = batch_size

            # print(">>>>>>>>>>>>>")
            # print("batch of {} input_sequence is: {}".format(batch_number, input_sequence))
            # print("batch of {} target_sequence is: {}".format(batch_number, target_sequence))
            # print("<<<<<<<<<<<<<<")
            if with_attention:
                batch_loss, train_accuracy = seq2seq_model.train_one_step(input_sequence, target_sequence,
                                                                          final_batch_size)
            else:
                batch_loss, train_accuracy = seq2seq_model.train_one_step_without_attention(input_sequence, target_sequence,
                                                                                            final_batch_size)
            total_loss += batch_loss

            if 0 == (batch_number % 100) and print_batch is True:
                print("Epoch: {} Batch: {} Loss: {:4f} train accuracy: {:4f}".format(
                    epoch,
                    batch_number,
                    batch_loss.numpy(),
                    train_accuracy.numpy()
                ))

        if int(validation_input_sequences.shape[0] % batch_size) is 0:
            batch_numbers_v = int(validation_input_sequences.shape[0] / batch_size)
        else:
            batch_numbers_v = int(validation_input_sequences.shape[0] / batch_size) + 1
        for batch_number in range(batch_numbers_v):
            if len(validation_input_sequences) < (batch_number + 1) * batch_size:
                validation_input_sequence = validation_input_sequences[batch_number * batch_size:]
                validation_target_sequence = validation_target_sequences[batch_number * batch_size:]
                validation_final_batch_size = validation_input_sequences.shape[0] - batch_number * batch_size
            else:
                validation_input_sequence = validation_input_sequences[batch_number * batch_size: (batch_number + 1) * batch_size]
                validation_target_sequence = validation_target_sequences[batch_number * batch_size: (batch_number + 1) * batch_size]
                validation_final_batch_size = batch_size

            if with_attention:
                validation_accuracy = seq2seq_model.validation_one_step(validation_input_sequence,
                                                                        validation_target_sequence,
                                                                        validation_final_batch_size)
            else:
                validation_accuracy = seq2seq_model.validation_one_step(validation_input_sequence,
                                                                        validation_target_sequence,
                                                                        validation_final_batch_size)
            validation_accuracy_sum += validation_accuracy

        if (epoch + 1) % 2 is 0:
            seq2seq_model.save_training_checkpoint()

        print("Epoch: {} Average loss: {:4f} train average accuracy: {:4f} validate average accuracy: {:4f}".format(
            epoch,
            total_loss / batch_numbers,
            train_accuracy,
            validation_accuracy_sum / batch_numbers_v
        ))
        print("Time taken for {}th epoch is {} sec".format(epoch, time.time() - start_time))


def predict_one_sentence(input_sentence, seq2seq_model, word2index, index2word, predict_max_len, predict_algorithm):
    input_sequence = MyDataProcessing.convert_sentence2sequence(sentence=input_sentence, word2index=word2index)
    input_sequence = np.expand_dims(input_sequence, 0)
    if predict_algorithm is 'greedy':
        predict_sequence, attention_weights = seq2seq_model.predict_one_sequence(input_sequence=input_sequence,
                                                                                 word2index=word2index,
                                                                                 output_max_len=predict_max_len)
    elif predict_algorithm is 'beam':
        predict_sequence, attention_weights, k_beam = seq2seq_model.predict_one_sequence_beam_search(input_sequence=input_sequence,
                                                                                                     word2index=word2index,
                                                                                                     output_max_len=predict_max_len,
                                                                                                     k=beam_size)
        """
        # print for test
        s1 = k_beam[0][-1]
        s2 = k_beam[1][-1]
        s3 = k_beam[2][-1]
        print([index2word[word] for word in s1])
        print([index2word[word] for word in s2])
        print([index2word[word] for word in s3])
        # test end
        """

    else:
        print("predict algorithm {} not exists!!!".format(predict_algorithm))
        return -1
    predict_sentence = MyDataProcessing.convert_sequence2sentence(sequence=predict_sequence, index2word=index2word)
    return predict_sentence, attention_weights


def train_and_get_word2vector_model(need_train):
    if need_train is True or os.path.exists(word_embedding_path) is False:
        prebuild_word2vector_model = train_word2vec_model_and_save_to(last_word2vec_model_path, merged_train_test_path)
        # reprocess dataset to make sentences same length and then retrain word2vec model
        prebuild_word2index = MyDataProcessing.get_word2index_from_file(word2index_path)
        reprocess_dataset_and_file_saved(prebuild_word2index)
        word2vector_model = update_word2vec_model(prebuild_word2vector_model, padded_cleaned_train_source_path)
        word2vector_model = update_word2vec_model(word2vector_model, padded_cleaned_test_source_path)
    else:
        word2vector_model = get_word2vec_model_from_file(update_word2vec_model_path)
    print(word2vector_model.wv.most_similar(['坏'], topn=10))
    return word2vector_model


def get_word_embedding_and_word2index_and_index2word(word2vector_model, need_draw_t_sne, t_sne_path):
    word2index = get_word2index_from_word2vec_model(word2vector_model)
    index2word = get_index2word_from_word2vec_model(word2vector_model)
    word_embedding = save_word_embedding_to_file(word2index, word2vector_model, word_embedding_path)
    # word_embedding = get_word_embedding_from_file(word_embedding_path)
    if need_draw_t_sne is True:
        draw_word2vec_by_tSNE(word_embedding, index2word, visual_size=len(word_embedding), fig_size=200, font=font_global, t_sne_path=t_sne_path)
    return word_embedding, word2index, index2word


def get_model(process_original_dataset=False, need_train_word2vec=False, need_draw_t_sne=False, test_mode=False):
    if process_original_dataset is True:
        process_original_dataset_and_save2file()
    word2vec_model = train_and_get_word2vector_model(need_train_word2vec)
    word_embedding, word2index, index2word = get_word_embedding_and_word2index_and_index2word(word2vec_model, need_draw_t_sne, t_SNE_path)

    '''
    print(word2index['，'])
    print(word2index['：'])
    print(word2index['。'])
    print(word2index['？'])
    print(word2index['！'])
    '''
    # print(word_embedding.shape[0])
    # print(len(word2index.keys()))
    # print(len(index2word.keys()))
    train_input_sequences, train_target_sequences = convert_train_data_to_sequences_and_save_to_file(padded_cleaned_train_source_path,
                                                                                                     train_sequences_path,
                                                                                                     word2index)
    target_max_len = len(train_target_sequences[0])

    # retrieve part of input sequences for debug
    if test_mode is True:
        train_input_sequences = train_input_sequences[0:50, :]
        train_target_sequences = train_target_sequences[0:50, :]

    seq2seq_model = Seq2Seq(word_embedding=word_embedding,
                            word2index=word2index,
                            units=dense_units,
                            checkpoint_path=checkpoint_path,
                            with_attention=with_attention,
                            cell_type=cell_type,
                            print_batch=print_batch)

    return seq2seq_model, word2index, index2word, word_embedding, train_input_sequences, train_target_sequences, target_max_len


def start_train(seq2seq_model, train_input_sequences, train_target_sequences, validation_input_sequences, validation_target_sequences, based_trained_seq2seq=False):
    if based_trained_seq2seq is True:
        # load trained checkpoint to train again!
        seq2seq_model.load_trained_checkpoint()

    train_seq2seq(seq2seq_model=seq2seq_model,
                  input_sequences=train_input_sequences,
                  target_sequences=train_target_sequences,
                  validation_input_sequences=validation_input_sequences,
                  validation_target_sequences=validation_target_sequences,
                  epoch_numbers=epoch_counts)


def get_model_and_predict(input_sentence, seq2seq_model, word2index, index2word, predict_max_len, predict_algorithm):
    # load trained latest checkpoint file
    seq2seq_model.load_trained_checkpoint()

    predict_sentence, attention_weights = predict_one_sentence(input_sentence=input_sentence,
                                                               seq2seq_model=seq2seq_model,
                                                               word2index=word2index,
                                                               index2word=index2word,
                                                               predict_max_len=predict_max_len,
                                                               predict_algorithm=predict_algorithm)
    return predict_sentence, attention_weights


def evaluate_seq2seq_model(seq2seq_model, word2index, index2word, predict_max_len, predict_algorithm, test_data_path, save_test_sequences_path, result_path, test_mode):
    test_input_sequences_list = convert_test_data_to_sequences_and_save_to_file(test_data_path, save_test_sequences_path, word2index)
    predict_sentences_list = []
    if test_mode is True:
        test_input_sequences_list = test_input_sequences_list[:3]
    for input_sequence_index in range(len(test_input_sequences_list)):
        if predict_algorithm is 'greedy':
            predict_sequence, attention_weights = seq2seq_model.predict_one_sequence(input_sequence=test_input_sequences_list[input_sequence_index],
                                                                                     word2index=word2index,
                                                                                     output_max_len=predict_max_len)
        elif predict_algorithm is 'beam':
            predict_sequence, attention_weights, k_beam = seq2seq_model.predict_one_sequence_beam_search(input_sequence=test_input_sequences_list[input_sequence_index],
                                                                                                         word2index=word2index,
                                                                                                         output_max_len=predict_max_len,
                                                                                                         k=beam_size)
        else:
            print("predict algorithm {} not exists!!!".format(predict_algorithm))
            return -1
        predict_sentence = MyDataProcessing.convert_sequence2sentence(sequence=predict_sequence, index2word=index2word)
        predict_sentences_list.append(predict_sentence)
    with open(result_path, 'w', newline='') as writer:
        writer.writelines(['QID', ',', 'Prediction', '\n'])
        for num, sentence in enumerate(predict_sentences_list):
            writer.writelines(['Q' + str(num + 1), ',', sentence, '\n'])


def start_answer(input_sentence,
                 load_existed_seq2seq=True,
                 based_trained_seq2seq=False,
                 process_original_dataset=False,
                 need_train_word2vector=False,
                 need_draw_t_sne=False,
                 predict_algorithm='beam',
                 predict_result=False,
                 test_mode=False):
    seq2seq, word2index, index2word, word2embedding, input_sequences, target_sequences, predict_max_len = get_model(
        process_original_dataset=process_original_dataset,
        need_train_word2vec=need_train_word2vector,
        need_draw_t_sne=need_draw_t_sne,
        test_mode=test_mode)
    if load_existed_seq2seq is False or os.path.exists(checkpoint_path) is False:
        sample_size = input_sequences.shape[0]
        train_size = int(sample_size * train_validation_scale)
        train_input_sequences = input_sequences[:train_size]
        train_target_sequences = target_sequences[:train_size]
        validation_input_sequences = input_sequences[train_size:]
        validation_target_sequences = target_sequences[train_size:]
        start_train(seq2seq, train_input_sequences, train_target_sequences, validation_input_sequences, validation_target_sequences, based_trained_seq2seq=based_trained_seq2seq)
        predict_sentence, attention_weights = predict_one_sentence(input_sentence,
                                                                   seq2seq_model=seq2seq,
                                                                   word2index=word2index,
                                                                   index2word=index2word,
                                                                   predict_max_len=predict_max_len,
                                                                   predict_algorithm=predict_algorithm)
    else:
        predict_sentence, attention_weights = get_model_and_predict(input_sentence,
                                                                    seq2seq_model=seq2seq,
                                                                    word2index=word2index,
                                                                    index2word=index2word,
                                                                    predict_max_len=predict_max_len,
                                                                    predict_algorithm=predict_algorithm)
    print(predict_sentence)
    if predict_result is True:
        evaluate_seq2seq_model(seq2seq, word2index, index2word, predict_max_len, predict_algorithm, no_padded_cleaned_test_source_path, test_sequences_path, prediction_result_path, test_mode)
    input_words = MyDataProcessing.cut_sentence2words(input_sentence)
    predict_words = MyDataProcessing.cut_sentence2words(predict_sentence)
    Seq2Seq.plot_attention_weights(attention_weights, input_words, predict_words, font_global, attention_path)


def main():
    # input_sentence = "车子中控是好的，车钥匙也是好的，也是有频率的，就是车钥匙控制不了车子，只能用机械钥匙开门，现在车门锁也开失灵了，怎么回事呀？"
    # input_sentence = "14年9代雅阁开三年了，打火时没力、时间长（大概5、6秒）吱吱响，请问是电瓶坏了，还是电瓶漏电了?"
    # input_sentence = "2012款奔驰c180怎么样，维修保养，动力，值得拥有吗"
    # input_sentence = "我今年刚买的宝马5系，结果漏油"
    # input_sentence = "挂了3挡车子推动了2下会打变速箱齿轮吗?"
    # input_sentence = "科鲁兹仪表出现一本书加i是什么意思？"
    input_sentence = "索八2.0，78000公里。刚启动开的时候油门重，开着还算正常，行驶时间长了，起步的时候给一点点油车就一怂，到4 5十迈，或者6 7迈左右给油没反应,感觉跟脱档一样转速也没反应 速度也上不去 猛踩一脚 油 强制降档 速度才上去 开着特别累。求大神指点 喷油嘴 节气门 汽油滤芯  都弄了 三元检查不堵。4s店换完火花塞好>了几天 后来又这样了。"
    start_answer(input_sentence,
                 load_existed_seq2seq=False,
                 based_trained_seq2seq=False,
                 process_original_dataset=False,
                 need_train_word2vector=False,
                 need_draw_t_sne=False,
                 predict_algorithm='beam',
                 predict_result=True,
                 test_mode=False)


if __name__ == "__main__":
    main()
