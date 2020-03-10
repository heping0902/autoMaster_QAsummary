import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
import jieba
import re

jieba.load_userdict("/home/heping/pycharm/aistudio/project1_auto_master_qa/resource/jieba_dictionary/user_dict.txt")
#jieba.load_userdict("./resource/jieba_dictionary/user_dict.txt")


class MyDataProcessing:

    def __init__(self,
                 stop_word_file_path='',
                 special_words=r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?]+',
                 all_lower_case=False,
                 all_upper_case=False):
        super().__init__()
        self.stop_word_file_path = stop_word_file_path
        self.special_words = special_words
        self.all_lower_case = all_lower_case
        self.all_upper_case = all_upper_case

    '''
    # written before, it is very time-consuming!
    def csv2txt(self):
        df = pd.read_csv(self.source_path)
        if self.remove_columns:
            df = df.drop(self.remove_columns, axis=1)
        df.to_csv(self.destination_path, index=False, header=False, sep=' ')

    def txt2vocab(self):
        words = []
        freq = {}
        number = 0
        with open(self.destination_path, 'r') as reader:
            for line in reader.readlines():
                seg_list = jieba.cut(line)
                if self.stop_word_file_path:
                    with open(self.stop_word_file_path, 'r') as stop_word_reader:
                        parsed_stop_words = [stop_word.strip() for stop_word in stop_word_reader.readlines()]
                        stop_word_reader.close()
                    seg_list = [w for w in seg_list if w not in parsed_stop_words]
                if self.special_words:
                    seg_list = [w for w in seg_list if w not in self.special_words]
                for word in seg_list:
                    if self.all_lower_case:
                        word = word.lower()
                    if self.all_upper_case:
                        word = word.upper()
                    if word not in words:
                        if 0 != len(word) and ' ' != word:
                            freq[word] = 1
                            words.append(word)
                    else:
                        freq[word] += 1
            reader.close()
        with open(self.vocab_file_path, 'w') as vocab_writer, open(self.freq_file_path, 'w') as freq_writer:
            for word in words:
                if freq[word] > self.keep_freq:
                    vocab_writer.writelines([str(word), ' ', str(number), '\n'])
                    freq_writer.writelines([str(word), ' ', str(freq[word]), '\n'])
                    number += 1
            vocab_writer.close()
            freq_writer.close()
    '''

    def clean_sentence_process(self, sentence):
        if self.special_words and isinstance(sentence, str):
            sentence = re.sub(
                self.special_words,
                '',
                sentence)
        else:
            sentence = ''
        if self.all_lower_case:
            sentence.lower()
        if self.all_upper_case:
            sentence.upper()
        if self.stop_word_file_path:
            with open(self.stop_word_file_path, 'r', encoding='utf-8') as reader:
                parsed_stop_words = [stop_word.strip() for stop_word in reader.readlines()]
                reader.close()
            seg_list = jieba.cut(sentence)
            sentence = [word for word in seg_list if word not in parsed_stop_words]
        sentence = ' '.join(sentence)
        return sentence

    @staticmethod
    def get_max_len(dataframe_per_column):
        """
        get most befittingly max_len for padding.
        usually the value is calculated by mean(len) + 2 * std(len)
        :param dataframe_per_column: dataframe specific column
        :return: dataframe padded
        """
        lengths = dataframe_per_column.apply(lambda x: x.count(' '))
        return int(np.mean(lengths) + 2 * np.std(lengths))

    @staticmethod
    def pad_sentences(sentence, max_len, vocab):
        """
        1. replace <unknown> with words not in vocab;
        2. insert <start> and <end> to start and end of sentence.
        3. pad <pad> to all sentences for being same length (max_len calculated by upper method);
        :param sentence: data
        :param max_len: the length all sentences will be padded
        :param vocab: word vocab
        :return: padded sentence
        """
        words = [word for word in sentence.strip().split(' ')]
        words = words[:(max_len - 2)]
        sentence = [word if word in vocab.keys() else '<UNK>' for word in words]
        sentence = ['<START>'] + sentence + ['<END>']
        sentence = sentence + ['<PAD>'] * (max_len - len(sentence))  # multiply negative value not impact
        return ' '.join(sentence)

    @staticmethod
    def pad_sequence(sequence, max_len, word2index):
        """
        sequence must be list data structure, sample: [1, 2, 3]
        pad sequence: input [1, '<END>'], output ['<START>', 1, '<END>', '<PAD>' ...]
        :param sequence: sequence ready to pad
        :param word2index: word2index dictionary
        :param max_len: padded sequence's length
        :return: padded sequence
        """
        sequence = sequence + [word2index['<PAD>']] * (max_len - len(sequence))
        return sequence

    @staticmethod
    def remove_pad_and_start(sequence, word2index):
        """
        remove '<PAD>' and '<START>' in sequence
        :param sequence: predict sequence
        :param word2index word2index dictionary
        :return: '<PAD>' removed sequence
        """
        return [s for s in sequence if s != word2index['<PAD>'] and s != word2index['<START>']]

    @staticmethod
    def add_start_and_end_to_sentence(sentence, word2index):
        """
        add '<START>' and '<END>' to the sentence
        :param sentence: input sequence
        :param word2index: word2index
        :return: sequence added '<START>' and '<END>'
        """
        words = [word for word in sentence.strip().split(' ')]
        sentence = [word if word in word2index.keys() else '<UNK>' for word in words]
        sentence = ['<START>'] + sentence + ['<END>']
        return ' '.join(sentence)

    @staticmethod
    def add_start(sequence, word2index):
        """
        add '<START>' in first sequence
        :param sequence: input sequence
        :param word2index: word2index dictionary
        :return: '<START>' added sequence
        """
        return [word2index['<START>']] + sequence

    @staticmethod
    def parallelize_process(func, dataframe):
        cpu_number = cpu_count()
        dataframe_chunks = np.array_split(dataframe, cpu_number)
        pools = Pool(processes=cpu_number)
        processed_dataframe = pd.concat(pools.map(func, dataframe_chunks))
        pools.close()
        pools.join()
        return processed_dataframe

    @staticmethod
    def convert_sentence2sequence(sentence, word2index):
        words = jieba.cut(sentence)
        # print("-----/\/\/\---------")
        # print([word for word in words])
        # print("-----\/\/\/---------")
        words = [word for word in words]
        words = ['<START>'] + words + ['<END>']
        sequence = [int(word2index[word]) if word in word2index.keys() else word2index['<UNK>'] for word in words]
        return sequence

    @staticmethod
    def cut_sentence2words(sentence):
        words = jieba.cut(sentence)
        return [word for word in words]

    @staticmethod
    def convert_sequence2sentence(sequence, index2word):
        words = [index2word[index] for index in sequence]
        return ''.join(words)

    @staticmethod
    def get_word2index_from_file(word2index_file_path):
        word2index_data = {}
        with open(word2index_file_path, 'r') as vocab_reader:
            for vocab_line in vocab_reader:
                word, index = vocab_line.strip().split(',')
                if '' != word and ' ' != word:
                    word2index_data[word] = int(index)
            vocab_reader.close()
        return word2index_data

    @staticmethod
    def get_index2word_from_file(index2word_file_path):
        index2word_data = {}
        with open(index2word_file_path, 'r') as vocab_reader:
            for vocab_line in vocab_reader:
                index, word = vocab_line.strip().split(',')
                if '' != word and ' ' != word:
                    index2word_data[int(index)] = word
            vocab_reader.close()
        return index2word_data


"""
if __name__ == '__main__':
    voc = pd.DataFrame()
    voc['a'] = 1
    voc['b'] = 2
    voc['c'] = 3
    voc['d'] = 4
    print(voc)
    v = pd.DataFrame({'x': ['a b c d e f g', 'p i y a b c 4 5 38 d s d'],
                      'y': ['a e g b c s', '1 3 5 s x t s b c']})
    print(v)
    sen1 = MyDataProcessing.pad_sentences(sentence='a 1 c', max_len=5, vocab=voc)
    sen2 = MyDataProcessing.pad_sentences(sentence='2 b', max_len=5, vocab=voc)
    sen3 = MyDataProcessing.pad_sentences(sentence='2 b b c 1 3 5 a c', max_len=5, vocab=voc)
    print(">>>>>>>>>>>>")
    print(sen1)
    print(sen2)
    print(sen3)
    print("<<<<<<<<<<<<")
    print(v['x'].apply(lambda x: MyDataProcessing.pad_sentences(x, max_len=5, vocab=voc)))
    print(v['y'].apply(lambda x: MyDataProcessing.pad_sentences(x, max_len=2, vocab=voc)))

    sen4 = '我爱祖国'
    w2i = {
        "a": 0,
        "b": 1,
        "c": 2,
        '<START>': 3,
        '<END>': 4,
        '<PAD>': 5,
        '<UNK>': 6
    }
    seq1 = [0, 1, 2, 3, 4, 5]
    i2w = {
        0: 'a',
        1: 'b',
        2: 'c',
        3: '<START>',
        4: '<END>',
        5: '<PAD>',
        6: '<UNK>'
    }
    print(MyDataProcessing.convert_sentence2sequence(sen4, word2index=w2i))
    print(MyDataProcessing.convert_sequence2sentence(seq1, index2word=i2w))

    orig_s_p = MyDataProcessing.pad_sequence([1], 4, w2i)
    orig_s_p_s = MyDataProcessing.add_start(orig_s_p, w2i)
    orig_s = MyDataProcessing.remove_pad_and_start(orig_s_p_s, w2i)
    print(orig_s_p)
    print(orig_s_p_s)
    print(orig_s)

    # s_s = ['<START>', '2010', '款', '宝马X1', '2011', '年', '出厂', '20', '排量', '通用', '<UNK>', '变速箱', '原地', '换挡', '位', 'PRND', '车辆', '闯动', '行驶', '升降', '档', '正常', '4', '轮离', '换挡', '无', '冲击', '感', '更换', '变速箱', '油', '12L', '无', '改变', '试', '一辆', '2014', '年', '进口', 'X1', '原地', '换挡', '位', '冲击', '感', '情况', '问题', '4', '缸', '自然', '吸气', '发动机', 'N46', '先', '挂', '空档', '再', '挂', '档', '有没有', '闯动', '变速箱', '油液', '位', '是否', '调整', '正常', '液位', 'N', 'D', '<UNK>', '没有', '<UNK>', 'PR', '最', '主要', '行驶', '中到', '红绿灯', '路口', '红灯', '停车', '<UNK>', '冲击', '感', '绿灯', '后', 'ND', '冲击', '感', '很小', '第一', '变速箱', '油位', '调整', '标准', '液位', '清除', '变速箱', '适应', '值', '第三', '升级', '变速箱', '程序', '遇到', '液力', '变矩器', '问题', '升级', '变速箱', '程序', '刷', '模块', '问题', '停车', '后', '档位', 'P', '挡', '松开', '刹车踏板', '时', '感觉', '车辆', '会', '动', '一下', '清除', '变速箱', '适应', '<UNK>', '简单', '排查', '可能', '程序', '问题', '可能', '液力', '变矩器', '轴头', '磨损', '泄压', '需要', '专用', '电脑', '清除', '变速箱', '适应', '值', '升级', '变速箱', '程序', '换', '变速箱', '油有', '焦糊', '味', '没', '变速箱', '油', '底壳', '带', '滤芯', '换', '没', '没有', '味', '滤芯', '换', '变矩器', '磨损', '车况', '上架', '4', '轮离', '换挡', '位', '没有', '冲击', '感', '先', '简单', '排查', '换', '油', '需要', '需要', '重新学习', '没', '换油', '之前', '是因为', '冲击', '才', '换', '油', '换油', '之前', '换挡', '冲击', '行驶', '冲击', '原地', '换挡', '位', '冲击', '换油', '行驶', '都', '没', '问题', '公里', '估计', '程序', '问题', '阀体', '里', '问题', '阀体', '电脑', '一体', '93', '万公里', '昨天', '去试', '4', '万多公里', 'X1', '是不是', '通病', '一点', '正常', '刹车', '踩', '重点', '用力', '踩', '刹车', '冲击', '感', '基本', '没有', '用力', '踩住', '刹车', '原地', '换挡', '位', '基本', '感觉', '不到', '冲击', '感', '行驶', '没有', '冲击', '应该', '没有', '<END>']
    # s_s_c = MyDataProcessing.pad_sentences(s_s, max_len=4, vocab=w2i)

    s_s_single = "宝马 奔驰 都 不错 hhh 其实 奥迪 也行 哈哈哈 开玩笑"
    s_s_add_s_e = MyDataProcessing.add_start_and_end_to_sentence(s_s_single, w2i)
    print(s_s_add_s_e)
"""
