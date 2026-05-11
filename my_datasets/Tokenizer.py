import jieba
import torch
import os
from opencc import OpenCC
from tqdm import tqdm


class Tokenizer(object):
    """
    定义分词器
    """

    def __init__(self, data_file, saved_dict=''):
        self.data_file = data_file
        self.saved_dict = saved_dict
        # 输入的词表
        self.input_word2idx = {}
        self.input_idx2word = {}
        self.input_dict_len = None
        self.input_embed_dim = 256
        self.input_hidden_size = 256
        # 输出的词表
        self.output_word2idx = {}
        self.output_idx2word = {}
        self.output_dict_len = None
        self.output_embed_dim = 256
        self.output_hidden_size = 256
        # 推理时，输出的最大长度
        self.output_max_len = 100
        # 英文的标点符号
        self.punctuations = [".", ",", "?", "!"]
        # 繁体转简体
        self.opencc = OpenCC("t2s")

    def build_dict(self):
        if os.path.exists(self.saved_dict):
            self.load()
            print("加载本地字典成功")
            return

        input_words = {"<UNK>", "<PAD>"}
        output_words = {"<UNK>", "<PAD>", "<SOS>", "<EOS>"}

        with open(file=self.data_file, mode="r", encoding="utf8") as f:
            for line in tqdm(f.readlines()):
                if line:
                    input_sentence, output_sentence = line.strip().split("\t")
                    input_sentence_words = self.split_input(input_sentence)
                    output_sentence_words = self.split_output(output_sentence)
                    input_words = input_words.union(set(input_sentence_words))
                    output_words = output_words.union(set(output_sentence_words))

        self.input_word2idx = {word: idx for idx, word in enumerate(input_words)}
        self.input_idx2word = {idx: word for word, idx in self.input_word2idx.items()}
        self.input_dict_len = len(self.input_word2idx)

        self.output_word2idx = {word: idx for idx, word in enumerate(output_words)}
        self.output_idx2word = {idx: word for word, idx in self.output_word2idx.items()}
        self.output_dict_len = len(self.output_word2idx)

        self.save()
        print("保存字典成功")

    def split_input(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace("'", " ")
        sentence = "".join(
            [" " + char + " " if char in self.punctuations else char for char in sentence]
        )
        words = [word for word in sentence.split(" ") if word]
        return words

    def split_output(self, sentence):
        sentence = self.opencc.convert(sentence)
        words = jieba.lcut(sentence)
        return words

    def encode_input(self, input_sentence, input_sentence_len):
        input_idx = [
            self.input_word2idx.get(word, self.input_word2idx.get("<UNK>"))
            for word in input_sentence
        ]
        input_idx = (input_idx + [self.input_word2idx.get("<PAD>")] * input_sentence_len)[:input_sentence_len]
        return input_idx

    def encode_output(self, output_sentence, output_sentence_len):
        output_sentence = output_sentence + ["<EOS>"]
        output_sentence_len += 1
        output_idx = [
            self.output_word2idx.get(word, self.output_word2idx.get("<UNK>"))
            for word in output_sentence
        ]
        output_idx = (output_idx + [self.output_word2idx.get("<PAD>")] * output_sentence_len)[:output_sentence_len]
        return output_idx

    def decode_output(self, pred):
        raw_result = []
        for idx in pred:
            raw_result.append(self.output_idx2word.get(idx))

        final_result = []
        for idx in pred:
            if idx == self.output_word2idx.get("<EOS>"):
                break
            elif idx == self.output_word2idx.get("<PAD>"):
                continue
            final_result.append(self.output_idx2word.get(idx))

        return raw_result, final_result

    def save(self):
        state_dict = {
            "input_word2idx": self.input_word2idx,
            "input_idx2word": self.input_idx2word,
            "input_dict_len": self.input_dict_len,
            "output_word2idx": self.output_word2idx,
            "output_idx2word": self.output_idx2word,
            "output_dict_len": self.output_dict_len,
        }
        torch.save(obj=state_dict, f=self.saved_dict)

    def load(self):
        if os.path.exists(self.saved_dict):
            state_dict = torch.load(f=self.saved_dict)
            self.input_word2idx = state_dict.get("input_word2idx")
            self.input_idx2word = state_dict.get("input_idx2word")
            self.input_dict_len = state_dict.get("input_dict_len")
            self.output_word2idx = state_dict.get("output_word2idx")
            self.output_idx2word = state_dict.get("output_idx2word")
            self.output_dict_len = state_dict.get("output_dict_len")


def get_tokenizer(data_file, saved_dict):
    tokenizer = Tokenizer(data_file=data_file, saved_dict=saved_dict)
    tokenizer.build_dict()
    return tokenizer
