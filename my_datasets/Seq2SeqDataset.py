import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Seq2SeqDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, data_file, tokenizer, preload_file='', part="train"):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.part = part
        self.data = None
        self.preload_file = preload_file
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.preload_file):
            self.data = torch.load(f=self.preload_file)[self.part]
            print(f"加载本地数据集成功,part={self.part}")
            return

        data = []
        with open(file=self.data_file, mode="r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                if line:
                    input_sentence, output_sentence = line.strip().split("\t")
                    input_sentence = self.tokenizer.split_input(input_sentence)
                    output_sentence = self.tokenizer.split_output(output_sentence)
                    data.append([input_sentence, output_sentence])

            train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

        if self.part == "train":
            self.data = train_data
        elif self.part == "test":
            self.data = test_data
        else:
            self.data = None

        torch.save(obj={'train': train_data, 'test': test_data}, f=self.preload_file)
        print('加载数据完成预处理')

    def __getitem__(self, idx):
        input_sentence, output_sentence = self.data[idx]
        return (
            input_sentence,
            len(input_sentence),
            output_sentence,
            len(output_sentence),
        )

    def __len__(self):
        return len(self.data)


def collate_fn(batch, tokenizer):
    batch = sorted(batch, key=lambda ele: ele[1], reverse=True)
    input_sentences, input_sentence_lens, output_sentences, output_sentence_lens = zip(*batch)

    input_sentence_len = input_sentence_lens[0]
    input_idxes = []
    for input_sentence in input_sentences:
        input_idxes.append(tokenizer.encode_input(input_sentence, input_sentence_len))

    output_sentence_len = max(output_sentence_lens)
    output_idxes = []
    for output_sentence in output_sentences:
        output_idxes.append(tokenizer.encode_output(output_sentence, output_sentence_len))

    input_idxes = torch.LongTensor(input_idxes).t()
    output_idxes = torch.LongTensor(output_idxes).t()
    input_sentence_lens = torch.LongTensor(input_sentence_lens)
    output_sentence_lens = torch.LongTensor(output_sentence_lens)

    return input_idxes, input_sentence_lens, output_idxes, output_sentence_lens
