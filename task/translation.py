import os
import sys
import random
import time
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_datasets.Tokenizer import get_tokenizer
from my_datasets.Seq2SeqDataset import Seq2SeqDataset, collate_fn
from model.Seq2Seq import Seq2Seq


class Translation(object):
    """
    把任务封装为一个类
    """

    def __init__(self, data_file, preload_file='', model_file='', saved_dict=''):
        self.data_file = data_file
        self.preload_file = preload_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_file = model_file
        self.saved_dict = saved_dict
        self.tokenizer = get_tokenizer(data_file=data_file, saved_dict=self.saved_dict)
        self.model = self._get_model()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.output_word2idx.get("<PAD>")
        )
        self.train_data = Seq2SeqDataset(
            data_file=self.data_file, tokenizer=self.tokenizer, preload_file=preload_file, part="train"
        )
        self.test_data = Seq2SeqDataset(
            data_file=self.data_file, tokenizer=self.tokenizer, preload_file=preload_file, part="test"
        )

    def _get_model(self):
        model = Seq2Seq(tokenizer=self.tokenizer)
        if os.path.exists(self.model_file):
            model.load_state_dict(state_dict=torch.load(self.model_file))
            print("加载本地模型成功")
        model = model.to(self.device)
        return model

    def get_acc(self, dataloader):
        self.model.eval()
        accs = []
        with torch.no_grad():
            for (x, x_len, y, y_len) in dataloader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                results = self.model.batch_infer(x, x_len)
                loss = self.get_loss(decoder_outputs=results, y=y)
                accs.append(loss)
        avg_acc = round(number=sum(accs) / len(accs), ndigits=6)
        return avg_acc

    def get_loss(self, decoder_outputs, y):
        decoder_outputs = decoder_outputs.to(device=self.device)
        y = y.contiguous().view(-1)
        decoder_outputs = decoder_outputs.contiguous().view(-1, decoder_outputs.size(-1))
        loss = self.loss_fn(decoder_outputs, y)
        return loss

    def get_real_output(self, y):
        y = y.t().tolist()
        results = []
        for s in y:
            results.append(
                [
                    self.tokenizer.output_idx2word.get(idx)
                    for idx in s
                    if idx not in [
                        self.tokenizer.output_word2idx.get("<EOS>"),
                        self.tokenizer.output_word2idx.get("<PAD>"),
                    ]
                ]
            )
        return results

    def get_real_input(self, x):
        x = x.t().tolist()
        results = []
        for s in x:
            results.append(
                [
                    self.tokenizer.input_idx2word.get(idx)
                    for idx in s
                    if idx not in [self.tokenizer.input_word2idx.get("<PAD>")]
                ]
            )
        return results

    def train(self, epochs=50, batch_size=32, cnt_interval=10, loss_thred=0.1):
        is_complete = False
        dataloader_train = DataLoader(
            dataset=self.train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
        )
        dataloader_eval = DataLoader(
            dataset=self.test_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
        )

        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            for batch_i, (x, x_len, y, y_len) in enumerate(dataloader_train):
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                results = self.model(x, x_len, y, y_len)
                loss = self.get_loss(decoder_outputs=results, y=y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (batch_i + 1) % cnt_interval == 0:
                    epoch_batches_left = len(dataloader_train) - (batch_i + 1)
                    time_left = datetime.timedelta(
                        seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1)
                    )
                    print(
                        f"epoch={epoch+1},剩余batch={epoch_batches_left},epoch剩余时间={time_left}秒,loss={loss}"
                    )

                if loss.item() < loss_thred:
                    is_complete = True
                    print(f"训练提前完成, 本批次损失为：{loss.item()}")
                    break

            with torch.no_grad():
                for (x, x_len, y, y_len) in dataloader_train:
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    x_true = self.get_real_input(x)
                    y_pred = self.model.batch_infer(x, x_len)
                    y_true = self.get_real_output(y)
                    samples = random.sample(population=range(x.size(1)), k=2)
                    print(f"\t第{epoch+1}轮已经训练完毕")
                    for idx in samples:
                        print("\t真实输入：", x_true[idx])
                        print("\t真实结果：", y_true[idx])
                        print("\t预测结果：", y_pred[idx][1])
                        print("\t----------------------------------------------------------")
                    break

            if is_complete:
                break

        return

    def infer(self, x="Am I wrong?"):
        """
        单样本推理
        """
        print("输入：", x)
        x = self.tokenizer.split_input(x)
        print("分词：", x)
        x = self.tokenizer.encode_input(x, len(x))
        print("编码：", x)
        x = torch.tensor(data=[x], dtype=torch.long).t().to(device=self.device)
        print("张量：", x)
        x_len = torch.tensor(data=[len(x)], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.batch_infer(x, x_len)
            print("原始输出：", y_pred[0][0])
            print("最终输出：", y_pred[0][1])
            return y_pred[0]
