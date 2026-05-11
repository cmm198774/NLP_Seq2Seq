import torch
from torch import nn
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_datasets.Tokenizer import Tokenizer


class Encoder(nn.Module):
    """
    定义一个 编码器
    """

    def __init__(self, tokenizer):
        super(Encoder, self).__init__()
        self.tokenizer = tokenizer
        self.embed = nn.Embedding(
            num_embeddings=self.tokenizer.input_dict_len,
            embedding_dim=self.tokenizer.input_embed_dim,
            padding_idx=self.tokenizer.input_word2idx.get("<PAD>"),
        )
        self.gru = nn.GRU(
            input_size=self.tokenizer.input_embed_dim,
            hidden_size=self.tokenizer.input_hidden_size,
            batch_first=False,
        )

    def forward(self, x, x_len):
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(input=x, lengths=x_len, batch_first=False)
        out, hn = self.gru(x)
        out, out_len = nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=False,
            padding_value=self.tokenizer.input_word2idx.get("<PAD>"),
        )
        return out, hn


class Decoder(nn.Module):
    def __init__(self, tokenizer):
        super(Decoder, self).__init__()
        self.tokenizer = tokenizer
        self.embed = nn.Embedding(
            num_embeddings=self.tokenizer.output_dict_len,
            embedding_dim=self.tokenizer.output_embed_dim,
            padding_idx=self.tokenizer.output_word2idx.get("<PAD>"),
        )
        self.gru = nn.GRU(
            input_size=self.tokenizer.output_embed_dim,
            hidden_size=self.tokenizer.output_hidden_size,
            batch_first=False,
        )
        self.fc = nn.Linear(
            in_features=self.tokenizer.output_hidden_size,
            out_features=self.tokenizer.output_dict_len,
        )

    def forward_step(self, decoder_input, decoder_hidden):
        decoder_input = self.embed(decoder_input)
        out, decoder_hidden = self.gru(decoder_input, decoder_hidden)
        out = out.squeeze(dim=0)
        out = self.fc(out)
        return out, decoder_hidden

    def forward(self, encoder_hidden, y, y_len):
        output_max_len = max(y_len.tolist()) + 1
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(
            [[self.tokenizer.output_word2idx.get("<SOS>")] * batch_size]
        ).to(encoder_hidden.device)
        decoder_outputs = torch.zeros(
            output_max_len, batch_size, self.tokenizer.output_dict_len
        ).to(encoder_hidden.device)

        decoder_hidden = encoder_hidden
        for t in range(output_max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t, :, :] = decoder_output_t
            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                decoder_input = y[t, :].unsqueeze(0)
            else:
                decoder_input = decoder_output_t.argmax(dim=-1).unsqueeze(0)
        return decoder_outputs

    def batch_infer(self, encoder_hidden):
        output_max_len = self.tokenizer.output_max_len
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(
            [[self.tokenizer.output_word2idx.get("<SOS>")] * batch_size]
        ).to(encoder_hidden.device)
        results = []
        decoder_hidden = encoder_hidden
        with torch.no_grad():
            for t in range(output_max_len):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_input = decoder_output_t.argmax(dim=-1).unsqueeze(0)
                results.append(decoder_input)
            results = torch.cat(tensors=results, dim=0)
        return results


class Seq2Seq(nn.Module):

    def __init__(self, tokenizer):
        super(Seq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Encoder(self.tokenizer)
        self.decoder = Decoder(self.tokenizer)

    def forward(self, x, x_len, y, y_len):
        out, hn = self.encoder(x, x_len)
        results = self.decoder(hn, y, y_len)
        return results

    def batch_infer(self, x, x_len):
        out, hn = self.encoder(x, x_len)
        preds = self.decoder.batch_infer(hn)
        results = []
        for s in preds.t():
            results.append(self.tokenizer.decode_output(s.tolist()))
        return results
