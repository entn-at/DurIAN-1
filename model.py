import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import modules
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TTS(nn.Module):
    def __init__(self,
                 num_phn=hp.num_phn,
                 encoder_dim=hp.encoder_dim,
                 encoder_n_layer=hp.encoder_n_layer,
                 kernel_size=hp.kernel_size,
                 stride=hp.stride,
                 padding=hp.padding,
                 decoder_dim=hp.decoder_dim,
                 decoder_n_layer=hp.decoder_n_layer,
                 output_dim=hp.num_mels,
                 dropout=hp.dropout):
        super(TTS, self).__init__()

        self.embedding = nn.Embedding(num_phn, encoder_dim)
        self.pos_emb = nn.Embedding.from_pretrained(
            modules.get_sinusoid_encoding_table(
                hp.num_position, hp.position_dim, padding_idx=0),
            freeze=True)
        self.conv_banks = nn.ModuleList([
            modules.BatchNormConv1d(encoder_dim,
                                    encoder_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    activation=nn.ReLU()) for _ in range(3)])
        self.encoder = nn.LSTM(encoder_dim,
                               encoder_dim // 2,
                               encoder_n_layer,
                               batch_first=True,
                               bidirectional=True)
        self.length_regulator = modules.LengthRegulator()

        self.prenet = modules.Prenet(hp.num_mels,
                                     hp.decoder_dim * 2,
                                     hp.decoder_dim)
        self.decoder = nn.LSTM(decoder_dim+encoder_dim+hp.position_dim,
                               decoder_dim,
                               decoder_n_layer,
                               batch_first=True,
                               bidirectional=False)
        self.mel_linear = modules.Linear(decoder_dim, output_dim)

        self.postnet = modules.CBHG(hp.num_mels, K=8,
                                    projections=[256, hp.num_mels])
        self.last_linear = nn.Linear(hp.num_mels*2, hp.num_mels)

    def get_lstm_cell(self):
        cell0 = nn.LSTMCell(self.decoder.input_size, self.decoder.hidden_size)
        cell0.weight_hh.data = self.decoder.weight_hh_l0.data
        cell0.weight_ih.data = self.decoder.weight_ih_l0.data
        cell0.bias_hh.data = self.decoder.bias_hh_l0.data
        cell0.bias_ih.data = self.decoder.bias_ih_l0.data

        cell1 = nn.LSTMCell(self.decoder.hidden_size, self.decoder.hidden_size)
        cell1.weight_hh.data = self.decoder.weight_hh_l1.data
        cell1.weight_ih.data = self.decoder.weight_ih_l1.data
        cell1.bias_hh.data = self.decoder.bias_hh_l1.data
        cell1.bias_ih.data = self.decoder.bias_ih_l1.data
        return cell0, cell1

    def get_cell_init(self):
        h0 = torch.zeros(1, self.decoder.hidden_size).float().to(device)
        c0 = torch.zeros(1, self.decoder.hidden_size).float().to(device)
        h1 = torch.zeros(1, self.decoder.hidden_size).float().to(device)
        c1 = torch.zeros(1, self.decoder.hidden_size).float().to(device)
        prev_mel_input = torch.zeros(1, hp.num_mels).float().to(device)
        return h0, c0, h1, c1, prev_mel_input

    def forward(self, src_seq, src_pos, prev_mel=None, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        encoder_input = self.embedding(src_seq).contiguous().transpose(1, 2)
        for conv in self.conv_banks:
            encoder_input = conv(encoder_input)
        encoder_input = encoder_input.contiguous().transpose(1, 2)

        input_lengths = torch.max(src_pos, -1)[0].cpu().numpy()
        encoder_input = nn.utils.rnn.pack_padded_sequence(encoder_input,
                                                          input_lengths,
                                                          batch_first=True)
        self.encoder.flatten_parameters()
        encoder_output, _ = self.encoder(encoder_input)
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(encoder_output,
                                                             batch_first=True)

        if self.training:
            memory, duration_predictor_output = self.length_regulator(encoder_output,
                                                                      target=length_target,
                                                                      alpha=alpha,
                                                                      mel_max_length=mel_max_length)
            pos_encoding = self.pos_emb(mel_pos)
            memory = torch.cat([memory, pos_encoding], -1)

            prev_mel = self.prenet(prev_mel)
            decoder_input = torch.cat([prev_mel, memory], -1)

            input_lengths = torch.max(mel_pos, -1)[0].cpu().numpy()
            index_arr = np.argsort(-input_lengths)
            input_lengths = input_lengths[index_arr]
            sorted_decoder_input = list()
            for ind in index_arr:
                sorted_decoder_input.append(decoder_input[ind])
            decoder_input = torch.stack(sorted_decoder_input)

            decoder_input = nn.utils.rnn.pack_padded_sequence(decoder_input,
                                                              input_lengths,
                                                              batch_first=True)
            self.decoder.flatten_parameters()
            decoder_output, _ = self.decoder(decoder_input)
            decoder_output, _ = nn.utils.rnn.pad_packed_sequence(decoder_output,
                                                                 batch_first=True)
            origin_decoder_output = [0 for _ in range(decoder_output.size(0))]
            for i, ind in enumerate(index_arr):
                origin_decoder_output[ind] = decoder_output[i]
            decoder_output = torch.stack(origin_decoder_output)
            mel_output = self.mel_linear(decoder_output)

            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output, duration_predictor_output
        else:
            memory, _ = self.length_regulator(encoder_output,
                                              target=length_target,
                                              alpha=alpha)

            mel_pred_len = memory.size(1)
            mel_list = list()
            h0, c0, h1, c1, prev_mel_input = self.get_cell_init()
            cell0, cell1 = self.get_lstm_cell()

            for i in range(mel_pred_len):
                memory_input = torch.cat(
                    [memory[:, i, :], self.pos_emb.weight[i+1].unsqueeze(0)], -1)
                prenet_output = self.prenet(prev_mel_input)
                decoder_input = torch.cat([prenet_output, memory_input], -1)

                cell0_h, cell0_c = cell0(decoder_input, (h0, c0))
                cell1_h, cell1_c = cell1(cell0_h, (h1, c1))

                mel_output = self.mel_linear(cell1_h)
                mel_list.append(mel_output)

                h0 = cell0_h
                c0 = cell0_c
                h1 = cell1_h
                c1 = cell1_c
                prev_mel_input = mel_output

            mel_output = torch.stack(mel_list).contiguous().transpose(0, 1)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output


if __name__ == "__main__":
    # Test
    model = TTS()
    print("number of model parameter:",
          sum(param.numel() for param in model.parameters()))
