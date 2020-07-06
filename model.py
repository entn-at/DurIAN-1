import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import utils
import modules
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DurIAN(nn.Module):
    """ GRU """

    def __init__(self,
                 num_phn=hp.num_phn,
                 encoder_dim=hp.encoder_dim,
                 encoder_n_layer=hp.encoder_n_layer,
                 kernel_size=hp.kernel_size,
                 stride=hp.stride,
                 padding=hp.padding,
                 decoder_dim=hp.decoder_dim,
                 decoder_n_layer=hp.decoder_n_layer,
                 dropout=hp.dropout):
        super(DurIAN, self).__init__()

        self.n_step = hp.n_frames_per_step
        self.embedding = nn.Embedding(num_phn, encoder_dim)

        self.conv_banks = nn.ModuleList([
            modules.BatchNormConv1d(encoder_dim,
                                    encoder_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    activation=nn.ReLU(),
                                    w_init_gain="relu") for _ in range(3)])
        self.encoder = nn.GRU(encoder_dim,
                              encoder_dim // 2,
                              encoder_n_layer,
                              batch_first=True,
                              bidirectional=True)

        self.length_regulator = modules.LengthRegulator()

        self.prenet = modules.Prenet(hp.num_mels,
                                     hp.prenet_dim,
                                     hp.prenet_dim)

        self.decoder = nn.GRU(decoder_dim,
                              decoder_dim,
                              decoder_n_layer,
                              batch_first=True,
                              bidirectional=False)

        self.mel_linear = modules.Linear(
            decoder_dim, hp.num_mels * self.n_step)
        self.postnet = modules.CBHG(hp.num_mels, K=8,
                                    projections=[256, hp.num_mels])
        self.last_linear = modules.Linear(hp.num_mels * 2, hp.num_mels)

    def get_gru_cell(self):
        cell0 = nn.GRUCell(self.decoder.input_size, self.decoder.hidden_size)
        cell0.weight_hh.data = self.decoder.weight_hh_l0.data
        cell0.weight_ih.data = self.decoder.weight_ih_l0.data
        cell0.bias_hh.data = self.decoder.bias_hh_l0.data
        cell0.bias_ih.data = self.decoder.bias_ih_l0.data

        cell1 = nn.GRUCell(self.decoder.hidden_size, self.decoder.hidden_size)
        cell1.weight_hh.data = self.decoder.weight_hh_l1.data
        cell1.weight_ih.data = self.decoder.weight_ih_l1.data
        cell1.bias_hh.data = self.decoder.bias_hh_l1.data
        cell1.bias_ih.data = self.decoder.bias_ih_l1.data

        return cell0, cell1

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def get_gru_cell_init(self, batch_size=1):
        h0 = torch.zeros(batch_size, hp.decoder_dim).float().to(device)
        h1 = torch.zeros(batch_size, hp.decoder_dim).float().to(device)
        prev_mel_input = torch.zeros(batch_size, hp.num_mels)\
            .float().to(device)
        return h0, h1, prev_mel_input

    def forward(self, src_seq, src_pos,
                prev_mel=None, mel_pos=None, mel_max_length=None,
                length_target=None, epoch=None, alpha=1.0):

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

            count_circle = memory.size(1) // self.n_step
            index_list = [(i+1)*self.n_step-1 for i in range(count_circle-1)]
            prev_mel_input = F.pad(
                prev_mel[:, index_list, :], (0, 0, 1, 0, 0, 0))
            prev_mel_input = self.prenet(prev_mel_input)

            memory_input = memory[:, :count_circle*self.n_step, :].contiguous()
            memory_input = F.adaptive_avg_pool1d(
                memory_input.contiguous().transpose(1, 2), count_circle)\
                .contiguous().transpose(1, 2)

            decoder_input = torch.cat([prev_mel_input, memory_input], -1)
            decoder_output, _ = self.decoder(decoder_input)
            mel_output = self.mel_linear(decoder_output)
            mel_output = mel_output.view(mel_output.size(0), -1, hp.num_mels)

            mel_pad = (0, 0, 0, mel_max_length-mel_output.size(1), 0, 0)
            mel_output = F.pad(mel_output, mel_pad)

            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual
            mel_postnet_output = self.mask_tensor(mel_postnet_output,
                                                  mel_pos,
                                                  mel_max_length)

            return mel_output, mel_postnet_output, duration_predictor_output
        else:
            self.decoder_cell0, self.decoder_cell1 = self.get_gru_cell()

            memory, _ = self.length_regulator(encoder_output,
                                              target=length_target,
                                              alpha=alpha)

            mel_list = list()
            h0, h1, prev_mel_input = self.get_gru_cell_init()
            count_circle = memory.size(1) // self.n_step

            memory_input = memory[:, :count_circle*self.n_step, :].contiguous()
            memory_input = F.adaptive_avg_pool1d(
                memory_input.contiguous().transpose(1, 2), count_circle)\
                .contiguous().transpose(1, 2)

            for i in range(count_circle):
                prenet_output = self.prenet(prev_mel_input)
                decoder_input = torch\
                    .cat([prenet_output, memory_input[:, i, :]], -1)

                h0 = self.decoder_cell0(decoder_input, h0)
                h1 = self.decoder_cell1(h0, h1)

                mel_output = self.mel_linear(h1)
                mel_output = mel_output.view(
                    mel_output.size(0), self.n_step, -1)
                mel_list.append(mel_output)

                prev_mel_input = mel_output[:, -1, :]

            mel_output = torch.cat(mel_list, 1)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output


if __name__ == "__main__":
    # Test
    model = DurIAN()
    print("number of durian parameter:",
          sum(param.numel() for param in model.parameters()))
