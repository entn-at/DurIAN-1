import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
import audio
from utils import process_text, pad_1D, pad_2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DNNDataset(Dataset):
    def __init__(self):
        self.metadata = process_data()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        mel_gt_name = self.metadata[idx][2]
        mel_gt_target = np.load(
            "/apdcephfs/share_1213607/zhxliu/DURIAN/durian.daji"+mel_gt_name[1:])[:3000, :]
        character = np.load(
            "/apdcephfs/share_1213607/zhxliu/DURIAN/durian.daji"+self.metadata[idx][4][1:])
        frame_ind = np.load(
            "/apdcephfs/share_1213607/zhxliu/DURIAN/durian.daji/"+self.metadata[idx][5][1:])
        D = [0 for i in range(character.shape[0])]
        for ind in frame_ind:
            D[ind] += 1
        D = np.array(D)

        sample = {"text": character,
                  "mel_target": mel_gt_target,
                  "D": D}

        return sample


def reprocess(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    Ds = [batch[ind]["D"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = np.array(src_pos)

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = np.array(mel_pos)

    texts = pad_1D(texts)
    Ds = pad_1D(Ds)
    mel_targets = pad_2D(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "D": Ds,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output


def process_data(mode="train"):
    metadata_filename = os.path.join(
        "/apdcephfs/share_1213607/zhxliu/DURIAN/durian.daji/linear_multi", "full.txt")

    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        hours = sum((int(x[3]) for x in metadata)) * \
            hparams.frame_shift_ms / (3600 * 1000)
        print('Loaded metadata for %d examples (%.2f hours)' %
              (len(metadata), hours))

    return metadata
