import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
import audio
import utils
import dataset
import text
import model as M
import waveglow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.DurIAN()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path))['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel


def get_data():
    test1 = "I am very happy to see you again!"
    test2 = "Durian model is a very good speech synthesis!"
    test3 = "The traditional holidays in our house when I was a child were spent timing elaborate meals around football games."
    data_list = list()
    data_list.append(text.text_to_sequence(test1, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test2, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test3, hp.text_cleaners))
    return data_list


if __name__ == "__main__":
    # Test
    WaveGlow = utils.get_WaveGlow()
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    print("use griffin lim")
    model = get_DNN(args.step)
    data_list = get_data()
    for i, phn in enumerate(data_list):
        mel, mel_cuda = synthesis(model, phn, args.alpha)
        if not os.path.exists("results"):
            os.mkdir("results")
        audio.tools.inv_mel_spec(
            mel, "results/"+str(args.step)+"_"+str(i)+".wav")
        waveglow.inference.inference(
            mel_cuda, WaveGlow, "results/"+str(args.step)+"_"+str(i)+"waveglow.wav")
