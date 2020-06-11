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
import dataset
import model as M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.TTS()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path))['model'])
    model.eval()
    return model


def synthesis(model, phn, ref_mel=None, style_index=0, duration=None, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    duration = np.stack([duration])
    ref_mel = np.stack([ref_mel])

    with torch.no_grad():
        sequence = torch.from_numpy(text).cuda().long()
        src_pos = torch.from_numpy(src_pos).cuda().long()
        ref_mel = torch.from_numpy(ref_mel).cuda().float()
        if duration is not None:
            duration = torch.from_numpy(duration).cuda().long()

        _, mel = model.module.forward(sequence,
                                      src_pos,
                                      length_target=duration,
                                      alpha=alpha)

        return mel[0].cpu().transpose(0, 1), mel.transpose(1, 2)


def get_data(num=3):
    metadata = dataset.process_data(mode="test")
    test_metadata = random.sample(metadata, num)
    data_list = list()
    origin_mel_list = list()
    wav_list = list()
    durations = list()
    for i in range(num):
        character = np.load(
            "/apdcephfs/share_1213607/zhxliu/DURIAN/durian.daji"+test_metadata[i][4][1:])
        data_list.append(character)
        mel = np.load(
            "/apdcephfs/share_1213607/zhxliu/DURIAN/durian.daji"+test_metadata[i][2][1:])
        origin_mel_list.append(mel)
        wav_list.append(test_metadata[i][0])
        frame_ind = np.load(
            "/apdcephfs/share_1213607/zhxliu/DURIAN/durian.daji/"+test_metadata[i][5][1:])
        D = [0 for i in range(character.shape[0])]
        for ind in frame_ind:
            D[ind] += 1
        D = np.array(D)
        durations.append(D)

    return data_list, origin_mel_list, wav_list, durations


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == 0:
        print("use griffin lim")
        model = get_DNN(args.step)
        data_list, _, _, _ = get_data()
        for i, phn in enumerate(data_list):
            mel, _ = synthesis(model, phn, args.alpha)
            if not os.path.exists("results"):
                os.mkdir("results")
            wav_mel = audio.inv_mel_spectrogram(mel)
            audio.save_wav(wav_mel, "results/" + str(args.step) +
                           "_" + str(args.mode) + "_" + str(i) + "_mel.wav")
    elif args.mode == 1:
        print("use griffin lim + multiband wavernn")
        model = get_DNN(args.step)
        data_list, mel_list, wav_list, durations = get_data()
        for i, phn in enumerate(data_list):
            mel, _ = synthesis(
                model, phn, ref_mel=mel_list[i], duration=durations[i], alpha=args.alpha)
            if not os.path.exists("results"):
                os.mkdir("results")
            wav_mel = audio.inv_mel_spectrogram(mel)
            audio.save_wav(wav_mel, "results/" + str(args.step) +
                           "_" + str(args.mode) + "_" + str(i) + "_mel.wav")
            np.save("temp.npy", mel.numpy())
            os.system("./synthesis.sh temp.npy %s" % "results/" + str(args.step) +
                      "_" + str(args.mode) + "_" + str(i) + "_wav.wav")
        print("convert original mel spectrogram")
        for i, mel in enumerate(mel_list):
            np.save("temp.npy", mel)
            os.system("./synthesis.sh temp.npy %s" % "results/" + str(args.step) +
                      "_" + str(args.mode) + "_" + str(i) + "_wav_original.wav")
        for i, wav_name in enumerate(wav_list):
            shutil.copyfile(wav_name, "results/" + str(args.step) +
                            "_" + str(args.mode) + "_" + str(i) + "_original.wav")
    else:
        print("No Mode")
