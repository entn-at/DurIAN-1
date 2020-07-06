# DurIAN
Implementation of [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700)

## Blog
[论文笔记：腾讯AI lab多模态语音合成模型DurIAN](https://zhuanlan.zhihu.com/p/105796626)

## Structure
...

## Sample
[here]()

## Usage
training:
1. `pip install -r requirements.txt`
2. Download and extract [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/)
3. Put LJSpeech dataset in `data`
4. `unzip alignments.zip`
5. `python3 preprocess.py`
6. `CUDA_VISIBLE_DEVICES=0 python3 train.py`

testing:
1. Put [Nvidia pretrained waveglow model](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing) in the `waveglow/pretrained_model`
2. `CUDA_VISIBLE_DEVICES=0 python3 test.py --step [step-of-checkpoint]`
