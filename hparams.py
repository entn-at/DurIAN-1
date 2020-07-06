# Mel
num_mels = 80
text_cleaners = ['english_cleaners']


# TTS-DNN
max_sep_len = 5000
num_phn = 384
kernel_size = 5
stride = 1
padding = 2
encoder_dim = 512
encoder_n_layer = 1

prenet_dim = 256
position_dim = encoder_dim
n_frames_per_step = 3
decoder_dim = prenet_dim + encoder_dim
decoder_n_layer = 2

duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3

dropout = 0.1

fft_conv1d_kernel = 3
fft_conv1d_padding = 1
ref_enc_filters = [32, 32, 64, 64, 128, 128]


# Train
checkpoint_path = "./model_new"
logger_path = "./logger"
mel_ground_truth = "./mels"
alignment_path = "./alignments"

batch_size = 32
epochs = 2000
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20

batch_expand_size = 32
