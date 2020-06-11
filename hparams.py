# Mel
num_mels = 80
num_freq = 1025
sample_rate = 24000
frame_length_ms = 50
frame_shift_ms = 10
preemphasis = 0.97
fmin = 40
min_level_db = -100
ref_level_db = 20
max_iters = 20000
signal_normalization = True
griffin_lim_iters = 60
power = 1.5


# TTS-DNN
max_sep_len = 5000
num_phn = 384
kernel_size = 5
stride = 1
padding = 2
encoder_dim = 512
encoder_n_layer = 1
decoder_dim = 512
decoder_n_layer = 2
dropout = 0.1
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
num_position = 3001
position_dim = 32


# Train
checkpoint_path = "./model_new"
logger_path = "./logger"
mel_ground_truth = "./mels"

batch_size = 32
epochs = 1000
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20
