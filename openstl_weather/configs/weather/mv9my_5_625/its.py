method = 'its'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'TAU'  # SimVP.V1
hid_S = 32
hid_T = 256 # 256
N_T = 8
N_S = 2
# training
lr = 2.5e-4 # 1e-3 
batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0
epoch = 50

momentum_ema = 0.9
