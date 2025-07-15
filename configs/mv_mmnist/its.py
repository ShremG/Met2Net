method = 'its'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'TAU'  # SimVP.V1
hid_S = 64
hid_T = 512 # 256
N_T = 8
N_S = 4
# training
lr = 5e-3 # 1e-2
batch_size = 16
drop_path = 0.
sched = 'cosine'
warmup_epoch = 0
epoch = 200

momentum_ema = 0.999
