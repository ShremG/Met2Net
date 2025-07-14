method = 'simvp_merge'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = 'tau'
hid_S = 32
hid_T = 256
N_T = 8
N_S = 2
alpha = 0.1

# merge model
ch=32 
ch_mult=(1, 1, 2) # 控制merge dec的上采样次数 = len - 1 (1, 2)  (1, 1, 2) 
resolution=64
num_res_blocks=4 
num_att_blocks=4
num_heads=8
merge_ratio=None
merge_num=798 # 64 -> 32 4096 - 1024 = 3072 32 -> 16 1024 - 256 = 798
patch_size=2 # 64 /1 64 /2 
model_path=None
cand_distribution='gaussian-6'
isotropic=256
z_channels=18

# training
lr = 1e-3
batch_size = 16
val_batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0