method = 'predformer'

lr = 1e-3

height: 128
width: 128
num_channels: 1
# video length in and out
pre_seq: 12
after_seq: 12
# patch size
patch_size: 16
dim: 256
heads: 8
dim_head: 32
# dropout
dropout: 0.0
attn_dropout: 0.0
drop_path: 0.0
scale_dim: 4
# depth
depth: 1
Ndepth: 6 # For FullAttention-24, for BinaryST, BinaryST, FacST, FacTS-12, for TST,STS-8, for TSST, STTS-6


