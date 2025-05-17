
import sys
sys.path.append('/linhaitao/lsh/OpenSTL-OpenSTL-Lightning/')
import math
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .attention import MultiHeadAttention
from .utils import shift_dim
from torchinfo import summary
from collections import namedtuple

class Encoder_mv(nn.Module):
    def __init__(self, n_hiddens=32, n_res_layers=4, downsample=(2,2,2), embedding_dim=4, in_channels=4, **kwargs):
        super(Encoder_mv, self).__init__()
        self.encoder = Encoder(n_hiddens, n_res_layers, downsample, in_channels)
        self.pre_vq_conv = SamePadConv3d(n_hiddens, embedding_dim, 1)

    def forward(self, imgs):
        x = imgs.clone()
        x = x.permute(0,2,1,3,4)
        z = self.pre_vq_conv(self.encoder(x))
        return z


class Decoder_mv(nn.Module):
    def __init__(self, n_hiddens=32, n_res_layers=4, downsample=(2,2,2), embedding_dim=4, out_channels=4,**kwargs):
        super(Decoder_mv, self).__init__()

        self.decoder = Decoder(n_hiddens, n_res_layers, downsample, out_channels)
        self.post_vq_conv = SamePadConv3d(embedding_dim, n_hiddens, 1)

    def forward(self, z):
        y = self.decoder(self.post_vq_conv(z))
        return y.permute(0,2,1,3,4)

class AudoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(AudoEncoder, self).__init__()
        # 使用 namedtuple 创建一个 Config 类型
        Config = namedtuple('Config', kwargs.keys())
        # 使用 Config 类创建一个 config 对象
        self.args  = Config(**kwargs)
        self.embedding_dim = self.args.embedding_dim
        self.n_codes = self.args.n_codes
        self.encoder = Encoder(self.args.n_hiddens, self.args.n_res_layers, self.args.downsample, self.args.in_channels)
        self.decoder = Decoder(self.args.n_hiddens, self.args.n_res_layers, self.args.downsample, self.args.out_channels)

        self.pre_vq_conv = SamePadConv3d(self.args.n_hiddens, self.args.embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(self.args.embedding_dim, self.args.n_hiddens, 1)

    def forward(self, imgs):
        x = imgs.clone()
        x = x.permute(0,2,1,3,4)
        z = self.pre_vq_conv(self.encoder(x))
        # z = self.high_pass_filter_torch(z,cutoff=2) # fft转换然后过高通滤波器 保留高频信息
        z = self.low_pass_filter_torch(z,cutoff=self.args.cutoff) # fft转换然后过低通滤波器 保留低频信息
        # z = self.mid_pass_filter_torch(z)
        # z = self.apply_gaussian_filter(z)
        x_recon = self.decoder(self.post_vq_conv(z))
        recon_loss = F.mse_loss(x_recon, x)
        x_recon = x_recon.permute(0,2,1,3,4)
        
        return recon_loss, x_recon
        

    def encode(self, x): # b t c h w
        x = x.permute(0,2,1,3,4) # b c t h w
        z = self.pre_vq_conv(self.encoder(x))
        # z = self.high_pass_filter_torch(z,cutoff=2)
        z = self.low_pass_filter_torch(z,cutoff=self.args.cutoff) # fft转换然后过低通滤波器 保留低频信息
        # z = self.mid_pass_filter_torch(z)
        # z = self.apply_gaussian_filter(z)
        return z

    def decode(self, z): # b c t h w
        h = self.post_vq_conv(z)
        return self.decoder(h).permute(0,2,1,3,4)

    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state_dict'])
    
    def high_pass_filter_torch(self, tensor, cutoff=2):
        """
        对PyTorch张量进行高通滤波处理，保留高频信息。

        参数:
        - tensor: 输入的PyTorch张量，维度为 (batch, dim, t, h, w)。
        - cutoff: 滤波器的截止频率，控制保留高频信息的范围，默认值为3。

        返回:
        - filtered_tensor: 经过高通滤波处理后的PyTorch张量。
        """
        batch, dim, t, h, w = tensor.shape

        # 对图像的最后两个维度进行傅里叶变换
        fft_tensor = torch.fft.fftshift(torch.fft.fftn(tensor, dim=(-2, -1)), dim=(-2, -1))

        # 创建一个高通滤波器
        crow, ccol = h // 2 , w // 2
        mask = torch.ones((h, w), device=tensor.device)
        mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0

        # 将掩码应用于频域
        filtered_fft_tensor = fft_tensor * mask

        # 逆傅里叶变换回到空间域
        filtered_tensor = torch.fft.ifftn(torch.fft.ifftshift(filtered_fft_tensor, dim=(-2, -1)), dim=(-2, -1)).real

        return filtered_tensor
    
    def low_pass_filter_torch(self, tensor, cutoff=3):
        """
        对PyTorch张量进行低通滤波处理，去除高频噪声，保留低频信息。

        参数:
        - tensor: 输入的PyTorch张量，维度为 (batch, dim, t, h, w)。
        - cutoff: 滤波器的截止频率，控制保留低频信息的范围，默认值为2。

        返回:
        - filtered_tensor: 经过低通滤波处理后的PyTorch张量。
        """
        batch, dim, t, h, w = tensor.shape
        # 对图像的最后两个维度进行傅里叶变换
        fft_tensor = torch.fft.fftshift(torch.fft.fftn(tensor, dim=(-2, -1)), dim=(-2, -1))
        # 创建一个低通滤波器
        crow, ccol = h // 2 , w // 2
        mask = torch.zeros((h, w), device=tensor.device)
        mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
        # 将掩码应用于频域
        filtered_fft_tensor = fft_tensor * mask
        # 逆傅里叶变换回到空间域
        filtered_tensor = torch.fft.ifftn(torch.fft.ifftshift(filtered_fft_tensor, dim=(-2, -1)), dim=(-2, -1)).real
        return filtered_tensor

    def mid_pass_filter_torch(self, tensor, low_cutoff=1, high_cutoff=3):
        """
        对五维PyTorch张量进行带通滤波处理，保留中频信息，带区是圆形的。

        参数:
        - tensor: 输入的PyTorch张量，维度为 (batch, dim, t, h, w)。
        - low_cutoff: 圆形带通滤波器的内圆半径，滤除半径小于此值的频率。
        - high_cutoff: 圆形带通滤波器的外圆半径，滤除半径大于此值的频率。

        返回:
        - filtered_tensor: 经过带通滤波处理后的PyTorch张量。
        """
        batch, dim, t, h, w = tensor.shape

        # 对图像的最后两个维度进行傅里叶变换
        fft_tensor = torch.fft.fftshift(torch.fft.fftn(tensor, dim=(-2, -1)), dim=(-2, -1))

        # 创建一个圆形带通滤波器
        crow, ccol = h // 2, w // 2
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        distance = torch.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
        mask = torch.logical_and(distance >= low_cutoff, distance <= high_cutoff).float().to(tensor.device)

        # 将掩码扩展到所有维度
        mask = mask.expand(batch, dim, t, h, w)

        # 将掩码应用于频域
        filtered_fft_tensor = fft_tensor * mask

        # 逆傅里叶变换回到空间域
        filtered_tensor = torch.fft.ifftn(torch.fft.ifftshift(filtered_fft_tensor, dim=(-2, -1)), dim=(-2, -1)).real

        return filtered_tensor

    def gaussian_blurse(self,kernel_size, sigma):
        """生成高斯核."""
        from math import ceil, pi, exp

        # 确保kernel_size是奇数
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        k = kernel_size // 2

        # 计算高斯核
        kernel = torch.tensor([[exp(-(i**2 + j**2) / (2 * sigma**2)) for j in range(-k, k+1)] for i in range(-k, k+1)])
        kernel /= kernel.sum()

        return kernel

    def apply_gaussian_filter(self,input_tensor, kernel_size=3, sigma=0.8): # 3 0.8 5 1.5
        """应用高斯滤波器到整个五维张量."""
        b,dim,t,h,w = input_tensor.shape
        # 创建高斯核
        kernel = self.gaussian_blurse(kernel_size, sigma)
        kernel = kernel.to(input_tensor.device).type(input_tensor.dtype)
        
        # 需要为每个通道和每个时间步创建一个单独的卷积核
        total_channels = input_tensor.shape[1] * input_tensor.shape[2]
        kernel = kernel.expand(total_channels, 1, kernel_size, kernel_size)

        # Padding
        padding = kernel_size // 2

        # 重新组织张量以适应分组卷积
        reshaped_input = input_tensor.view(b,-1,h,w)  # (batch*depth, channel, height, width)

        # 应用高斯滤波
        output_tensor = F.conv2d(reshaped_input, kernel, padding=padding, groups=total_channels)
        output_tensor = output_tensor.view(input_tensor.shape)

        return output_tensor

class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])# 32 5 16 160
        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity,distances=distances.reshape(z.shape[0],
                                                                                                       z.shape[2],
                                                                                                       z.shape[3],
                                                                                                       z.shape[4],
                                                                                                       -1))

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample, in_channels=1):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        # max_ds = nums_enc
        for i in range(max_ds):
            in_channels = in_channels if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(n_hiddens, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample, out=1,nums_enc=4):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        # for i in range(nums_enc - max_us):
        #     convt = SamePadConv3d(n_hiddens, n_hiddens, 4, stride=1)
        #     self.convts.append(convt)

        for i in range(nums_enc - max_us, nums_enc):
            out_channels = out if i == nums_enc - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

        # convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4,stride=(2,1,1))
        # self.convts.append(convt)
        # self.conv3d = nn.Conv3d(n_hiddens,out,1)

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts)-1:
                h = F.relu(h)
        # h = self.conv3d(h)
        return h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))
    
class InterpolateAndConv3d(nn.Module):
    def __init__(self, in_channels ,out_channels, kernel_size, target_size=None, stride=1,mode='trilinear'):
        """
        初始化模块。
        :param out_channels: 卷积层的输出通道数。
        :param kernel_size: 卷积核的大小，可以是一个整数或者一个元组 (k_d, k_h, k_w)。
        :param target_size: 插值的目标尺寸，形式为 (depth, height, width)。
        :param mode: 插值模式，默认为 'trilinear' 三线性插值。
        """
        super(InterpolateAndConv3d, self).__init__()
        self.target_size = target_size
        self.mode = mode
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0)

    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 [batch_size, channels, depth, height, width]。
        :return: 卷积后的张量。
        """
        # 如果指定了目标尺寸，则进行插值
        if self.target_size:
            x = F.interpolate(x, size=self.target_size, mode=self.mode, align_corners=True if self.mode == 'trilinear' else None)

        # 应用3D卷积
        x = self.conv(F.pad(x, self.pad_input))
        return x
    
if __name__ == '__main__':
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser(description="Video-VQVAE")
    # parser.add_argument('--data_path', type=str, default='/home/wilson/data/datasets/bair.hdf5')
    # parser.add_argument('--sequence_length', type=int, default=10)
    # parser.add_argument('--resolution', type=int, default=64)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--n_codes', type=int, default=2048)
    parser.add_argument('--n_hiddens', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--n_res_layers', type=int, default=4)
    parser.add_argument('--downsample', nargs='+', type=int, default=(2, 4, 4))
    args = parser.parse_args()
    config = args.__dict__
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.randn((32, 10, 1, 64, 64)).to(device)
    model = AudoEncoder(**config).to(device)
    summary(model, data.shape, device=device)

