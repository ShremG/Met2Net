import sys
sys.path.append('/home/gc/projects/openstl_weather/')
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
import torchvision.transforms as transforms
import numpy as np
from timm.layers import DropPath, trunc_normal_

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)
from openstl.models.simvp_ema.ciatt_modules import CIMidNet


class SimVP_Model(nn.Module):

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, momentum_ema=0.9,**kwargs):
        super(SimVP_Model, self).__init__()
        self.momentum=momentum_ema
        self.hid_s = hid_S
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False

        self.enc_q_list = nn.ModuleList([Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace) for _ in range(C)])
        self.enc_k_list = nn.ModuleList([Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace) for _ in range(C)])

        self.dec_q_list = nn.ModuleList([Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace) for _ in range(C)])
        self.dec_k_list = nn.ModuleList([Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace) for _ in range(C)])

        self.hid_q = CIMidNet(in_channels=T*hid_S,d_model=hid_T, n_layers=N_T, heads=8)
        self.hid_k = CIMidNet(in_channels=T*hid_S,d_model=hid_T, n_layers=N_T, heads=8)
        self.init_weights()
    
    def init_weights(self):
        """Initialize the weights of model.
        """

        for enc_q, enc_k in zip(self.enc_q_list,self.enc_k_list):
            for param_q, param_k in zip(enc_q.parameters(),
                                        enc_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)

        
        for dec_q, dec_k in zip(self.dec_q_list,self.dec_k_list):
            for param_q, param_k in zip(dec_q.parameters(),
                                        dec_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)


        for param_q, param_k in zip(self.hid_q.parameters(),
                                    self.hid_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for enc_q, enc_k in zip(self.enc_q_list,self.enc_k_list):
            for param_q, param_k in zip(enc_q.parameters(),
                                        enc_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)

    
    def _momentum_update_key_decoder(self):
        """Momentum update of the key decoder."""
        for dec_q, dec_k in zip(self.dec_q_list,self.dec_k_list):
            for param_q, param_k in zip(dec_q.parameters(),
                                        dec_k.parameters()):
                param_k.requires_grad = False
                param_k.data.copy_(param_q.data)
    
    def _momentum_update_key_translator(self):
        """Momentum update of the key translator."""
        for param_q, param_k in zip(self.hid_q.parameters(),
                                    self.hid_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    def forward(self, x_raw, y_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.clone()
        y = y_raw.clone()
        # if self.training:
        #     x,_ = self.mask_replace_blocks(x)
        #     x = self.mask_time_dimension_global_mean(x)
        _, T1, _, _, _ = y_raw.shape
        x = x.view(B*T, C, H, W)
        y = y.view(B*T1, C, H, W)
        # 一阶段
        h_s1 = []
        for i in range(C):
            hh, _ = self.enc_q_list[i](x[:,i:i+1,:,:])
            h_s1.append(hh)

        H_, W_ = h_s1[-1].shape[-2],h_s1[-1].shape[-1]

        z_x = torch.stack(h_s1, dim=1) # b*t,n,c,h,w
        z_x = z_x.reshape(B,T,C,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B,C,-1,H_, W_)
        self.hid_k.train()
        z_y_pre = self.hid_k(z_x)
        z_y_pre = z_y_pre.reshape(B,C,T,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B*T,C,-1,H_, W_)

        rec_list = []
        for i in range(C):
            rec = self.dec_q_list[i](z_y_pre[:,i,:,:])
            rec_list.append(rec)
        rec_y = torch.stack(rec_list, dim=1)
        rec_y = rec_y.reshape(B,T,-1,H,W)

        loss_rec = F.mse_loss(rec_y,y_raw)

        # 二阶段
        self._momentum_update_key_encoder()
        self._momentum_update_key_decoder()


        h_s2 = []
        for i in range(C):
            hh, _ = self.enc_k_list[i](x[:,i:i+1,:,:])
            h_s2.append(hh)
        z_x = torch.stack(h_s2, dim=1) # b*t,4,c,h,w
        z_x = z_x.reshape(B,T,C,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B,C,-1,H_, W_)
        self.hid_k.train()
        z_y_pre = self.hid_q(z_x)
        z_y_pre = z_y_pre.reshape(B,C,T,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B*T,C,-1,H_, W_)


        h_y = []
        for i in range(C):
            hh, _ = self.enc_k_list[i](y[:,i:i+1,:,:])
            h_y.append(hh)

        z_y = torch.stack(h_y, dim=1) # b*t,4,c,h,w
        loss_latent = F.mse_loss(z_y_pre,z_y)

        self._momentum_update_key_translator()

        pre_y_list = []
        for i in range(C):
            rec = self.dec_k_list[i](z_y_pre[:,i,:,:])
            pre_y_list.append(rec)
        pre_y = torch.stack(pre_y_list, dim=1)
        pre_y = pre_y.reshape(B,T,-1,H,W)

        loss_pre = F.mse_loss(pre_y,y_raw)

        loss = loss_rec + loss_latent + 0*loss_pre

        return pre_y, loss, loss_rec, loss_latent, loss_pre
    
    def sample(self,batch_x):
        B, T, C, H, W = batch_x.shape
        x = batch_x.clone()
        x = x.view(B*T, C, H, W)

        h_s1 = []
        for i in range(C):
            hh, _ = self.enc_q_list[i](x[:,i:i+1,:,:])
            h_s1.append(hh)
        H_, W_ = h_s1[-1].shape[-2],h_s1[-1].shape[-1]
        z_x = torch.stack(h_s1, dim=1) # b*t,4,c,h,w
        z_x = z_x.reshape(B,T,C,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B,C,-1,H_, W_)

        z_y_pre = self.hid_q(z_x)
        z_y_pre = z_y_pre.reshape(B,C,T,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B*T,C,-1,H_, W_)

        pre_y_list = []
        for i in range(C):
            rec = self.dec_q_list[i](z_y_pre[:,i,:,:])
            pre_y_list.append(rec)
        rec_y = torch.stack(pre_y_list, dim=1)
        rec_y = rec_y.reshape(B,T,-1,H,W)
        return rec_y

    def mask_replace_blocks(self, tensor):
        # Generate a random number and decide to mask or not based on it
        do_mask = torch.rand(1).item() < 0.5  # 50% probability to mask

        # If do_mask is False, return the original tensor and a None mask
        if not do_mask:
            return tensor, None
        # Calculate the mean of each channel across the entire batch
        channel_means = tensor.mean(dim=(0, 1, -2, -1), keepdim=True)

        # Get basic dimensions of the tensor
        b, t, c, h, w = tensor.shape

        # Generate a random mask for 2x2 blocks
        block_mask = torch.rand(b, t, c, h//2, w//2) < 0.5  # 30% probability

        # Expand the block mask to cover the entire dimensions of the blocks
        mask = block_mask.repeat(1, 1, 1, 2, 2)

        # Broadcast the channel means across the tensor dimensions
        channel_means_expanded = channel_means.repeat(b, t, 1, h, w)

        # Apply the mask and replace the selected blocks
        tensor[mask] = channel_means_expanded[mask]

        return tensor, mask
    
    def mask_time_dimension_global_mean(self,tensor):
        # Generate a random number and decide to mask or not based on it
        do_mask = torch.rand(1).item() < 0.5  # 50% probability to mask

        # If do_mask is False, return the original tensor and a None mask
        if not do_mask:
            return tensor
        # Calculate the global mean across the entire tensor
        global_mean = tensor.mean()

        # Get basic dimensions of the tensor
        b, t, c, h, w = tensor.shape

        # Generate a random mask to select 20% of the time steps for replacement
        time_mask = torch.rand(b, t) < 0.5  # 20% probability

        # Expand the time mask to cover the channels, height, and width
        time_mask_expanded = time_mask.view(b, t, 1, 1, 1).expand(b, t, c, h, w)

        # Replace the selected time steps with the global mean
        tensor[time_mask_expanded] = global_mean

        return tensor
    
def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        # latent = self.pre_vq_conv(latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        # self.readout = nn.Conv3d(C_hid, C_out, 1)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        # hid = self.post_vq_conv(hid)
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        if enc1 != None:
            Y = self.dec[-1](hid + enc1)
        else:
            Y = self.dec[-1](hid)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)

class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y

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

class BasicConv3d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv3d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = SamePadConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=(1,2,2))
            # self.conv = nn.Sequential(*[
            #     nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
            #               stride=1, padding=padding, dilation=dilation),
            #     nn.PixelShuffle(2)
            # ])
        else:
            self.conv = SamePadConv3d(in_channels, out_channels, kernel_size=4, stride=1)
            # self.conv = nn.Conv2d(
            #     in_channels, out_channels, kernel_size=kernel_size,
            #     stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC3D(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC3D, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv3d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.randn((16, 12, 8, 32, 64)).to(device)
    model = SimVP_Model(in_shape=[12,8,32,64],hid_S=32,N_S=2,N_T=8,hid_T=256).to(device)
    summary(model, [data.shape,data.shape], device=device)