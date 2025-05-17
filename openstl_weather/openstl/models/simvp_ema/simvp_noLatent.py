import sys
sys.path.append('/linhaitao/lsh/openstl_weather/')
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)

from openstl.models.simvp_ema.ciatt_modules import CIMidNet
from openstl.models.simvp_ema.simvp_new_3ddec import ConvSC3D


class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, momentum_ema=0.9,**kwargs):
        super(SimVP_Model, self).__init__()
        self.momentum=momentum_ema
        self.hid_s = hid_S
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False

        self.enc_1 = Encoder(2, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_2 = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_3 = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_4 = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)


        self.dec_1 = Decoder(hid_S, 2, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.dec_2 = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.dec_3 = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.dec_4 = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)


        # self.init_weights()

        # model_type = 'gsta' if model_type is None else model_type.lower()
        # if model_type == 'incepu':
        #     self.hid = MidIncepNet(T*hid_S*4, hid_T, N_T)
        # else:
        #     self.hid = MidMetaNet(T*hid_S*4, hid_T, N_T,
        #         input_resolution=(H, W), model_type=model_type,
        #         mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.hid = CIMidNet(in_channels=T*hid_S,d_model=hid_T, n_layers=N_T, heads=8)
    

    def forward(self, x_raw, y_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        # _, T1, _, _, _ = y_raw.shape
        x = x_raw.view(B*T, C, H, W)
        # y = y_raw.view(B*T1, C, H, W)
        # 重建
        h_u10, _ = self.enc_1(x[:,0:2,:,:])
        h_v10, _ = self.enc_2(x[:,2:3,:,:])
        h_t2m, _ = self.enc_3(x[:,3:4,:,:])
        h_tcc, _ = self.enc_4(x[:,4:5,:,:])

        H_, W_ = h_tcc.shape[-2],h_tcc.shape[-1]
        # z = torch.cat((h_u10, h_v10, h_t2m, h_tcc), dim=1)
        # z = z.reshape(B,T,-1, H_, W_)
        z = torch.stack((h_u10, h_v10, h_t2m, h_tcc), dim=1) # b*t,4,c,h,w
        z = z.reshape(B,T,4,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B,4,-1,H_, W_)
        hid = self.hid(z)
        # hid = hid.reshape(B*T, -1, H_, W_)
        hid = hid.reshape(B,4,T,-1,H_, W_).permute(0,2,1,3,4,5).reshape(B*T,4,-1,H_, W_)

        # h_pre_u10 = hid[:,:self.hid_s,:,:]
        # h_pre_v10 = hid[:,self.hid_s:2*self.hid_s,:,:]
        # h_pre_t2m = hid[:,2*self.hid_s:3*self.hid_s,:,:]
        # h_pre_tcc = hid[:,3*self.hid_s:4*self.hid_s,:,:]
        h_pre_u10 = hid[:,0,:,:]
        h_pre_v10 = hid[:,1,:,:]
        h_pre_t2m = hid[:,2,:,:]
        h_pre_tcc = hid[:,3,:,:]

        pre_u10 = self.dec_1(h_pre_u10)
        pre_v10 = self.dec_2(h_pre_v10)
        pre_t2m = self.dec_3(h_pre_t2m)
        pre_tcc = self.dec_4(h_pre_tcc)

        pre_y = torch.cat((pre_u10, pre_v10, pre_t2m, pre_tcc), dim=1)
        pre_y = pre_y.reshape(B,T,C,H,W)

        loss_pre = F.mse_loss(pre_y,y_raw)

        loss = loss_pre
        
        return pre_y, loss, loss_pre, loss_pre, loss_pre


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
        # self.pre_vq_conv = nn.Sequential(
        #     nn.Conv2d(C_hid, 1, 3, 1, 1),
        #     nn.SiLU()
        # ) 

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
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        # self.post_vq_conv = nn.Sequential(
        #     nn.Conv2d(1, C_hid, 3, 1, 1),
        #     nn.SiLU()
        # ) 

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

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.randn((16, 12, 5, 32, 64)).to(device)
    model = SimVP_Model(in_shape=[12,5,32,64],hid_S=32,hid_T=256,N_S=2,N_T=8).to(device)
    summary(model, [data.shape,data.shape], device=device)