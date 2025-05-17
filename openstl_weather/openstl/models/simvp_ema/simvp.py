import sys
sys.path.append('/linhaitao/lsh/openstl_weather/')
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


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

        self.enc_u10_q = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_u10_k = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)

        self.enc_v10_q = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_v10_k = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)

        self.enc_t2m_q = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_t2m_k = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)

        self.enc_tcc_q = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_tcc_k = Encoder(1, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)

        self.dec_u10_q = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.dec_u10_k = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.dec_v10_q = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.dec_v10_k = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.dec_t2m_q = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.dec_t2m_k = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.dec_tcc_q = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.dec_tcc_k = Decoder(hid_S, 1, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.init_weights()

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S*C, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*hid_S*C, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    
    def init_weights(self):
        """Initialize the weights of model.
        """
        for param_q, param_k in zip(self.enc_u10_q.parameters(),
                                    self.enc_u10_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
        for param_q, param_k in zip(self.enc_v10_q.parameters(),
                                    self.enc_v10_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
        for param_q, param_k in zip(self.enc_t2m_q.parameters(),
                                    self.enc_t2m_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
        for param_q, param_k in zip(self.enc_tcc_q.parameters(),
                                    self.enc_tcc_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)

        for param_q, param_k in zip(self.dec_u10_q.parameters(),
                                    self.dec_u10_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
        for param_q, param_k in zip(self.dec_v10_q.parameters(),
                                    self.dec_v10_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
        for param_q, param_k in zip(self.dec_t2m_q.parameters(),
                                    self.dec_t2m_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
        for param_q, param_k in zip(self.dec_tcc_q.parameters(),
                                    self.dec_tcc_k.parameters()):
            param_k.requires_grad = False
            param_k.data.copy_(param_q.data)
        
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.enc_u10_q.parameters(),
                                    self.enc_u10_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
        
        for param_q, param_k in zip(self.enc_v10_q.parameters(),
                                    self.enc_v10_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
        
        for param_q, param_k in zip(self.enc_t2m_q.parameters(),
                                    self.enc_t2m_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
        
        for param_q, param_k in zip(self.enc_tcc_q.parameters(),
                                    self.enc_tcc_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
    
    def _momentum_update_key_decoder(self):
        """Momentum update of the key decoder."""
        for param_q, param_k in zip(self.dec_u10_q.parameters(),
                                    self.dec_u10_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
        
        for param_q, param_k in zip(self.dec_v10_q.parameters(),
                                    self.dec_v10_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
        
        for param_q, param_k in zip(self.dec_t2m_q.parameters(),
                                    self.dec_t2m_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
        
        for param_q, param_k in zip(self.dec_tcc_q.parameters(),
                                    self.dec_tcc_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1. - self.momentum)
            # param_k.requires_grad = False
        

    def forward(self, x_raw, y_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.clone()
        if self.training:
            x,_ = self.mask_replace_blocks(x)
            x = self.mask_time_dimension_global_mean(x)
        _, T1, _, _, _ = y_raw.shape
        x = x.view(B*T, C, H, W)
        y = y_raw.view(B*T1, C, H, W)
        # 重建
        h_u10, _ = self.enc_u10_q(x[:,0:1,:,:])
        h_v10, _ = self.enc_v10_q(x[:,1:2,:,:])
        h_t2m, _ = self.enc_t2m_q(x[:,2:3,:,:])
        h_tcc, _ = self.enc_tcc_q(x[:,3:4,:,:])

        rec_u10 = self.dec_u10_q(h_u10)
        rec_v10 = self.dec_v10_q(h_v10)
        rec_t2m = self.dec_t2m_q(h_t2m)
        rec_tcc = self.dec_tcc_q(h_tcc)
        rec_x = torch.cat((rec_u10, rec_v10, rec_t2m, rec_tcc), dim=1)
        rec_x = rec_x.reshape(B,T,C,H,W)
        
        loss_rec = F.mse_loss(rec_x,x_raw)

        self._momentum_update_key_encoder()
        self._momentum_update_key_decoder()

        h_u10, s_u10 = self.enc_u10_k(x[:,0:1,:,:])
        h_v10, s_v10 = self.enc_v10_k(x[:,1:2,:,:])
        h_t2m, s_t2m = self.enc_t2m_k(x[:,2:3,:,:])
        h_tcc, s_tcc = self.enc_tcc_k(x[:,3:4,:,:])
        H_, W_ = h_tcc.shape[-2],h_tcc.shape[-1]
        z = torch.cat((h_u10, h_v10, h_t2m, h_tcc), dim=1)
        z = z.reshape(B,T,-1, H_, W_)
        hid = self.hid(z)

        hid = hid.reshape(B*T, -1, H_, W_)
        h_pre_u10 = hid[:,:self.hid_s,:,:]
        h_pre_v10 = hid[:,self.hid_s:2*self.hid_s,:,:]
        h_pre_t2m = hid[:,2*self.hid_s:3*self.hid_s,:,:]
        h_pre_tcc = hid[:,3*self.hid_s:4*self.hid_s,:,:]

        h_y_u10, _ = self.enc_u10_k(y[:,0:1,:,:])
        h_y_v10, _ = self.enc_v10_k(y[:,1:2,:,:])
        h_y_t2m, _ = self.enc_t2m_k(y[:,2:3,:,:])
        h_y_tcc, _ = self.enc_tcc_k(y[:,3:4,:,:])
        
        u10_latent = F.mse_loss(h_pre_u10,h_y_u10)
        v10_latent = F.mse_loss(h_pre_v10,h_y_v10)
        t2m_latent = F.mse_loss(h_pre_t2m,h_y_t2m)
        tcc_latent = F.mse_loss(h_pre_tcc,h_y_tcc)
        loss_latent = torch.mean(torch.stack([u10_latent, v10_latent, t2m_latent, tcc_latent]))

        pre_u10 = self.dec_u10_k(h_pre_u10)
        pre_v10 = self.dec_v10_k(h_pre_v10)
        pre_t2m = self.dec_t2m_k(h_pre_t2m)
        pre_tcc = self.dec_tcc_k(h_pre_tcc)

        # # dec 会受pre_loss的影响
        # pre_u10 = self.dec_u10_q(h_pre_u10)
        # pre_v10 = self.dec_v10_q(h_pre_v10)
        # pre_t2m = self.dec_t2m_q(h_pre_t2m)
        # pre_tcc = self.dec_tcc_q(h_pre_tcc)
        pre_y = torch.cat((pre_u10, pre_v10, pre_t2m, pre_tcc), dim=1)
        pre_y = pre_y.reshape(B,T,C,H,W)

        loss_pre = F.mse_loss(pre_y,y_raw)

        loss = loss_rec + loss_latent + 0.01*loss_pre
        
        return pre_y, loss, loss_rec, loss_latent, loss_pre

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
    data = torch.randn((16, 12, 4, 32, 32)).to(device)
    model = SimVP_Model(in_shape=[12,4,32,64],hid_S=32).to(device)
    summary(model, [data.shape,data.shape], device=device)