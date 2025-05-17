import sys
sys.path.append('/linhaitao/lsh/openstl_weather/')
import copy

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary
import math
from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class layerNormFeedForward(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.ff1 = TAUSubBlock(dim=d_model)
        # self.ff1 = GASubBlock(dim=d_model)
        

    def forward(self, x):
        b, c , t , h ,w = x.shape

        x = x + self.ff1(x.reshape(b*c,t, h, w)).view(b, c, t, h, w)
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 使用单个矩阵一次性计算出queries,keys,values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # 将queries，keys和values划分为num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
        
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # 在最后一个维度上相加
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)

        att = F.softmax(energy, dim=-1) / scaling

        att = self.att_drop(att)

        
        # 在第三个维度上相加
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 256, patch_size: int = 4, emb_size: int = 4096, img_size= (16, 32)):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 将卷积操作后的patch铺平
            Rearrange('b e h w -> b (h w) e'),
        )
        self.positions = nn.Parameter(torch.randn((img_size[0] // patch_size)*(img_size[1] // patch_size), emb_size))
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x += self.positions
        return x


class CIAttPatchBlock(nn.Module):
    def __init__(self, d_model, heads, emb_size = 256, patch_size = 4, img_size= (16, 32)):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.img_size = img_size
        self.emb_size = d_model
        self.patch_emb = PatchEmbedding(in_channels=d_model,patch_size=patch_size,emb_size=d_model,img_size=img_size)
        self.projection = nn.Linear(d_model, patch_size*patch_size*d_model)
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.attn_1 = MultiHeadAttention(emb_size=d_model,num_heads=heads,dropout=0.1) # 多头注意力机制
        self.ff = layerNormFeedForward(d_model) # 前馈
        # self.ff = layerNormFeedForward2Hid(d_model)

    def forward(self, x):
        b, s, c, h, w = x.size() # 16 4 256 16 32

        x_pathes = self.patch_emb(x.reshape(-1,c,h,w)).reshape(b,-1,self.emb_size)


        x = x_pathes + self.attn_1(x_pathes) # b 4*... c
        x = self.projection(x)
        x = x.reshape(b,s,-1,self.patch_size*self.patch_size,c).permute(0, 1, 4, 2, 3).reshape(b,s,c,self.img_size[0]//self.patch_size,self.img_size[1]//self.patch_size,-1)
        x = rearrange(x, 'b s c h_p w_p (p1 p2) -> b s c (h_p p1) (w_p p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.norm_1(x.view(-1, c, h, w)).view(b, s, c, h, w) # 单独的对每个通道进行独立的归一化

        x = x + self.ff(x)
        x = self.norm_2(x.view(-1, c, h, w)).view(b, s, c, h, w) # 单独的对每个通道进行独立的归一化
        return x

class CIPatchMidNet(nn.Module):
    def __init__(self, in_channels, d_model, n_layers, heads):
        super().__init__()
        self.N = n_layers
        self.d_model = d_model
        self.conv1 = nn.Conv2d(in_channels,d_model,1)
        self.layers = get_clones(CIAttPatchBlock(d_model, heads), n_layers)
        self.conv2 = nn.Conv2d(d_model,in_channels,1)


    def forward(self, x):
        
        b, c, t, h, w = x.shape
        x = x.reshape(b*c,t,h,w)
        x = self.conv1(x).reshape(b,c,-1,h,w)
        for i in range(self.N):
            x = self.layers[i](x)
        x = x.reshape(b*c,-1,h,w)
        x = self.conv2(x).reshape(b,c,-1,h,w)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.randn((16, 4, 32, 16, 32)).to(device) # b t c h w
    model = CIPatchMidNet(in_channels=32,d_model=256, n_layers=6, heads=8).to(device)
    summary(model, (data.shape), device=device)
    
    