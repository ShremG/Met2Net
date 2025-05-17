import sys
sys.path.append('/lishaohan/code/openstl_weather/')
import copy

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
import math
from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)

from openstl.models.ITS.module.superTokenAttention import Unfold, Fold
from openstl.models.ITS.module.mid_tokenmixer_ffnpool import GASuperTokenSubBlock


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class layerNormFeedForward(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.ff1 = GASuperTokenSubBlock(dim=d_model,stoken_size=[8,8])
        # self.ff1 = GASubBlock(dim=d_model)
        

    def forward(self, x):
        b, c , t , h ,w = x.shape

        x = x + self.ff1(x.reshape(b*c,t, h, w)).view(b, c, t, h, w)
        
        return x



def attention_s(q, k, v):
    # q,k,v : [b*heads, s, c, h, w]
    # [b*heads,s,c*h*w] * [b*heads,c*h*w,s] => [b*heads,s,s]
    scores_s = torch.matmul(q.view(q.size(0), q.size(1), -1), k.view(k.size(0), k.size(1), -1).permute(0, 2, 1)) \
             / math.sqrt(q.size(2) * q.size(3) * q.size(4))

    # [b*heads,s,s] 按照最后一维 经过softmax 得到相应的权重
    scores_s = F.softmax(scores_s, dim=-1)

    # [b*heads,s,s] * [b*heads,s,c*h*w] =>  [b*heads,s,c*h*w]
    v_s = torch.matmul(scores_s, v.reshape(v.size(0), v.size(1), -1))

    # [b*heads,s,c*h*w] => [b*heads, s, c, h, w]
    output = v_s.reshape(v.size())
    return output


class MultiHeadAttention_S(nn.Module):
    def __init__(self, heads, d_model, qk_dim=32):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.h = heads

        self.q_Conv = nn.Sequential(nn.Conv2d(self.d_model, qk_dim, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, qk_dim),
                                    )
        self.v_Conv = nn.Sequential(nn.Conv2d(self.d_model, qk_dim, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, qk_dim),
                                    )
        self.k_Conv = nn.Sequential(nn.Conv2d(self.d_model, qk_dim, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, qk_dim),
                                    )
        self.v_post_f = nn.Sequential(
            nn.Conv2d(qk_dim, d_model, 1, 1, 0, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=False),
        )

    def forward(self, q, k, v):
        b_q, s_q, c_q, h_q, w_q = q.size()
        b_k, s_k, c_k, h_k, w_k = k.size()
        b_v, s_v, c_v, h_v, w_v = v.size()
        # wq * x；输入[b*s,c,h,w] 输出 q : [b*s,heads,c//heads,h,w]；heads 为多头注意力的头的数量
        q = self.q_Conv(q.reshape(q.size(0) * q.size(1), *q.shape[2:])).reshape(q.size(0)*q.size(1), self.h,
                                                                                self.qk_dim // self.h, h_q, w_q)

        # 将q的形状改为 [b,heads,s,c,h,w]
        q = q.reshape(b_q, s_q, self.h, self.qk_dim // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        # [b*heads,s,c,h,w]
        q = q.reshape(q.size(0)*q.size(1), *q.shape[2:])

        # wk * x 得到 k
        k = self.k_Conv(k.reshape(k.size(0) * k.size(1), *k.shape[2:])).reshape(k.size(0) * k.size(1), self.h,
                                                                                self.qk_dim // self.h, h_q, w_q)
        k = k.reshape(b_k, s_k, self.h, self.qk_dim // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        # [b*heads,s,c,h,w]
        k = k.reshape(k.size(0) * k.size(1), *k.shape[2:])

        # wv * x 得到 v
        v = self.v_Conv(v.reshape(v.size(0) * v.size(1), *v.shape[2:])).reshape(v.size(0) * v.size(1), self.h,
                                                                                self.qk_dim // self.h, h_v, w_v)
        v = v.reshape(b_v, s_v, self.h, self.qk_dim // self.h, h_v, w_v).permute(0, 2, 1, 3, 4, 5)
        # [b*heads,s,c,h,w]
        v = v.reshape(v.size(0) * v.size(1), *v.shape[2:])
        # [b, s, heads, c, h, w]
        output = attention_s(q, k, v).reshape(b_q, self.h, s_q, self.qk_dim // self.h, h_v, w_v).permute(0, 2, 1, 3, 4, 5)
        # [b, s, c, h, w]
        output = self.v_post_f(output.reshape(b_q*s_q, self.h, self.qk_dim // self.h,
                                              h_v, w_v).reshape(b_q*s_q, self.qk_dim, h_v, w_v)).view(b_v, s_v, c_v, h_v, w_v)

        return output


class CIAttBlock(nn.Module):
    def __init__(self, d_model, heads, stoken_size=[8,8]):
        super().__init__()
        self.stoken_size = stoken_size
        self.d_model = d_model
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.attn_1 = MultiHeadAttention_S(heads, d_model) # 多头注意力机制
        self.ff = layerNormFeedForward(d_model) # 前馈
        # self.ff = layerNormFeedForward2Hid(d_model)
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        self.n_iter = 1
        self.scale = d_model ** - 0.5

    def forward(self, x):
        b, s, c, h, w = x.size()

        sh, sw = self.stoken_size
        hh, ww = h // sh, w // sw
        xx = x.reshape(-1,c,h,w)
        B,C,H,W = xx.shape
        stoken_features = F.adaptive_avg_pool2d(xx, (hh, ww))
        pixel_features = x.reshape(B, C, hh, sh, ww, sw).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, sh * sw, C)
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)

                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)

        stoken_features = stoken_features.reshape(b,s,c,hh,ww)
        # stoken_features = self.stoken_refine(stoken_features)
        x = x + self.attn_1(stoken_features, stoken_features, x) # q k v

        x = x + self.attn_1(x, x, x) # q k v
        x = self.norm_1(x.view(-1, c, h, w)).view(b, s, c, h, w) # 单独的对每个通道进行独立的归一化

        x = x + self.ff(x)
        x = self.norm_2(x.view(-1, c, h, w)).view(b, s, c, h, w) # 单独的对每个通道进行独立的归一化
        return x

class CIMidNet(nn.Module):
    def __init__(self, in_channels, d_model, n_layers, heads):
        super().__init__()
        self.N = n_layers
        self.d_model = d_model
        self.conv1 = nn.Conv2d(in_channels,d_model,1)
        self.layers = get_clones(CIAttBlock(d_model, heads), n_layers)
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
    data = torch.randn((16, 4, 12*32, 16, 32)).to(device) # b t c h w
    model = CIMidNet(in_channels=12*32,d_model=256, n_layers=6, heads=8).to(device)
    y = model(data)
    summary(model, (data.shape), device=device)
    
    