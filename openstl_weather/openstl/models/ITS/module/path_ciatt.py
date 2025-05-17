import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torchinfo import summary

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def posemb_sincos_1d(t, dim, temperature=10000, dtype=torch.float32):
    """
    生成1D正弦余弦位置编码，用于时间维度。
    
    参数：
        t (int): 时间步数（即 T 维度的长度）。
        dim (int): 嵌入维度。
        temperature (int, 可选): 控制频率的温度因子。
        dtype (torch.dtype, 可选): 输出的数据类型。
        
    返回：
        torch.Tensor: 形状为 (T, dim) 的时间位置编码。
    """
    assert (dim % 2) == 0, "Embedding dimension must be even for sin-cos encoding"
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature ** omega)

    # 生成时间步的位置
    time = torch.arange(t)[:, None] * omega[None, :]
    pe = torch.cat((time.sin(), time.cos()), dim=1)
    return pe.type(dtype)


def patch_split(tensor, patch_size=4):
    """
    对 H 和 W 维度进行 patch 拆分，将 (H, W) 转为 (H//patch_size, W//patch_size) 的小块。
    
    参数：
        tensor (torch.Tensor): 输入张量，形状为 (B, N, T, C, H, W)。
        patch_size (int): Patch 的大小，默认为 4。
        
    返回：
        torch.Tensor: 输出张量，形状为 (B, N, T, num_tokens, new_C)。
    """
    B, N, T, C, H, W = tensor.shape
    
    # 确保 H 和 W 是 patch_size 的倍数
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size."

    # 计算新的 token 数量
    num_tokens_h = H // patch_size
    num_tokens_w = W // patch_size
    num_tokens = num_tokens_h * num_tokens_w
    
    # 将 H 和 W 维度拆分成 (H//patch_size, patch_size) 和 (W//patch_size, patch_size)
    tensor = tensor.view(B, N, T, C, H // patch_size, patch_size, W // patch_size, patch_size)
    
    # 调整维度顺序，将小块放到一起
    tensor = tensor.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
    
    # 将 (patch_size, patch_size) 的 patch 合并到 C 维度，得到新的嵌入维度 new_C
    new_C = C * (patch_size ** 2)

    tensor = tensor.view(B, N, T, num_tokens, new_C)

     # 生成空间位置编码，并扩展到 batch 和序列的维度
    pos_emb = posemb_sincos_2d(num_tokens_h, num_tokens_w, new_C).to(tensor.device)  # (num_tokens, new_C)
    pos_emb = pos_emb.view(1, 1, 1, num_tokens, new_C)  # 扩展到 (1, 1, 1, num_tokens, new_C)
    
    # 添加位置编码
    output_tensor = tensor + pos_emb
    
    return output_tensor

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.att = Attention(dim, heads = heads, dim_head = dim_head)
        self.ff = FeedForward(dim, mlp_dim)

    def forward(self, x):
        x = self.att(x) + x
        x = self.norm_1(x)
        x = self.ff(x) + x

        return self.norm_2(x)

class PredFormer(nn.Module):
    def __init__(self, image_size = [16, 32], patch_size=4, dim=256, depth=6, heads=8, channels = 256, **args):
        super().__init__()
        self.patch_size = patch_size
        self.patchnum_h = image_size[0] // patch_size
        self.patchnum_w = image_size[1] // patch_size
        patch_height, patch_width = pair(patch_size)

        patch_dim = channels * patch_height * patch_width
        # dim = patch_dim

        self.to_patch_embedding = nn.Sequential(
            # Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.trans_N = []
        self.trans_T = []
        self.trans_S = []
        mlp_dim = dim * 4
        dim_head = dim // heads
        self.depth = depth
        for i in range(depth):
            self.trans_N.append(Transformer(dim, heads, dim_head, mlp_dim))
            self.trans_T.append(Transformer(dim, heads, dim_head, mlp_dim))
            self.trans_S.append(Transformer(dim, heads, dim_head, mlp_dim))
        
        self.trans_N = nn.ModuleList(self.trans_N)
        self.trans_T = nn.ModuleList(self.trans_T)
        self.trans_S = nn.ModuleList(self.trans_S)

        self.linear_head = nn.Linear(dim, patch_dim)

    def forward(self, x):
        device = x.device
        B, N, T, C_O, H, W = x.shape
        x = patch_split(x,patch_size=self.patch_size)
        B, N, T, M, C = x.shape
        x = x.view(B, N * T * M, C)
        x = self.to_patch_embedding(x)
        x = x.view(B, N, T, M, -1)

        B, N, T, M, C = x.shape
        pos_time_emb = posemb_sincos_1d(T, C).to(device).view(1, 1, T, 1, C)
        x = x + pos_time_emb

        for i in range(self.depth):
            x = x.permute(0, 3, 1, 2, 4) # 处理变量维度
            x = x.reshape(-1,N*T,C)
            x = self.trans_N[i](x).reshape(B,M,N,T,C)

            x = x.permute(0, 2, 1, 3, 4) # 处理 T
            x = x.reshape(-1,T,C)
            x = self.trans_T[i](x).reshape(B,N,M,T,C)

            x = x.permute(0, 1, 3, 2, 4) # 处理 T
            x = x.reshape(-1,M,C)
            x = self.trans_S[i](x).reshape(B,N,T,M,C)

        x = self.linear_head(x)
        x = x.reshape(B,N,T,self.patchnum_h,self.patchnum_w,C_O,self.patch_size,self.patch_size)
        x = x.permute(0, 1, 2, 5, 3, 6, 4, 7)
        x = x.reshape(B,N,T,C_O,self.patchnum_h*self.patch_size,self.patchnum_w*self.patch_size)


        return x
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.randn((16,4,16,256,16,32)).to(device) # b t c h w
    model = PredFormer().to(device)
    y = model(data)
    summary(model, (data.shape), device=device)



