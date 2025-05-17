import torch
import torch.nn as nn

class ConvSC3D(nn.Module):
    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=(3, 3, 3),
                 stride=(2, 2, 2),act_inplace=True):
        super(ConvSC3D, self).__init__()

        self.conv =  nn.Conv3d(
                    in_channels=C_in,     
                    out_channels=C_out,     
                    kernel_size=kernel_size, 
                    stride=stride,    
                    padding=1            
                )
        self.norm = nn.GroupNorm(2,C_out)
        self.act = nn.SiLU(inplace=act_inplace)
        
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        return y
    
class UNConvSC3D(nn.Module):
    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=(3, 3, 3),
                 stride=(2, 2, 2),output_padding=(1,1,1),act_inplace=True):
        super(UNConvSC3D, self).__init__()

        self.conv =  nn.ConvTranspose3d(
                    in_channels=C_in,     
                    out_channels=C_out,     
                    kernel_size=kernel_size, 
                    stride=stride,    
                    padding=1,
                    output_padding=output_padding       
                )
        self.norm = nn.GroupNorm(2,C_out)
        self.act = nn.SiLU(inplace=act_inplace)
        
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        return y