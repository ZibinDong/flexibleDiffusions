import math
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn


def choose_conv(tensor_dim: int):
    ''' Choose the appropriate convolutional layer for the tensor dimension '''
    if tensor_dim == 1:
        return nn.Conv1d
    elif tensor_dim == 2:
        return nn.Conv2d
    else:
        raise NotImplementedError(f"tensor_dim={tensor_dim} is not implemented")
    
def choose_conv_transpose(tensor_dim: int):
    if tensor_dim == 1:
        return nn.ConvTranspose1d
    elif tensor_dim == 2:
        return nn.ConvTranspose2d
    else:
        raise NotImplementedError(f"tensor_dim={tensor_dim} is not implemented")
    
def choose_norm(tensor_dim: int, use_norm: bool, **kwargs):
    if not use_norm: return nn.Identity()
    if tensor_dim == 1:
        return nn.BatchNorm1d(**kwargs)
    elif tensor_dim == 2:
        return nn.GroupNorm(**kwargs)
    else:
        raise NotImplementedError(f"tensor_dim={tensor_dim} is not implemented")

def at_least_ndim(x: torch.Tensor, ndim: int):
    ''' Add dimensions to the end of tensor until it has at least ndim dimensions '''
    if x.ndim >= ndim:
        return x
    return x[(...,) + (None,) * (ndim - x.ndim)]

class TimeEmbedding(nn.Module):
    ''' Time Embedding Module: (batch_size, ) -> (batch_size, n_channels) '''
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(n_channels // 4, n_channels),
            nn.Mish(),
            nn.Linear(n_channels, n_channels),
        )
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class UNetResidualBlock(nn.Module):
    ''' 
    UNet Residual Block: 
    (batch, in_channels, ...) -> (batch, out_channels, ...) 
    '''
    def __init__(self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        tensor_dim: int = 2,
        
        use_norm: bool = True,
        n_groups: int = 32,
    ):
        super().__init__()

        conv = choose_conv(tensor_dim)
        if tensor_dim == 1:
            norm1 = choose_norm(tensor_dim, use_norm, num_features=in_channels)
            norm2 = choose_norm(tensor_dim, use_norm, num_features=out_channels)
        elif tensor_dim == 2:
            norm1 = choose_norm(tensor_dim, use_norm, num_groups=n_groups, num_channels=in_channels)
            norm2 = choose_norm(tensor_dim, use_norm, num_groups=n_groups, num_channels=out_channels)
        else:
            raise NotImplementedError(f"tensor_dim={tensor_dim} is not implemented")

        self.conv1 = nn.Sequential(
            norm1,
            nn.Mish(),
            conv(in_channels, out_channels, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            norm2,
            nn.Mish(),
            conv(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = conv(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        self.time_emb = nn.Sequential(
            nn.Linear(time_channels, out_channels),
            nn.Mish(),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        Input:
            - x: (batch, in_channels, ...)
            - t: (batch, time_channels)
        
        Output:
            - y: (batch, out_channels, ...)
        '''
        h = self.conv1(x)
        h += at_least_ndim(self.time_emb(t), h.ndim)
        h = self.conv2(h)
        return h + self.shortcut(x)

class UNetAttentionBlock(nn.Module):
    ''' 
    UNet Attention Block: 
    (batch, in_channels, ...) -> (batch, out_channels, ...) 
    '''
    def __init__(self, 
        in_channels: int, 
        n_heads: int = 1, 
        d_k: int = None, 
    ):
        super().__init__()
        
        d_k = in_channels if d_k is None else d_k

        self.projection = nn.Linear(in_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, in_channels)
        self.scale = d_k ** -0.5
        
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.d_k = d_k
        
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        '''
        Input:
         - x: (batch, in_channels, ...)
         - t: (batch, time_channels)
         
        Output:
         - y: (batch, in_channels, ...)
        '''
        _ = t
        batch_size = x.shape[0]
        tensor_shape = x.shape[2:]
        
        # (b, c, ...) -> (b, d, c)
        x = torch.reshape(x, (batch_size, self.in_channels, -1)).permute(0, 2, 1)
        dim = x.shape[1]

        # (b, d, c) -> (b, h, d, 3 * d_k), q/k/v: (b, h, d, d_k)
        q, k, v = torch.chunk(
            self.projection(x).reshape(batch_size, dim, self.n_heads, -1).permute(0, 2, 1, 3),
            3, -1
        )
        
        # (b, h, d, d)
        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # (b, h, d, d) @ (b, h, d, d_k) -> (b, h, d, d_k)
        res = attn @ v
        
        # (b, h, d, d_k) -> (b, d, c)
        res = self.output(einops.rearrange(res, 'b h d k -> b d (h k)'))
        res += x
        
        res = torch.reshape(res.permute(0, 2, 1), (batch_size, self.in_channels, *tensor_shape))
        return res