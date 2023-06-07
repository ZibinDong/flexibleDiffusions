from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import utils

# ============================================ Components of UNet ==============================================

class DownBlock(nn.Module):
    '''
    This combines `ResidualBlock` and `AttentionBlock`. 
    These are used in the first half of U-Net at each resolution.
    '''
    def __init__(self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        
        n_groups: int = 32,
        tensor_dim: int = 2,
        
        use_attn: bool = True,
        use_norm: bool = True,
        n_heads: int = 1,
        d_k: Optional[int] = None,
    ):
        super().__init__()
        self.res = utils.UNetResidualBlock(
            in_channels, out_channels, time_channels,
            tensor_dim, use_norm, n_groups)
        self.attn = utils.UNetAttentionBlock(
            out_channels, n_heads, d_k
        ) if use_attn else nn.Identity()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x, t))
    
class UpBlock(nn.Module):
    '''
    This combines `ResidualBlock` and `AttentionBlock`. 
    These are used in the second half of U-Net at each resolution.
    '''
    def __init__(self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        
        n_groups: int = 32,
        tensor_dim: int = 2,
        
        use_attn: bool = True,
        use_norm: bool = True,
        n_heads: int = 1,
        d_k: Optional[int] = None,
    ):
        super().__init__()
        self.res = utils.UNetResidualBlock(
            in_channels+out_channels, out_channels, time_channels,
            tensor_dim, use_norm, n_groups)
        self.attn = utils.UNetAttentionBlock(
            out_channels, n_heads, d_k
        ) if use_attn else nn.Identity()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x, t))
    
class MiddleBlock(nn.Module):
    '''
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    '''
    def __init__(self,
        n_channels: int,
        time_channels: int,
        
        n_groups: int = 32,
        tensor_dim: int = 2,
        
        use_attn: bool = True,
        use_norm: bool = True,
        n_heads: int = 1,
        d_k: Optional[int] = None,
    ):
        super().__init__()
        self.res1 = utils.UNetResidualBlock(
            n_channels, n_channels, time_channels,
            tensor_dim, use_norm, n_groups)
        self.attn = utils.UNetAttentionBlock(
            n_channels, n_heads, d_k
        ) if use_attn else nn.Identity()
        self.res2 = utils.UNetResidualBlock(
            n_channels, n_channels, time_channels,
            tensor_dim, use_norm, n_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.res2(self.attn(self.res1(x, t)), t)
    
class Upsample(nn.Module):
    '''
    Scale up the feature map by 2x
    '''
    def __init__(self, n_channels: int, tensor_dim: int):
        super().__init__()
        self.conv = utils.choose_conv_transpose(tensor_dim)\
            (n_channels, n_channels, 4, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)
    
class Downsample(nn.Module):
    '''
    Scale down the feature map by 0.5x
    '''
    def __init__(self, n_channels: int, tensor_dim: int):
        super().__init__()
        self.conv = utils.choose_conv(tensor_dim)\
            (n_channels, n_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)
    
    
# ============================================ UNet Model ==============================================

class UNet(nn.Module):
    def __init__(self,
        data_channels: int = 3,
        
        n_channels: int = 64,
        time_channels: int = 64,

        cond_dim: Optional[int] = None,
        tensor_dim: int = 2,
        
        n_groups: int = 4,
        ch_mults: Union[Tuple[int,...], List[int]] = (1, 2, 2, 4),
        use_attn: Union[Tuple[bool,...], List[int]] = (False, False, True, True),
        use_norm: Union[Tuple[bool,...], List[int]] = (True, True, True, True),
        n_heads: int = 1,
        n_blocks: int = 2,
    ):
        super().__init__()
        n_resolutions = len(ch_mults)
        self.n_channels = n_channels
        self.ch_mults = ch_mults
        self.use_attn = use_attn
        self.use_norm = use_norm
        
        conv = utils.choose_conv(tensor_dim)
        
        self.projection = conv(data_channels, n_channels, 3, padding=1)
        self.time_emb = utils.TimeEmbedding(time_channels)
        self.use_cond = (cond_dim is not None)
        if self.use_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, time_channels), nn.Mish(),
                nn.Linear(time_channels, time_channels * 4), nn.Mish(),
                nn.Linear(time_channels * 4, time_channels),
            )
            time_channels *= 2
        
        up, down = [], []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = int(in_channels * ch_mults[i])
            for _ in range(n_blocks):
                down.append(DownBlock(
                    in_channels, out_channels, time_channels,
                    n_groups, tensor_dim, use_attn[i], use_norm[i], n_heads))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, tensor_dim))
        
        mid_channels = out_channels
        in_channels = out_channels
        
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(
                    in_channels, out_channels, time_channels,
                    n_groups, tensor_dim, use_attn[i], use_norm[i], n_heads))
            out_channels = int(out_channels // ch_mults[i])
            up.append(UpBlock(in_channels, out_channels, time_channels, 
                    n_groups, tensor_dim, use_attn[i], use_norm[i], n_heads))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, tensor_dim))
        
        self.down = nn.ModuleList(down)
        self.mid = MiddleBlock(
            mid_channels, time_channels, n_groups, tensor_dim, True, True, n_heads)
        self.up = nn.ModuleList(up)
        
        # self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.Mish()
        self.final = conv(out_channels, data_channels, 3, padding=1)
        
    def __repr__(self):
        n_params = utils.count_parameters(self)
        info = {
            "type": "UNet",
            "n_channels": self.n_channels,
            "ch_mults": self.ch_mults,
            "use_attn": self.use_attn,
            "use_norm": self.use_norm,
            "n_params": utils.abbreviate_number(n_params),
        }
        return json.dumps(info)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Input:
            - x:    (batch, data_channels, ...)
            - t:    (batch,)
            - cond: (batch, cond_dim)
        
        Output:
            - y:    (batch, data_channels, ...)
        '''
        t = self.time_emb(t)
        if cond is not None and self.use_cond:
            t = torch.cat([t, self.cond_proj(cond)], dim=-1)
        x = self.projection(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        
        x = self.mid(x, t)
        
        for m in self.up:
            x = torch.cat([x, h.pop()], dim=1) if not isinstance(m, Upsample) else x
            x = m(x, t)
        
        return self.final(self.act(x)) 