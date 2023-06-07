from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

import utils
from models.unet import DownBlock, Downsample, MiddleBlock

from .basic_classifier import BasicClassifier


class HalfUNetClassifier(BasicClassifier):
    ''' Half-UNet implementation for basic classifier. '''
    def __init__(self,
        data_channels: int = 3,
        n_channels: int = 64,
        time_channels: int = 64,
        tensor_dim: int = 2,
        
        n_groups: int = 4,
        ch_mults: Union[Tuple[int,...], List[int]] = (1, 2, 2, 4),
        use_attn: Union[Tuple[bool,...], List[int]] = (False, False, True, True),
        use_norm: Union[Tuple[bool,...], List[int]] = (True, True, True, True),
        n_heads: int = 1,
        n_blocks: int = 2,
        
        y_type: str = 'label',
        n_categories: Optional[int] = 10,
    ):
        super().__init__(y_type, n_categories)
        
        n_resolutions = len(ch_mults)
        self.n_channels = n_channels
        self.ch_mults = ch_mults
        self.use_attn = use_attn
        self.use_norm = use_norm
        
        conv = utils.choose_conv(tensor_dim)
        
        self.projection = conv(data_channels, n_channels, 3, padding=1)
        self.time_emb = utils.TimeEmbedding(time_channels)
        
        down = []
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

        self.down = nn.ModuleList(down)
        self.flatten = nn.Flatten()
        self.final = None
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        Input:
            - x: (batch, data_channels, ...)
            - t: (batch,)
        
        Output:
            - y: (batch, 1) for y_type='logp' or (batch, n_categories) for y_type='label'
        '''
        t = self.time_emb(t)
        x = self.projection(x)
        for m in self.down:
            x = m(x, t)
        x = self.flatten(x)

        if self.final is None:
            dim = x.shape[-1]
            layers = [
                nn.Linear(dim, dim // 2), nn.Mish(),
                nn.Linear(dim // 2, dim // 2), nn.Mish(),
            ]
            if self.y_type == 'logp':
                layers.append(nn.Linear(dim // 2, 1))
            elif self.y_type == 'label':
                layers.append(nn.Linear(dim // 2, self.n_categories))
            else:
                raise ValueError(f'Unknown y_type: {self.y_type}')
            self.final = nn.Sequential(*layers).to(x.device)
            
        return self.final(x)
