import json
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

import utils

from .eps_model import EpsModel

# hidden_size = 384 | 768 | 1024 | 1152
# depth =       12  | 24  | 28
# patch_size =  2   | 4   | 8
# n_heads =     6   | 12  | 16  (hidden_size can be divided by n_heads)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, n_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, n_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6)
        )
        
    def forward(self, x, c):
        '''
        Input:
            - x: (batch_size, seq_len, hidden_size)
            - c: (batch_size, hidden_size)
        
        Output:
            - y: (batch_size, seq_len. hidden_size)
        '''
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    
class Finallayer1d(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
        
class FinalLayer2d(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class DiT(EpsModel):
    def __init__(self,
        hidden_size: int, 
        cond_dim: Optional[int] = None,
        n_heads: int = 16,
        mlp_ratio: float = 4.,
        depth: int = 28,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.depth = depth
        
        self.x_embedder: nn.Module
        self.t_embedder = utils.TimeEmbedding(hidden_size)
        self.use_cond = (cond_dim is not None)
        if self.use_cond:
            self.cond_embedder = nn.Sequential(
                nn.Linear(cond_dim, hidden_size), nn.Mish(),
                nn.Linear(hidden_size, hidden_size)
            )
        # Will use fixed sin-cos embedding:
        self.pos_embed: nn.Parameter

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer: nn.Module
        
    def __repr__(self):
        cfg = {
            "name": "DiT",
            "hidden_size": self.hidden_size,
            "n_heads": self.n_heads,
            "depth": self.depth,
        }
        return json.dumps(cfg)
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        # pos_embed is required to be initialized
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        Input:
            - x:    (batch, data_channels, ...)
            - t:    (batch,)
            - cond: (batch, cond_dim)
        
        Output:
            - y:    (batch, data_channels, ...)
        """
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        
        if self.use_cond:
            t += self.cond_embedder(cond)
            
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        return x
    
class DiT1d(DiT):
    def __init__(
        self,
        
        input_size: int = 32,
        seq_len: int = 10,
        
        hidden_size: int = 1152, 
        cond_dim: Optional[int] = None,
        n_heads: int = 16,
        mlp_ratio: float = 4.,
        depth: int = 28,
    ):
        super().__init__(hidden_size, cond_dim, n_heads, mlp_ratio, depth)
        self.input_size = input_size
        self.seq_len = seq_len
        
        self.x_embedder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)
        
        self.final_layer = Finallayer1d(hidden_size, input_size)
        self.initialize_weights()
        
    def __repr__(self):
        cfg = json.loads(super().__repr__())
        cfg["name"] = "DiT1d"
        cfg["seq_len"] = self.seq_len
        cfg["input_size"] = self.input_size
        cfg["n_params"] = utils.abbreviate_number(utils.count_parameters(self))
        return json.dumps(cfg)
    
    def initialize_weights(self):
        super().initialize_weights()
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, np.arange(self.seq_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
class DiT2d(DiT):
    def __init__(
        self,
        
        input_size: int = 32,
        patch_size: int = 2,
        data_channels: int = 3,
        
        hidden_size: int = 1152, 
        cond_dim: Optional[int] = None,
        n_heads: int = 16,
        mlp_ratio: float = 4.,
        depth: int = 28,
    ):
        super().__init__(hidden_size, cond_dim, n_heads, mlp_ratio, depth)
        self.data_channels = data_channels
        self.patch_size = patch_size
        
        self.x_embedder = PatchEmbed(
            input_size, patch_size, data_channels, hidden_size
        ) # (b, c, h, w) -> (b, (h/p)**2, d) (h==w)
            
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.final_layer = FinalLayer2d(hidden_size, patch_size, data_channels)
        self.initialize_weights()
        
    def __repr__(self):
        cfg = json.loads(super().__repr__())
        cfg["name"] = "DiT2d"
        cfg["data_channels"] = self.data_channels
        cfg["patch_size"] = self.patch_size
        cfg["n_params"] = utils.abbreviate_number(utils.count_parameters(self))
        return json.dumps(cfg)

    def initialize_weights(self):
        super().initialize_weights()
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def unpatchify(self, x: torch.Tensor):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.data_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None):
        return self.unpatchify(super().forward(x, t, cond))

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb