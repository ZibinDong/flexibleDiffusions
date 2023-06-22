import torch
import torch.nn as nn


class EpsModel(nn.Module):
    ''' epsilon model for diffusion: y = f(x, t, cond) such that y.shape == x.shape '''
    def __init__(self,):
        super().__init__()
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError