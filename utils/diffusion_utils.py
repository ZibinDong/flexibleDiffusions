import numpy as np
import torch
from .network_utils import at_least_ndim

def gather(consts: torch.Tensor, t: torch.Tensor, ndim: int = 3):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return at_least_ndim(c, ndim)

def beta_schedual_linear(begin=1e-4, end=2e-2, T=1000, device="cpu"):
    return torch.from_numpy(np.linspace(begin, end, T)).float().to(device)

def beta_schedual_cosine(s=0.008, T=1000, device='cpu'):
    f = lambda t: np.cos((s+t/T)/(s+1)*np.pi/2.) ** 2
    t = np.arange(T)+1
    alpha_bar = f(t) / f(0)
    alpha_bar_prev = np.concatenate([[1.], alpha_bar[:-1]])
    return torch.from_numpy(1 - alpha_bar/alpha_bar_prev).float().clip(0.,0.999).to(device)


if __name__ == "__main__":
    
    import os

    import matplotlib.pyplot as plt
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    f = plt.figure()
    
    beta1 = beta_schedual_linear()
    beta2 = beta_schedual_cosine()
    
    alpha1 = 1 - beta1
    alpha2 = 1 - beta2
    
    alpha_bar1 = torch.cumprod(alpha1, dim=0)
    alpha_bar2 = torch.cumprod(alpha2, dim=0)
    
    plt.plot(alpha_bar1)
    plt.plot(alpha_bar2)
    
    plt.show()
    
    
    