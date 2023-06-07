from typing import Optional

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

import models
import utils.diffusion_utils as diffusion_utils


class DenoiseDiffusion():
    def __init__(self,
        eps_model: models.UNet,
        uncond_prob: Optional[float] = 0.25,
        
        T: int = 1000,
        device: str = "cuda",
        loss_type: str = "l2",
        beta_schedule: str = "cosine",
        **kwargs,
    ):
        self.eps_model = eps_model.to(device)
        self.T = T
        self.device = device
        self.loss_type = loss_type
        self.x_shape = None
        
        # uncond mask
        assert (uncond_prob is not None) == eps_model.use_cond
        if eps_model.use_cond:
            self.uncond_mask_dist = dist.Bernoulli(probs=1-uncond_prob)
        
        if beta_schedule == "linear":
            self.betas = diffusion_utils.beta_schedual_linear(
                **kwargs, T = T, device=device)
        elif beta_schedule == "cosine":
            self.betas = diffusion_utils.beta_schedual_cosine(
                **kwargs, T = T, device=device)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.ones(1, device=device), self.alphas_bar[:-1]])
        
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ''' forward process q(x_t|x_0) '''
        n_dim = len(x0.shape)
        alpha_bar = diffusion_utils.gather(self.alphas_bar, t, n_dim)
        
        mean = alpha_bar.sqrt() * x0
        std = (1 - alpha_bar).sqrt()
        
        return mean, std
        
    def loss(self, 
        x0: torch.Tensor, 
        cond: Optional[torch.Tensor] = None):
        
        if self.x_shape is None: self.x_shape = x0.shape[1:]
        batch_size = x0.shape[0]
        t = torch.randint(self.T, (batch_size,), device=self.device)
        
        mean, std = self.q_xt_x0(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + std * eps
        
        # uncond mask
        if self.eps_model.use_cond:
            uncond_mask = self.uncond_mask_dist.sample((batch_size, 1))
            cond *= uncond_mask.to(self.device)
        pred_eps = self.eps_model(xt, t, cond)
        
        if self.loss_type == "l1":
            loss = F.l1_loss(pred_eps, eps)
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred_eps, eps)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss
