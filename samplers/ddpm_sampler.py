from typing import Iterable, List, Optional, Union

import torch

from classifiers.basic_classifier import BasicClassifier
from diffusion.denoise_diffusion import DenoiseDiffusion

from .basic_sampler import BasicSampler


class DDPMSampler(BasicSampler):
    def __init__(self,
        diffusion: DenoiseDiffusion,
        
        classifier: Optional[BasicClassifier] = None,
        cg_strength: float = 1.0,
        cf_strength: float = 1.0,
        
    ):
        super().__init__(diffusion, classifier, cg_strength, cf_strength)
        
        self.xt_coeffs = 1 / self.alphas.sqrt()
        self.eps_coeffs = self.betas / (1 - self.alphas_bar).sqrt() * self.xt_coeffs
        
    def p_xtm1_xt(self, 
        xt: torch.Tensor, t: int,
        y: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_t = torch.ones((xt.shape[0],), device=xt.device, dtype=torch.long) * t
        
        eps = self.predict_eps(xt, batch_t, cond)
        logp, classifier_guidance = self.calculate_gradient_guidance(xt, batch_t, y)
        
        mean = self.xt_coeffs[t] * xt + self.eps_coeffs[t] * eps
        std = self.stds[t]
        
        mean += (std**2) * classifier_guidance
        
        return mean, std, logp
    
    def sample(self, 
        n_samples: int,
        y: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
        inpainting_mask: Optional[torch.Tensor] = None, inpainting_value: Optional[torch.Tensor] = None,
        save_denoise_history: bool = False,
    ):
        return super().sample(
            n_samples, range(self.T),
            y, cond, inpainting_mask, inpainting_value,
            save_denoise_history,
        )