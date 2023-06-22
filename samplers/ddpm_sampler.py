import json
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
        
    def __repr__(self):
        param = json.loads(super().__repr__())
        param["name"] = "DDPM sampler"
        return json.dumps(param)
        
    def p_xtm1_xt(self, 
        xt: torch.Tensor, t: int,
        y: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_t = torch.ones((xt.shape[0],), device=xt.device, dtype=torch.long) * t
        
        eps = self.predict_eps(xt, batch_t, cond)
        logp, classifier_guidance = self.calculate_gradient_guidance(xt, batch_t, y)
        
        mean = self.xt_coeffs[t] * xt - self.eps_coeffs[t] * eps
        std = self.stds[t]
        
        mean += (std**2) * classifier_guidance
        
        return mean, std, logp
    
    def sample(self, 
        n_samples: int,
        y: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
        inpainting_mask: Optional[torch.Tensor] = None, inpainting_value: Optional[torch.Tensor] = None,
        noisy_inpainting: bool = False,
        save_denoise_history: bool = False,
    ):
        ''' Sample from the diffusion model.
        
        Args:
            - n_samples: number of samples to generate
            - y: torch.Tensor of shape (n_samples, 1). If provided, the classifier will be used to guide the sampling process.
            - cond: torch.Tensor of shape (n_samples, cond_dim). If provided, the conditional diffusion model will be used.
            - inpainting_mask: torch.Tensor of shape (1, *x_shape). If provided, the inpainting mask will be applied to the samples.
                It is supposed to be a binary mask with 1s indicating the masked out regions.
            - inpainting_value: torch.Tensor of shape (1, *x_shape). If provided, the inpainting mask will be applied to the samples.
                It is supposed to be a tensor of the same shape as the samples.
            - noisy_inpainting: If True, the inpainting will be done in a noisy manner according to the forward diffusion process. 
                If False, the masked out regions will be directly filled with inpainting_value.
            - save_denoise_history: If True, the denoised samples at each timestep will be saved and returned.

        Returns:
            - xt: torch.Tensor of shape (n_samples, *x_shape). The generated samples.
            - logp: torch.Tensor of shape (n_samples,). The log-probability of the generated samples. Only returned if y is provided.
            - denoise_history: List[torch.Tensor] of length len(t_seq). The denoised samples at each timestep. Only returned if save_denoise_history is True.

        '''
        return super().sample(
            n_samples, range(self.ddim_sample_steps),
            y, cond, inpainting_mask, inpainting_value, noisy_inpainting,
            save_denoise_history,
        )