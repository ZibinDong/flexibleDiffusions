import json
from typing import Iterable, List, Optional, Union

import torch

import utils
from classifiers.basic_classifier import BasicClassifier
from diffusion.denoise_diffusion import DenoiseDiffusion

from .basic_sampler import BasicSampler


class DDIMSampler(BasicSampler):
    def __init__(self,
        diffusion: DenoiseDiffusion,
        ddim_eta: float,
        ddim_sample_steps: int = 20,
        ddim_discretize: str = 'uniform',
        classifier: Optional[BasicClassifier] = None,
        cg_strength: float = 1.0,
        cf_strength: float = 1.0,
        
    ):
        super().__init__(diffusion, classifier, cg_strength, cf_strength)
        
        if ddim_discretize == 'uniform':
            self.sample_t_array = utils.sample_t_array_uniform(self.T, ddim_sample_steps)
        elif ddim_discretize == 'quad':
            self.sample_t_array = utils.sample_t_array_quad(self.T, ddim_sample_steps, coeff=0.8)
        else:
            raise ValueError(f'Unknown ddim_discretize: {ddim_discretize}')
        
        self.ddim_eta = ddim_eta
        self.ddim_sample_steps = ddim_sample_steps
        self.ddim_discretize = ddim_discretize
        
        self.sigma = ddim_eta * ((1-self.alphas_bar_prev)/(1-self.alphas_bar)).sqrt() * (1-self.alphas_bar/self.alphas_bar_prev).sqrt()
        
        self.coeff1 = self.alphas_bar_prev.sqrt()
        self.coeff2 = (1 - self.alphas_bar_prev - self.sigma**2).sqrt()
        self.coeff3 = self.sigma
        self.guidance_coeff = (1 - self.alphas_bar).sqrt()
        
    def __repr__(self):
        param = json.loads(super().__repr__())
        param["name"] = "DDIM sampler"
        param["ddim_eta"] = self.ddim_eta
        param["ddim_sample_steps"] = self.ddim_sample_steps
        param["ddim_discretize"] = self.ddim_discretize
        return json.dumps(param)
        
    def p_xtm1_xt(self, 
        xt: torch.Tensor, t: int,
        y: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_t = torch.ones((xt.shape[0],), device=xt.device, dtype=torch.long) * t
        
        eps = self.predict_eps(xt, batch_t, cond)
        logp, classifier_guidance = self.calculate_gradient_guidance(xt, batch_t, y)
        eps -= self.guidance_coeff[t] * classifier_guidance
        
        mean = (
            self.coeff1[t] * (xt - self.guidance_coeff[t] * eps)/self.alphas_bar[t].sqrt() +
            self.coeff2[t] * eps
        )
        std = self.coeff3[t]
        
        return mean, std, logp
    
    def sample(self, 
        n_samples: int,
        y: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
        inpainting_mask: Optional[torch.Tensor] = None, inpainting_value: Optional[torch.Tensor] = None,
        save_denoise_history: bool = False,
    ):
        return super().sample(
            n_samples, self.sample_t_array,
            y, cond, inpainting_mask, inpainting_value,
            save_denoise_history,
        )