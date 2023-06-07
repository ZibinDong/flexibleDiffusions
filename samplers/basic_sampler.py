import json
from typing import Iterable, Optional

import torch

from classifiers.basic_classifier import BasicClassifier
from diffusion.denoise_diffusion import DenoiseDiffusion


class BasicSampler():
    def __init__(self,
        diffusion: DenoiseDiffusion,
        classifier: Optional[BasicClassifier] = None,
        cg_strength: float = 1.0,
        cf_strength: float = 1.0,
    ):
        self.diffusion = diffusion
        self.T = diffusion.T
        self.classifier = classifier
        self.cg_strength = cg_strength
        self.cf_strength = cf_strength
        
        self.betas = diffusion.betas
        self.alphas = diffusion.alphas
        self.alphas_bar = diffusion.alphas_bar
        self.alphas_bar_prev = diffusion.alphas_bar_prev
        self.stds = ((1-self.alphas_bar_prev)/(1-self.alphas_bar)*self.betas).sqrt()
        
        self.use_cg_guidance = classifier is not None
        self.use_cf_guidance = diffusion.eps_model.use_cond
        
    def __repr__(self):
        return json.dumps({
            "name": "Basic sampler",
            "use_classifier_guided_guidance": self.use_cg_guidance,
            "use_classifier_free_guidance": self.use_cf_guidance,
            "classifier_guided_strength": self.cg_strength,
            "classifier_free_strength": self.cf_strength,
        })
        
    def predict_eps(self,
        xt: torch.Tensor, t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ):
        if self.use_cf_guidance:
            assert cond is not None
            cond_eps = self.diffusion.eps_model(xt, t, cond)
            uncond_eps = self.diffusion.eps_model(xt, t, torch.zeros_like(cond))
            return cond_eps + (cond_eps - uncond_eps) * self.cf_strength
        else:
            return self.diffusion.eps_model(xt, t)
        
    def calculate_gradient_guidance(self,
        xt: torch.Tensor, t: torch.Tensor,
        y: torch.Tensor,
    ):
        if self.use_cg_guidance:
            logp, grad = self.classifier.gradients(xt, t, y)
            guidance = grad * self.cg_strength
            return logp, guidance
        else:
            return 0., 0.
        
    def apply_inpainting_condition(self,
        xt: torch.Tensor,
        inpainting_mask: Optional[torch.Tensor] = None,
        inpainting_value: Optional[torch.Tensor] = None,
    ):
        if inpainting_mask is None or inpainting_value is None:
            return xt
        else:
            return xt * (1 - inpainting_mask) + inpainting_mask * inpainting_value

    def p_xtm1_xt(self, 
        xt: torch.Tensor, t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self, 
        n_samples: int, t_seq: Iterable[int],
        y: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
        inpainting_mask: Optional[torch.Tensor] = None, inpainting_value: Optional[torch.Tensor] = None,
        save_denoise_history: bool = False,
    ):
        if save_denoise_history: denoise_history = []
        
        xt = torch.randn((n_samples, *self.diffusion.x_shape), device=self.diffusion.device)
        xt = self.apply_inpainting_condition(xt, inpainting_mask, inpainting_value)
        
        for i in reversed(t_seq):
            
            mean, std, logp = self.p_xtm1_xt(xt, i, y, cond)
            
            if i != 0:
                xt = mean + std * torch.randn_like(mean)
            else:
                xt = mean
                
            xt = self.apply_inpainting_condition(xt, inpainting_mask, inpainting_value)
            
            if save_denoise_history: denoise_history.append(torch.clone(xt))
        
        return xt, logp
