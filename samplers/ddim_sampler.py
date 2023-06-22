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
        
        self.ddim_alphas = self.alphas_bar[self.sample_t_array]
        self.ddim_alphas_prev = torch.cat([self.ddim_alphas[:1], self.ddim_alphas[:-1]])
        self.sigma = ddim_eta * ((1-self.ddim_alphas_prev)/(1-self.ddim_alphas)).sqrt() * (1-self.ddim_alphas/self.ddim_alphas_prev).sqrt()
        
        self.coeff1 = self.ddim_alphas_prev.sqrt()
        self.coeff2 = (1 - self.ddim_alphas_prev - self.sigma**2).sqrt()
        self.coeff3 = self.sigma
        self.guidance_coeff = (1 - self.ddim_alphas).sqrt()
        
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
        
        batch_t = torch.ones((xt.shape[0],), device=xt.device, dtype=torch.long) * self.sample_t_array[t]
        
        eps = self.predict_eps(xt, batch_t, cond)
        logp, classifier_guidance = self.calculate_gradient_guidance(xt, batch_t, y)
        eps -= self.guidance_coeff[t] * classifier_guidance
        
        mean = (
            self.coeff1[t] * (xt - self.guidance_coeff[t] * eps)/self.ddim_alphas[t].sqrt() +
            self.coeff2[t] * eps
        )
        std = self.coeff3[t]
        
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