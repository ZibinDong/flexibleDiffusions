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
        t: int = 0,
        inpainting_mask: Optional[torch.Tensor] = None,
        inpainting_value: Optional[torch.Tensor] = None,
        noisy_inpainting: bool = False,
    ):
        if hasattr(self, "sample_t_array"): t = self.sample_t_array[t]
        if inpainting_mask is None or inpainting_value is None:
            return xt
        elif t == 0 or not noisy_inpainting:
            return xt * (1 - inpainting_mask) + inpainting_mask * inpainting_value
        else:
            return xt * (1 - inpainting_mask) + inpainting_mask * (
                self.alphas_bar[t].sqrt() * inpainting_value +
                (1-self.alphas_bar[t]).sqrt() * torch.randn_like(xt)
            )

    def p_xtm1_xt(self, 
        xt: torch.Tensor, t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @torch.no_grad()
    def sample(self, 
        n_samples: int, t_seq: Iterable[int],
        y: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
        inpainting_mask: Optional[torch.Tensor] = None, inpainting_value: Optional[torch.Tensor] = None,
        noisy_inpainting: bool = False,
        save_denoise_history: bool = False,
    ):
        ''' Sample from the diffusion model.
        
        Args:
            - n_samples: number of samples to generate
            - t_seq: sequence of timesteps to sample at
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
        self.diffusion.eps_model.eval()
        
        if save_denoise_history: denoise_history = []
        
        xt = torch.randn((n_samples, *self.diffusion.x_shape), device=self.diffusion.device)
        xt = self.apply_inpainting_condition(xt, t_seq[-1], inpainting_mask, inpainting_value, noisy_inpainting)
        
        for i in reversed(t_seq):
            
            mean, std, logp = self.p_xtm1_xt(xt, i, y, cond)
            
            if i != 0:
                xt = mean + std * torch.randn_like(mean)
            else:
                xt = mean
                
            xt = self.apply_inpainting_condition(xt, i, inpainting_mask, inpainting_value, noisy_inpainting)
            
            if save_denoise_history: denoise_history.append(torch.clone(xt))
            
        self.diffusion.eps_model.train()
        return xt, logp, denoise_history if save_denoise_history else None

    # TODO: Trajectory planning for decision making, which may preserve history observations.
    # @torch.no_grad()
    # def plan_trajectories(self,
    #     obs: torch.Tensor, n_samples: int, t_seq: Iterable[int],
    #     cond: torch.Tensor,
    #     inpainting_mask: Optional[torch.Tensor] = None, inpainting_value: Optional[torch.Tensor] = None, ptr: int = 0,
    #     max_history_len: int = 20,
    # ):
    #     '''
    #     Input:
    #         - obs: torch.Tensor of shape (1, observation_dim)
    #         - n_samples: number of samples to generate
    #         - cond: torch.Tensor of shape (1, cond_dim)
    #         - inpainting_mask: torch.Tensor of shape (1, seq_len, observation_dim)
    #         - inpainting_value: torch.Tensor of shape (1, seq_len, observation_dim)
        
    #     Output:
    #         - trajs: torch.Tensor of shape (n_samples, seq_len, observation_dim)
    #         - inpainting_mask: torch.Tensor of shape (1, seq_len, observation_dim)
    #         - inpainting_value: torch.Tensor of shape (1, seq_len, observation_dim)
    #     '''
    #     cond = cond.repeat(n_samples, 1)
        
    #     if inpainting_mask is None or inpainting_value is None:
    #         inpainting_mask = torch.zeros((1, *self.diffusion.x_shape), device=self.diffusion.device)
    #         inpainting_value = torch.zeros((1, *self.diffusion.x_shape), device=self.diffusion.device)
            
    #     inpainting_mask[0, ptr] = 1
    #     inpainting_value[0, ptr, :] = obs[0]
    #     reach_max_history_len = ((ptr == max_history_len) or max_history_len == 0)
        
    #     trajs, _ = self.sample(n_samples, t_seq, cond=cond, inpainting_mask=inpainting_mask, inpainting_value=inpainting_value)
        
    #     if reach_max_history_len:
    #         inpainting_mask = torch.cat([inpainting_mask[:,1:], inpainting_mask[:,-1:]], dim=1)
    #         inpainting_value = torch.cat([inpainting_value[:,1:], inpainting_value[:,-1:]], dim=1)

    #     return trajs, inpainting_mask, inpainting_value, min(ptr+1, max_history_len)
