from models.unet import UNet
from diffusion import DenoiseDiffusion
import torch

device = "cuda"

eps_model = UNet(
    data_channels=3,
    n_channels=16,
    time_channels=16,
    cond_dim=5,
    tensor_dim=2,
    n_groups=1,
    ch_mults=(1, 2),
    use_attn=(False, True),
    n_heads=1,
    n_blocks=2,
).to(device)

diffusion = DenoiseDiffusion(
    eps_model=eps_model,
    uncond_prob=0.25,
    T=1000,
    device=device,
    beta_schedule="cosine",
    s=0.008,
)

x = torch.randn((1, 3, 32, 32), device=device)
cond = torch.randn((1, 5), device=device)

loss = diffusion.loss(x, cond)

print(diffusion.eps_model)

