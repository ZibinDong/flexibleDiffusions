from models.unet import UNet
from models.dit import DiT2d
from diffusion import DenoiseDiffusion
from classifiers import HalfUNetClassifier
from samplers import DDIMSampler
import torch

device = "cuda:7"

x = torch.randn((1, 3, 32, 32), device=device)
t = torch.randint(1000, (1,), device=device)
y = torch.randint(10, (1,), device=device)
cond = torch.randn((1, 5), device=device)

# eps_model = UNet(
#     data_channels=3,
#     n_channels=16,
#     time_channels=16,
#     cond_dim=5,
#     tensor_dim=2,
#     n_groups=1,
#     ch_mults=(1, 2),
#     use_attn=(False, True),
#     use_norm=(True, True),
#     n_heads=1,
#     n_blocks=2,
# ).to(device)

eps_model = DiT2d(
    input_size=32,
    patch_size=2,
    cond_dim=5,
    data_channels=3,
    hidden_size=384,
    n_heads=6,
    depth=12,
).to(device)

print("[eps_model]", end="")
print(eps_model)

diffusion = DenoiseDiffusion(
    eps_model=eps_model,
    uncond_prob=0.25,
    T=1000,
    device=device,
    beta_schedule="cosine",
    s=0.008,
)

loss = diffusion.loss(x, cond)

print("[diffusion]", end="")
print(diffusion)

classifier = HalfUNetClassifier(
    data_channels=3,
    n_channels=16,
    time_channels=16,
    tensor_dim=2,
    n_groups=1,
    ch_mults=(1, 2),
    use_attn=(False, True),
    use_norm=(True, True),
    n_heads=1,
    n_blocks=2,
    y_type="label",
    n_categories=10,
).to(device)

logp, grad = classifier.gradients(x, t, y)

print("[classifier]", end="")
print(classifier)

sampler = DDIMSampler(
    diffusion=diffusion,
    ddim_eta=0.0,
    ddim_sample_steps=20,
    ddim_discretize="quad",
    classifier=classifier,
)

print("[sampler]", end="")
print(sampler)


y = torch.randint(10, (10,), device=device)
cond = torch.randn((10, 5), device=device)

sampled_x, logp, denoise_history = sampler.sample(
    n_samples=10,
    y=y,
    cond=cond,
)