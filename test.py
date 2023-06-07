from models.unet import UNet
from diffusion import DenoiseDiffusion
from classifiers import HalfUNetClassifier
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
    use_norm=(True, True),
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

x = torch.randn((1, 3, 32, 32), device=device)
t = torch.randint(1000, (1,), device=device)
y = torch.randint(10, (1,), device=device)

loss = classifier.loss(x, t, y)
grad = classifier.gradients(x, t, y)

