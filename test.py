from models.unet import UNet
import torch

device = "cuda"

m = UNet(
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

x = torch.randn((1, 3, 32, 32), device=device)
t = torch.randint(1000, (1,), device=device)
cond = torch.randn((1, 5), device=device)

y = m(x, t, cond)

