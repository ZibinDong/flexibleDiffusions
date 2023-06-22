import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
import wandb
from models import UNet
from samplers import DDIMSampler, DDPMSampler
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from diffusion import DenoiseDiffusion

device = "cuda:5"

# prepare dataset
dataset = MNIST(
    root='/home/rl/datasets/minist',
    train=True, download=True,
    transform=Compose([
        ToTensor(),
        Resize((32, 32)),
        Normalize((0.5), (0.5))
    ])
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# prepare model
eps_model = UNet(
    data_channels=1,
    n_channels=32,
    time_channels=32,
    cond_dim=10,
    tensor_dim=2,
    n_groups=4,
    ch_mults=(1, 2, 2, 2),
    use_attn=(False, False, True, True),
    use_norm=(True, True, True, True),
    n_heads=1,
    n_blocks=2,
).to(device)

diffusion = DenoiseDiffusion(
    eps_model=eps_model,
    uncond_prob=0.2,
    T=1000,
    device=device,
    beta_schedule="linear",
    # s=0.008,
    begin = 0.0001,
    end = 0.02,
)

sampler = DDIMSampler(
    diffusion=diffusion,
    ddim_eta=0.0,
    ddim_sample_steps=20,
    ddim_discretize="uniform",
    cf_strength=5.,
)
# sampler = DDPMSampler(
#     diffusion=diffusion,
#     cf_strength=0.5,
# )

optim = torch.optim.Adam(eps_model.parameters(), lr=5e-5)


# useful functions
def eval():
    sampled_x, _ = sampler.sample(
        16, 
        cond=F.one_hot(torch.randint(10, (16,), device=device), 10).float()
    )
    sampled_x = torch.cat(torch.chunk(sampled_x, 4, 0), 2)
    sampled_x = torch.cat(torch.chunk(sampled_x, 4, 0), 3)
    sampled_x = sampled_x.permute(0, 2, 3, 1)[0]
    sampled_x = torch.round((sampled_x.clip(-1.,1.)+1.)/2.*255.).long()
    return sampled_x.cpu().numpy()

wandb.init(project="flexible_diffusions", entity="grandpadzb", name="minist")
step=0
for e in range(30):
    if e % 5 == 1:
        wandb.log({
            "samples": [wandb.Image(eval())],
        }, step = step)
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for x, y in dataloader:
            x = x.to(device)
            y = F.one_hot(y, 10).float().to(device)
            
            optim.zero_grad()
            loss = diffusion.loss(x, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(eps_model.parameters(), 10.)
            optim.step()
            
            pbar.set_description(f"loss: {loss.item():.4f}")
            pbar.update(1)
            step += 1

            if step % 100 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                }, step=step)

# wandb.log({
#     "samples": [wandb.Image(eval())],
# }, step = step)
# torch.save(eps_model.state_dict(), "eps_model.pt")