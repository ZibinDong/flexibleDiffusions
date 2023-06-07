import torch.nn as nn


def abbreviate_number(number):
    if number < 1000:
        return str(number)
    elif number < 1000000:
        return str(round(number/1000, 2)) + "K"
    else:
        return str(round(number/1000000, 2)) + "M"

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
