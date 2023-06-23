import torch.nn as nn
from typing import List


class EvalModules:
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules
    
    def __enter__(self):
        for module in self.modules:
            module.eval()
    def __exit__(self, *args):
        for module in self.modules:
            module.train()

class TrainModules:
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules
    
    def __enter__(self):
        for module in self.modules:
            module.train()
    def __exit__(self, *args):
        for module in self.modules:
            module.eval()

class FreezeModules:
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules
    
    def __enter__(self):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = False
    def __exit__(self, *args):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = True
                
class UnfreezeModules:
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules
    
    def __enter__(self):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = True
    def __exit__(self, *args):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = False

def abbreviate_number(number):
    if number < 1000:
        return str(number)
    elif number < 1000000:
        return str(round(number/1000, 2)) + "K"
    else:
        return str(round(number/1000000, 2)) + "M"

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
