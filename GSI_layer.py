import torch.nn as nn
import math
import torch as t


class Add_norm(nn.Module):
    def __init__(self, Omodel):
        super(Add_norm, self).__init__()
        self.Omodel = Omodel
        self.mean = nn.Parameter(t.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = nn.Parameter(t.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.Omodel(x)
        return x


class Add_GSInorm(nn.Module):
    def __init__(self, Omodel):
        super(Add_GSInorm, self).__init__()
        self.Omodel = Omodel
        self.mean = nn.Parameter(t.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = nn.Parameter(t.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        with t.no_grad():
            x_t = t.clamp(t.round(x * 255), 0, 255) / 255
            T = x_t - x
        x = T + x
        x = t.sin(x * 510 * math.pi + math.pi) + x
        x = (x - self.mean) / self.std
        x = self.Omodel(x)
        return x
