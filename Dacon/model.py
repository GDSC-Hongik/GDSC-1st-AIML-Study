import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.linear1 = nn.Linear(1000, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        out = self.model(x)
        out = self.linear1(out)
        out = self.linear2(out)

        return out
