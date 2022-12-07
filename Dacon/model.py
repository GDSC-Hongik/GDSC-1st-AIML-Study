import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InceptionV3(nn.Module):
    def __init__(self):
        self.model = models.inception_v3()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 10),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        out = self.model(x)
        return out

