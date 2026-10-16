"""A frozen ImageNet ResNet-18 with a fitted binary logistic regression head."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class DefectNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.head = nn.Linear(512, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, x):
        return self.encoder((x - self.mean) / self.std)

    def forward(self, x):
        return torch.sigmoid(self.head(self.features(x)))
