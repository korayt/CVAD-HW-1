import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np

class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor,self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.l1 = nn.Linear(1000, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 4)

    def forward(self, img):
        out = self.model(img)
        out = torch.relu(self.l1(out))
        out = torch.relu(self.l2(out))
        out = self.l3(out)
        out = out.T
        out[3] = torch.sigmoid(out[3])
        return out
