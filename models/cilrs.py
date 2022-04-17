import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
import PIL

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = tensor.permute(1,2,0)
    tensor = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(tensor)

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS,self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.reduce = nn.Linear(1000, 7)
        self.l1 = nn.Linear(8, 16)
        self.l2 = nn.Linear(16, 3)

    def forward(self, img, command, speed):
        # Output from resnet has shape (N, 1000)
        out = self.model(img)
        # Reduce dimension, now has shape (N, 7)
        out = torch.relu(self.reduce(out))
        if type(command) is not int:
            command = command.reshape(64, 1)
            #speed = speed.reshape(64, 1)
            #speed = speed.float()
        else:
            command = torch.tensor([command], dtype=torch.float32)
            #speed = torch.tensor([speed], dtype=torch.float32)
            command = command.reshape(1,1)
            #speed = speed.reshape(1,1)
        # Concat price along feature dimension, shape (N, 7+1)
        out = torch.cat([out, command], dim=1)
        out = torch.relu(self.l1(out))
        out = self.l2(out)
        return out.T
