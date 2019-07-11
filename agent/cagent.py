import sys

import torch
from agent.networks import ResNetAgent
from agent.efficientnet_pytorch.model import EfficientNet

class CBCAgent:
    
    def __init__(self, device = None, history=1, name=''):
        if name=='resnet18':
            self.net = ResNetAgent(classes=9, history=history)
        elif name=='efficient-double-large' :
            self.net = EfficientNet.from_name('efficientnet-b1', {'num_classes': 3}, history=history, double=True)
        elif name=='efficient-double' :
            self.net = EfficientNet.from_name('efficientnet-b0', {'num_classes': 3}, history=history, double=True)
        else :
            self.net = EfficientNet.from_name('efficientnet-b0', {'num_classes': 9}, history=history)
        self.net.to(device)

    @torch.no_grad()
    def predict(self, frames):
        return self.net(frames)
    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)

