import sys

import torch
from agent.networks import ResNetAgent
from agent.efficientnet_pytorch.model import EfficientNet

class CBCAgent:
    
    def __init__(self, device = None, history=1):
        self.net = EfficientNet.from_name('efficientnet-b0', {'num_classes': 9}, history=history)
        self.net.to(device)

    @torch.no_grad()
    def predict(self, frames):
        return self.net(frames, '')
