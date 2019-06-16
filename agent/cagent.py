import sys

import torch
from agent.networks import ResNetAgent

class CBCAgent:
    
    def __init__(self, device = None, history=1):
        self.net = ResNetAgent(classes=9, history=history)
        self.net.to(device)

    @torch.no_grad()
    def predict(self, *input, **kwargs):
        return self.net(input, kwargs)

    def act(self, frames):
        pass

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name+"_model")
        torch.save(self.optimizer.state_dict(), file_name+"_optimizer")

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name+"_model"))
        self.optimizer.load_state_dict(torch.load(file_name+"_optimizer"))