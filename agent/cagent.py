import sys

import torch
from networks import CNN

class CBCAgent:
    
    def __init__(self, device = None, weighted=False, history=1):
        self.net = CNN(channels=3, history_length=history,n_classes=9)
        self.class_weights = torch.Tensor([1, 1, 1, 1, 1]).to(cuda)
        if weighted:
            self.class_weights = torch.Tensor([0.39177557, 1., 2.17533432, 0.64403549, 7.8568873]).to(cuda)
        self.net.to(device)

    @torch.no_grad()
    def predict(self, X):
        return self.net(X)

    def act(self, frames):
        pass

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name+"_model")
        torch.save(self.optimizer.state_dict(), file_name+"_optimizer")

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name+"_model"))
        self.optimizer.load_state_dict(torch.load(file_name+"_optimizer"))