import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, model_urls

"""
Imitation learning network
"""

class ResNetConv(ResNet):    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # necessary ?
        x = self.layer4(x)
        return x

class ResNetAgent(nn.Module):
    def __init__(self, classes):
        super().__init__()

        # base network
        self.resnet = ResNetConv(BasicBlock, [2, 2, 2, 2])
        
        # other network modules
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, classes)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs, filename):
        x = self.resnet(inputs)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x