import torch
import torch.nn as nn
from .blocks import ConvBlock

class ModernCNNv1(nn.Module):
    def __init__(self, num_classes: int=10):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.classifier(x)
        return x