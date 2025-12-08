import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Modern CNN block:
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)