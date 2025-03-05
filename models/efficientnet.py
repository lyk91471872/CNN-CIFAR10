import torch
import torch.nn as nn
from .base import BaseModel

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(FusedMBConv, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return out

class CustomEfficientNetV2_B0(BaseModel):
    def __init__(self, num_classes=10):
        super(CustomEfficientNetV2_B0, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # MBConv blocks
        self.layer1 = self._make_layer(32, 16, 2, stride=1, expand_ratio=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2, expand_ratio=4)
        self.layer3 = self._make_layer(32, 48, 3, stride=2, expand_ratio=6)
        self.layer4 = self._make_layer(48, 96, 4, stride=2, expand_ratio=6)
        
        # Final conv and pooling
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, expand_ratio):
        layers = []
        layers.append(FusedMBConv(in_channels, out_channels, stride, expand_ratio))
        for _ in range(1, num_blocks):
            layers.append(FusedMBConv(out_channels, out_channels, 1, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
        
    def __str__(self):
        return "efficientnet_v2_b0" 