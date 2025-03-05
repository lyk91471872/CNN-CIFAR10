import torch
import torch.nn as nn
from .base import BaseModel

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(FusedMBConv, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        if expand_ratio != 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Define the CustomEfficientNetV2_B0 architecture (standard EfficientNetV2-B0) then scale channels by 2.
class CustomEfficientNetV2_B0(BaseModel):
    def __init__(self, num_classes=10):
        super(CustomEfficientNetV2_B0, self).__init__()
        # Stem: original uses 32 channels; scale by 2 -> 64 channels.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        # Stage 1: 2 blocks, no expansion, output remains 64.
        self.stage1 = nn.Sequential(
            FusedMBConv(64, 64, stride=1, expand_ratio=1),
            FusedMBConv(64, 64, stride=1, expand_ratio=1)
        )
        # Stage 2: 3 blocks, expansion factor 2, output becomes 64*2 = 128.
        self.stage2 = nn.Sequential(
            FusedMBConv(64, 128, stride=2, expand_ratio=2),
            FusedMBConv(128, 128, stride=1, expand_ratio=2),
            FusedMBConv(128, 128, stride=1, expand_ratio=2)
        )
        # Stage 3: 4 blocks, expansion factor 2, output becomes 112*2 = 224.
        self.stage3 = nn.Sequential(
            FusedMBConv(128, 224, stride=2, expand_ratio=2),
            FusedMBConv(224, 224, stride=1, expand_ratio=2),
            FusedMBConv(224, 224, stride=1, expand_ratio=2)
        )
        # Stage 4: 1 block, expansion factor 2, output becomes 192*2 = 384.
        self.stage4 = nn.Sequential(
            FusedMBConv(224, 384, stride=2, expand_ratio=2)
        )
        # Head: add a 1x1 conv to increase channels to 1280, then average pool and FC.
        self.head_conv = nn.Sequential(
            nn.Conv2d(384, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        
    def __str__(self):
        return "efficientnet_v2_b0" 
