import torch
import torch.nn as nn

import config as conf
from .base import BaseModel


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avgpool(x).view(b, c)
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        # First pre-activation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.silu1 = nn.SiLU(inplace=True)
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        # Second pre-activation
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu2 = nn.SiLU(inplace=True)
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.se = SEBlock(out_channels, reduction=16)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        # Shortcut should branch off directly from the input
        shortcut = self.shortcut(x)
        
        # Pre-activation and first convolution
        out = self.bn1(x)
        out = self.silu1(out)
        out = self.conv1(out)
        
        # Second pre-activation and convolution
        out = self.bn2(out)
        out = self.silu2(out)
        out = self.conv2(out)
        
        # Apply SE
        out = self.se(out)

        # Add shortcut
        out = out + shortcut
        
        return out


class PreActResNet18(BaseModel):
    def __init__(self, num_classes=10):
        super(PreActResNet18, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 42, kernel_size=3, stride=1, padding=1, bias=False)

        # No BN or activation after initial conv to match the pre-activation pattern

        # Four layer groups with different output channels and varying strides
        self.layer1 = self._make_layer(PreActBlock, 42, 42, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 42, 84, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 84, 168, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 168, 336, num_blocks=2, stride=2)

        # Final batch norm and activation
        self.bn = nn.BatchNorm2d(336)
        self.silu = nn.SiLU(inplace=True)

        # Global average pooling and classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(336 * PreActBlock.expansion, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block in the layer may downsample
        layers.append(block(in_channels, out_channels, stride))

        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        out = self.conv1(x)

        # Four main layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Final BN and activation
        out = self.bn(out)
        out = self.silu(out)

        # Global average pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        # Classification
        out = self.fc(out)

        return out

    def __str__(self):
        return "preact_resnet18"

    def get_config(self):
        return {
            "type": "PreActResNet18",
            "num_classes": self.fc[1].out_features,
            "activation": "SiLU",
            "dropout_rate": 0.2
        } 
