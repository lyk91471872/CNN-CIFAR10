import torch
import torch.nn as nn

import config as conf
from .base import BaseModel

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.silu(out)
        return out

class CustomResNet18(BaseModel):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.silu = nn.SiLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 232, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 232, 268, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(268 * BasicBlock.expansion, num_classes)
        )

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def __str__(self):
        return "resnet18" 

class CustomResNet34(BaseModel):
    def __init__(self, num_classes=10):
        super(CustomResNet34, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.silu = nn.SiLU(inplace=True)

        # Standard ResNet34 layer configuration: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, num_blocks=3, stride=2)

        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512 * BasicBlock.expansion, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may have a stride to downsample
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
        out = self.bn1(out)
        out = self.silu(out)

        # ResNet blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global average pooling and classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def __str__(self):
        return "resnet34"

    def get_config(self):
        """Get model configuration as a dictionary."""
        return {
            "type": "CustomResNet34",
            "num_classes": self.fc[1].out_features,
            "activation": "SiLU",
            "dropout_rate": 0.2
        } 

class CustomResNet18X(BaseModel):
    def __init__(self, x=conf.X, num_classes=10):  # Find largest x for <5M parameters
        super(CustomResNet18X, self).__init__()
        self.conv1 = nn.Conv2d(3, x, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(x)
        self.silu = nn.SiLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, x, x, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, x, 2*x, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 2*x, 4*x, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 4*x, 8*x, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8*x * BasicBlock.expansion, num_classes)
        )

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def __str__(self):
        return "resnet18x" 
