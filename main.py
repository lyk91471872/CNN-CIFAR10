# %% [code]
import subprocess
import sys

def install_requirements():
    try:
        import pkg_resources
        required = set(line.strip() for line in open('requirements.txt'))
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        if missing:
            print(f"Installing missing packages: {', '.join(missing)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception as e:
        print(f"Error installing requirements: {e}")

install_requirements()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:33.565282Z","iopub.execute_input":"2025-03-03T21:29:33.565675Z","iopub.status.idle":"2025-03-03T21:29:40.034711Z","shell.execute_reply.started":"2025-03-03T21:29:33.565645Z","shell.execute_reply":"2025-03-03T21:29:40.034013Z"}}
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchsummary import summary
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image
from enum import IntEnum

# Global flags and constants
USE_AMP = True
USE_MIXUP = True
NUM_WORKERS = 64
BATCH_SIZE = 512
PREFETCH_FACTOR = 8

# Define ModelType using IntEnum; index 0: CustomResNet18, index 1: CustomEfficientNetV2_B0_Scaled
class ModelType(IntEnum):
    CUSTOM_RESNET18 = 0
    CUSTOM_EFFNETV2_B0_SCALED = 1

# Set MODEL_TYPE here (e.g., 1 for EfficientNetV2_B0_Scaled)
MODEL_TYPE = 0

# Enable cudnn benchmark for fixed-size inputs
torch.backends.cudnn.benchmark = True

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:40.035812Z","iopub.execute_input":"2025-03-03T21:29:40.036277Z","iopub.status.idle":"2025-03-03T21:29:40.042075Z","shell.execute_reply.started":"2025-03-03T21:29:40.036245Z","shell.execute_reply":"2025-03-03T21:29:40.041269Z"}}
# EarlyStopping helper class
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:40.044313Z","iopub.execute_input":"2025-03-03T21:29:40.044659Z","iopub.status.idle":"2025-03-03T21:29:40.074514Z","shell.execute_reply.started":"2025-03-03T21:29:40.044626Z","shell.execute_reply":"2025-03-03T21:29:40.073727Z"}}
# Data loading function
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:40.075790Z","iopub.execute_input":"2025-03-03T21:29:40.076078Z","iopub.status.idle":"2025-03-03T21:29:40.094559Z","shell.execute_reply.started":"2025-03-03T21:29:40.076049Z","shell.execute_reply":"2025-03-03T21:29:40.093815Z"}}
# Dataset classes
# Training dataset returns (augmented, original, label)
class CIFAR10Dataset(Dataset):
    def __init__(self, data_paths, transform):
        self.data = []
        self.labels = []
        for path in data_paths:
            dict_batch = load_cifar_batch(path)
            self.data.append(dict_batch[b'data'])
            self.labels.extend(dict_batch[b'labels'])
        # Reshape to (N, 32, 32, 3)
        self.data = np.concatenate(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.transform = transform
        # Base transform: only ToTensor and Normalize (clean image)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        original = self.base_transform(pil_img)
        augmented = self.transform(pil_img)
        label = self.labels[idx]
        return augmented, original, label

# Test dataset returns (transformed, idx)
class CIFAR10TestDataset(Dataset):
    def __init__(self, file_path, transform=None):
        dict_batch = load_cifar_batch(file_path)
        self.data = dict_batch[b'data']
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, idx

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:40.095150Z","iopub.execute_input":"2025-03-03T21:29:40.095396Z","iopub.status.idle":"2025-03-03T21:29:40.109366Z","shell.execute_reply.started":"2025-03-03T21:29:40.095366Z","shell.execute_reply":"2025-03-03T21:29:40.108537Z"}}
# Define transforms.
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomCrop(32, padding=2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    # Uncomment next line for stronger augmentation if desired:
    # transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    # Uncomment next line for RandomErasing if desired:
    # transforms.RandomErasing(p=0.3, scale=(0.02,0.33), ratio=(0.3,3.3))
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:40.110225Z","iopub.execute_input":"2025-03-03T21:29:40.110418Z","iopub.status.idle":"2025-03-03T21:29:42.484911Z","shell.execute_reply.started":"2025-03-03T21:29:40.110402Z","shell.execute_reply":"2025-03-03T21:29:42.483838Z"}}
# File paths and dataset creation.
cifar10_dir = 'cifar-10-python/cifar-10-batches-py'
train_files = [os.path.join(cifar10_dir, f"data_batch_{i}") for i in range(1, 6)]
test_file = 'cifar_test_nolabel.pkl'
train_dataset = CIFAR10Dataset(train_files, transform=train_transforms)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:42.485823Z","iopub.execute_input":"2025-03-03T21:29:42.486080Z","iopub.status.idle":"2025-03-03T21:29:42.493147Z","shell.execute_reply.started":"2025-03-03T21:29:42.486060Z","shell.execute_reply":"2025-03-03T21:29:42.492294Z"}}
# Define the BasicBlock for ResNet-18.
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:42.494231Z","iopub.execute_input":"2025-03-03T21:29:42.494598Z","iopub.status.idle":"2025-03-03T21:29:42.512992Z","shell.execute_reply.started":"2025-03-03T21:29:42.494565Z","shell.execute_reply":"2025-03-03T21:29:42.512177Z"}}
# Define the CustomResNet18 architecture.
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
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
        layers = [block(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:42.515651Z","iopub.execute_input":"2025-03-03T21:29:42.515881Z","iopub.status.idle":"2025-03-03T21:29:42.532287Z","shell.execute_reply.started":"2025-03-03T21:29:42.515862Z","shell.execute_reply":"2025-03-03T21:29:42.531559Z"}}
# Define the FusedMBConv block as used in EfficientNetV2.
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:42.533602Z","iopub.execute_input":"2025-03-03T21:29:42.533820Z","iopub.status.idle":"2025-03-03T21:29:42.553010Z","shell.execute_reply.started":"2025-03-03T21:29:42.533801Z","shell.execute_reply":"2025-03-03T21:29:42.552285Z"}}
# Define the CustomEfficientNetV2_B0 architecture (standard EfficientNetV2-B0) then scale channels by 2.
class CustomEfficientNetV2_B0(nn.Module):
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:42.553789Z","iopub.execute_input":"2025-03-03T21:29:42.554069Z","iopub.status.idle":"2025-03-03T21:29:42.571064Z","shell.execute_reply.started":"2025-03-03T21:29:42.554050Z","shell.execute_reply":"2025-03-03T21:29:42.570259Z"}}
# Create a list of model classes; index 0: CustomResNet18, index 1: CustomEfficientNetV2_B0 (scaled).
model_list = [CustomResNet18, CustomEfficientNetV2_B0]

# Helper function to get the model based on the global MODEL_TYPE.
def get_model(num_classes=10):
    return model_list[MODEL_TYPE](num_classes=num_classes)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:42.571706Z","iopub.execute_input":"2025-03-03T21:29:42.571896Z","iopub.status.idle":"2025-03-03T21:29:43.732274Z","shell.execute_reply.started":"2025-03-03T21:29:42.571880Z","shell.execute_reply":"2025-03-03T21:29:43.731438Z"}}
# Instantiate the selected model and print its summary.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=10)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
model = model.to(device)
if isinstance(model, nn.DataParallel):
    summary(model.module, (3, 32, 32))
else:
    summary(model, (3, 32, 32))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:43.733097Z","iopub.execute_input":"2025-03-03T21:29:43.733354Z","iopub.status.idle":"2025-03-03T21:29:43.738150Z","shell.execute_reply.started":"2025-03-03T21:29:43.733335Z","shell.execute_reply":"2025-03-03T21:29:43.737247Z"}}
# Optional mixup function (not used if USE_MIXUP is False).
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    else:
        lam = torch.tensor(1.0, device=x.device)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam.item()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:43.739015Z","iopub.execute_input":"2025-03-03T21:29:43.739289Z","iopub.status.idle":"2025-03-03T21:29:43.752327Z","shell.execute_reply.started":"2025-03-03T21:29:43.739261Z","shell.execute_reply":"2025-03-03T21:29:43.751636Z"}}
# Initialize GradScaler for AMP.
scaler = torch.amp.GradScaler(enabled=USE_AMP)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:43.753054Z","iopub.execute_input":"2025-03-03T21:29:43.753266Z","iopub.status.idle":"2025-03-03T21:29:43.765951Z","shell.execute_reply.started":"2025-03-03T21:29:43.753247Z","shell.execute_reply":"2025-03-03T21:29:43.765241Z"}}
# Training and validation loop functions.
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_desc="Epoch"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    loop = tqdm(loader, desc=epoch_desc, leave=False)
    for augmented, original, labels in loop:
        augmented, original, labels = augmented.to(device), original.to(device), labels.to(device)
        optimizer.zero_grad()
        if USE_MIXUP:
            augmented, targets_a, targets_b, lam = mixup_data(augmented, labels, alpha=1.0)
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(augmented)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(augmented)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(augmented)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(augmented)
                loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    clean_outputs = model(original)
                    clean_loss = criterion(clean_outputs, labels)
            else:
                clean_outputs = model(original)
                clean_loss = criterion(clean_outputs, labels)
        running_loss += clean_loss.item()
        _, preds = torch.max(clean_outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, epoch_time

def val_one_epoch(model, loader, criterion, device, epoch_desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    loop = tqdm(loader, desc=epoch_desc, leave=False)
    with torch.no_grad():
        for batch in loop:
            # If three values are returned, ignore the augmented image.
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _, inputs, labels = batch
            else:
                inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, epoch_time

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:43.766821Z","iopub.execute_input":"2025-03-03T21:29:43.767115Z","iopub.status.idle":"2025-03-03T21:29:43.792529Z","shell.execute_reply.started":"2025-03-03T21:29:43.767072Z","shell.execute_reply":"2025-03-03T21:29:43.791797Z"}}
# Cross-validation function using K-Fold, SGD with momentum, and early stopping.
def cross_validate_model(dataset, k_folds=5, max_epochs=100, lr=0.1, batch_size=BATCH_SIZE, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n[Fold {fold+1}/{k_folds}]")
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
        model = get_model(num_classes=10)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(max_epochs):
            print(f"\nFold {fold+1}, Epoch {epoch+1}/{max_epochs}")
            train_loss, train_acc, train_time = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                                 epoch_desc=f"Fold{fold+1}-Train-Epoch{epoch+1}")
            val_loss, val_acc, val_time = val_one_epoch(model, val_loader, criterion, device,
                                                       epoch_desc=f"Fold{fold+1}-Val-Epoch{epoch+1}")
            scheduler.step()
            total_epoch_time = train_time + val_time
            print(f"  -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | Epoch Time: {total_epoch_time:.2f}s")
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:43.793369Z","iopub.execute_input":"2025-03-03T21:29:43.793640Z","iopub.status.idle":"2025-03-03T21:29:43.812616Z","shell.execute_reply.started":"2025-03-03T21:29:43.793621Z","shell.execute_reply":"2025-03-03T21:29:43.811762Z"}}
# Final training and prediction function with early stopping.
def train_and_predict(dataset, test_file, max_epochs=100, batch_size=BATCH_SIZE, lr=0.1, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=10)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(max_epochs):
        print(f"\n[Final Training] Epoch {epoch+1}/{max_epochs}")
        train_loss, train_acc, train_time = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                             epoch_desc=f"Final-Train-Epoch{epoch+1}")
        scheduler.step()
        print(f"  -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Epoch Time: {train_time:.2f}s")
        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered during final training.")
            break
    test_dataset = CIFAR10TestDataset(test_file, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
    model.eval()
    predictions = []
    indices = []
    with torch.no_grad():
        loop_test = tqdm(test_loader, desc="Test Inference", leave=False)
        for inputs, idx in loop_test:
            inputs = inputs.to(device)
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            indices.extend(idx.cpu().numpy())
    submission = pd.DataFrame({'ID': indices, 'Label': predictions})
    submission.to_csv('prediction.csv', index=False)
    print("\nTest predictions saved to prediction.csv")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-03T21:29:43.813559Z","iopub.execute_input":"2025-03-03T21:29:43.813842Z","execution_failed":"2025-03-03T23:55:17.127Z"}}
# Optionally, run cross-validation to assess performance.
# Uncomment the next line to run cross-validation.
cross_validate_model(train_dataset, k_folds=5, max_epochs=100, lr=0.1, batch_size=BATCH_SIZE, patience=10)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-03-03T23:55:17.127Z"}}
# Final training on all data and prediction on test set.
train_and_predict(train_dataset, test_file, max_epochs=100, batch_size=BATCH_SIZE, lr=0.1, patience=10)
