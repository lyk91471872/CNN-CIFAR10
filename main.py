import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pickle

from config import *
from dataset import CIFAR10Dataset, CIFAR10TestDataset
from models import CustomResNet18, CustomEfficientNetV2_B0
from utils import train_one_epoch, val_one_epoch, EarlyStopping, mixup_data

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

def get_model(num_classes=10):
    """Get the model based on MODEL_TYPE configuration."""
    if MODEL_TYPE == ModelTypeEnum.CUSTOM_RESNET18:
        return CustomResNet18(num_classes=num_classes)
    elif MODEL_TYPE == ModelTypeEnum.CUSTOM_EFFNETV2_B0:
        return CustomEfficientNetV2_B0(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

def cross_validate_model(dataset, k_folds=5, max_epochs=MAX_EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, patience=PATIENCE):
    """Perform k-fold cross validation on the model."""
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\nFOLD {fold+1}/{k_folds}')
        print('--------------------------------')
        
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)
        
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                                prefetch_factor=PREFETCH_FACTOR)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                              prefetch_factor=PREFETCH_FACTOR)
        
        model = get_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        best_val_acc = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = val_one_epoch(model, val_loader, criterion, DEVICE)
            
            scheduler.step()
            early_stopping(val_loss, model)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        })
    
    return fold_results

def train_and_predict(dataset, test_file, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, patience=PATIENCE):
    """Train the model on the full dataset and generate predictions for the test set."""
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                            prefetch_factor=PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                          prefetch_factor=PREFETCH_FACTOR)
    
    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        early_stopping(val_loss, model)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model and generate predictions
    model.load_state_dict(torch.load('best_model.pth'))
    test_dataset = CIFAR10TestDataset(test_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                           prefetch_factor=PREFETCH_FACTOR)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Acc')
    plt.plot(history['val_accs'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    install_requirements()
    
    # Load dataset
    dataset = CIFAR10Dataset(data_paths=DATA_PATHS, transform=None)
    
    # Perform cross-validation
    print("\nStarting cross-validation...")
    fold_results = cross_validate_model(dataset)
    
    # Print cross-validation results
    print("\nCross-validation results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Best validation accuracy = {result['best_val_acc']:.2f}%")
    
    # Train on full dataset and generate predictions
    print("\nTraining on full dataset...")
    predictions, history = train_and_predict(dataset, TEST_FILE)
    
    # Plot training history
    plot_training_history(history)
    
    # Save predictions
    with open('predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    print("\nPredictions saved to predictions.pkl")

if __name__ == "__main__":
    main()
