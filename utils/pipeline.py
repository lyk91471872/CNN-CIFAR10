import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

import config as conf
from .early_stopping import EarlyStopping
from .augmentation import mixup_data

class Pipeline:
    def __init__(self, model: nn.Module):
        self.model = model.to(conf.TRAIN['device'])
        self.device = conf.TRAIN['device']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), **conf.OPTIMIZER)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **conf.SCHEDULER)
        self.early_stopping = EarlyStopping(
            patience=conf.TRAIN['early_stopping_patience'],
            delta=conf.TRAIN['early_stopping_min_delta'],
            verbose=True
        )
        summary(self.model, (3, 32, 32))

    def train_one_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch and return loss and accuracy."""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for aug_inputs, clean_inputs, targets in train_loader:
            aug_inputs, targets = aug_inputs.to(self.device), targets.to(self.device)

            # Mixup augmentation
            if conf.TRAIN['mixup_alpha'] > 0:
                aug_inputs, targets_a, targets_b, lam = mixup_data(
                    aug_inputs, targets, conf.TRAIN['mixup_alpha']
                )

            optimizer.zero_grad()
            outputs = self.model(aug_inputs)

            # Calculate loss
            if conf.TRAIN['mixup_alpha'] > 0:
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            if conf.TRAIN['mixup_alpha'] > 0:
                # Weighted accuracy based on label proportions
                correct += (lam * predicted.eq(targets_a).float() + 
                          (1 - lam) * predicted.eq(targets_b).float()).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc

    def val_one_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch and return loss and accuracy."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for aug_inputs, clean_inputs, targets in val_loader:
                clean_inputs, targets = clean_inputs.to(self.device), targets.to(self.device)
                outputs = self.model(clean_inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def train(self, train_loader, val_loader, epochs=None, optimizer=None, scheduler=None):
        """Train the model on the provided data loaders."""
        epochs = epochs or conf.TRAIN['epochs']
        optimizer = optimizer or self._create_optimizer()
        scheduler = scheduler or self._create_scheduler(optimizer)
        
        early_stopping = EarlyStopping(
            patience=conf.TRAIN['early_stopping_patience'],
            min_delta=conf.TRAIN['early_stopping_min_delta']
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Get dataset from train_loader to update augmentation probability
        dataset = train_loader.dataset
        if hasattr(dataset, 'update_augmentation_prob'):
            # For DataLoader with Subset, get the original dataset
            if isinstance(dataset, torch.utils.data.Subset):
                orig_dataset = dataset.dataset
                if hasattr(orig_dataset, 'update_augmentation_prob'):
                    dataset = orig_dataset
        
        # Progress bar for epochs
        pbar = tqdm(range(epochs), desc="Training")
        
        for epoch in pbar:
            # Update augmentation probability if dataset supports it
            if hasattr(dataset, 'update_augmentation_prob'):
                dataset.update_augmentation_prob(epoch, conf.PROGRESSIVE_LEARNING)
                aug_prob = dataset.augmentation_prob
            else:
                aug_prob = 1.0  # Default to full augmentation
                
            # Train for one epoch
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer)
            
            # Validate
            val_loss, val_acc = self.val_one_epoch(val_loader)
            
            # Update learning rate
            if scheduler:
                scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}", 
                'train_acc': f"{train_acc:.2f}%", 
                'val_loss': f"{val_loss:.4f}", 
                'val_acc': f"{val_acc:.2f}%",
                'aug_prob': f"{aug_prob:.2f}"
            })
            
            # Check early stopping
            if early_stopping(val_loss):
                print(f"Early stopping triggered")
                break
                
            # Save model if validation loss improved
            if early_stopping.is_best():
                self.model.save()

    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and return them along with their indices."""
        self.model.eval()
        predictions = []
        indices = []

        with torch.no_grad():
            for inputs, idx in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = outputs.max(1)
                predictions.extend(preds.cpu().numpy())
                indices.extend(idx.cpu().numpy())

        return np.array(predictions), np.array(indices)

    def cross_validate(self, dataset: torch.utils.data.Dataset, k_folds: int = 5) -> List[Dict]:
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f'\nFOLD {fold+1}/{k_folds}')
            print('--------------------------------')

            train_subsampler = Subset(dataset, train_ids)
            val_subsampler = Subset(dataset, val_ids)

            train_loader = DataLoader(train_subsampler, shuffle=True, **conf.DATALOADER)
            val_loader = DataLoader(val_subsampler, shuffle=False, **conf.DATALOADER)

            # Create new model for this fold
            self.model = self.model.__class__()
            self.model.to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), **conf.OPTIMIZER)
            self.scheduler = ReduceLROnPlateau(self.optimizer, **conf.SCHEDULER)
            self.early_stopping = EarlyStopping(
                patience=conf.TRAIN['early_stopping_patience'],
                delta=conf.TRAIN['early_stopping_min_delta'],
                verbose=True
            )

            history = self.train(
                train_loader=train_loader,
                val_loader=val_loader,
                should_save=False  # Don't save weights during cross-validation
            )

            fold_results.append({
                'fold': fold + 1,
                'best_val_acc': max(history['val_accs']),
                'history': history
            })

        return fold_results 
