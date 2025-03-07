import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from tqdm import tqdm
from typing import Dict, List, Tuple
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
        self.use_augmentation = False
        summary(self.model, (3, 32, 32))

    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch and return loss and accuracy."""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for aug_inputs, clean_inputs, targets in train_loader:
            inputs = aug_inputs if self.use_augmentation else clean_inputs
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Mixup augmentation
            if self.use_augmentation and conf.TRAIN['mixup_alpha'] > 0:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, conf.TRAIN['mixup_alpha']
                )

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Calculate loss
            if self.use_augmentation and conf.TRAIN['mixup_alpha'] > 0:
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            if self.use_augmentation and conf.TRAIN['mixup_alpha'] > 0:
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

    def train(self, train_loader: DataLoader, val_loader: DataLoader, should_save: bool = True) -> Dict[str, List[float]]:
        """Train the model and optionally save the best model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            should_save: Whether to save the best model based on validation loss

        Returns:
            Dictionary containing training history
        """
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }

        best_val_loss = float('inf')
        best_model_state = None
        completed_epochs = 0
        
        pbar = tqdm(range(conf.TRAIN['epochs']), desc="Training")
        for epoch in pbar:
            self.use_augmentation = (epoch >= conf.TRAIN['no_augmentation_epochs'])
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.val_one_epoch(val_loader)
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss)  # Early stopping based on validation loss
            
            completed_epochs = epoch + 1  # Keep track of completed epochs
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model state dictionary (not to disk yet)
                best_model_state = self.model.state_dict().copy()
                print(f"Epoch {epoch+1}: New best model with validation loss {val_loss:.4f}")
                
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_accs'].append(train_acc)
            history['val_accs'].append(val_acc)

            pbar.set_postfix({
                'tr_loss': f'{train_loss:.2f}',
                'tr_acc': f'{train_acc:.2f}%',
                'val_loss': f'{val_loss:.2f}',
                'val_acc': f'{val_acc:.2f}%'
            })
        
        # After training is complete, load the best model state and save it
        if should_save and best_model_state is not None:
            # Load the best model state
            self.model.load_state_dict(best_model_state)
            # Save the best model to disk with the number of epochs
            self.model.save(epochs=completed_epochs)
            print(f"Training completed after {completed_epochs} epochs. Best model saved with validation loss {best_val_loss:.4f}")

        return history

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
