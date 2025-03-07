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

from sklearn.metrics import confusion_matrix
import pandas as pd

import config as conf
from dataset import create_dataset
from utils.early_stopping import EarlyStopping
from utils.augmentation import mixup_data
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_crossval_history, plot_crossval_confusion_matrices
from utils.session import SessionTracker, get_session_filename
import copy

class Pipeline:
    def __init__(self, model: nn.Module):
        self.model = model.to(conf.TRAIN['device'])
        self.device = conf.TRAIN['device']
        self.criterion = nn.CrossEntropyLoss()

        # Store initial learning rate for warmup
        self.initial_lr = conf.OPTIMIZER['lr']

        # Create the optimizer
        self.optimizer = optim.SGD(self.model.parameters(), **conf.OPTIMIZER)

        # Create scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, **conf.SCHEDULER)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=conf.TRAIN['early_stopping_patience'],
            delta=conf.TRAIN['early_stopping_min_delta'],
            verbose=True
        )
        self.use_augmentation = False
        summary(self.model, (3, 32, 32))

    def _warmup_learning_rate(self, epoch):
        """Apply linear warmup to learning rate during initial epochs."""
        if epoch < conf.TRAIN.get('warmup_epochs', 0):
            # Linear warmup from 10% to 100% of base learning rate
            warmup_factor = 0.1 + 0.9 * epoch / conf.TRAIN.get('warmup_epochs', 1)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * warmup_factor
            print(f"Warmup epoch {epoch+1}: LR = {self.initial_lr * warmup_factor:.6f}")
        elif epoch == conf.TRAIN.get('warmup_epochs', 0):
            # Reset to initial learning rate after warmup
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr
            print(f"Warmup complete: LR = {self.initial_lr:.6f}")

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
        train_acc = 100 * correct / total
        return train_loss, train_acc

    def val_one_epoch(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray]:
        """Validate for one epoch and return loss and accuracy."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        # For confusion matrix
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for aug_inputs, clean_inputs, targets in val_loader:
                inputs = aug_inputs if self.use_augmentation else clean_inputs
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Collect targets and predictions for confusion matrix
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)

        return val_loss, val_acc, cm

    def train(self, train_loader: DataLoader, val_loader: DataLoader, should_save: bool = True) -> Dict[str, List[float]]:
        """Train the model and optionally save the best model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            should_save: Whether to save the best model

        Returns:
            Dictionary containing training history
        """
        # Create a session tracker
        session = SessionTracker(self.model, "training")
        
        history = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': []
        }

        # Keep track of best val metrics for early stopping and model saving
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_confusion_matrix = None
        best_epoch = 0
        patience_counter = 0
        completed_epochs = 0
        
        print(f"\nTraining on {self.device}...")
        
        # Loop over epochs
        for epoch in range(1, conf.TRAIN['epochs'] + 1):
            # Decide whether to use augmentation for this epoch
            no_aug_epochs = conf.TRAIN.get('no_augmentation_epochs', 0)
            self.use_augmentation = epoch > no_aug_epochs
            
            # Apply learning rate warmup if in warmup phase
            if epoch < conf.TRAIN['warmup_epochs']:
                self._warmup_learning_rate(epoch)
            
            # Train and validate for one epoch
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, confusion_matrix = self.val_one_epoch(val_loader)
            
            # Update session with current epoch data
            session.update_epoch(epoch, val_acc / 100.0)  # Convert from % to 0-1 range
            
            # Update history
            history['train_losses'].append(train_loss)
            history['train_accs'].append(train_acc)
            history['val_losses'].append(val_loss)
            history['val_accs'].append(val_acc)
            
            # Print metrics
            print(f"Epoch {epoch}/{conf.TRAIN['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Step the scheduler based on validation loss
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Check if this is the best model so far
            is_best = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_confusion_matrix = confusion_matrix
                best_epoch = epoch
                session.update_confusion_matrix(confusion_matrix)
                patience_counter = 0
                is_best = True
                # Save model if should_save is True and we've passed min_save_epoch
                if should_save and epoch >= conf.TRAIN.get('min_save_epoch', 0):
                    self.model.save(epoch=epoch, accuracy=val_acc/100, path=session.weights_path)
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= conf.TRAIN['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
            
            # Store current epoch count
            completed_epochs = epoch
        
        # After training completes:
        
        # 1. Plot and save the training history
        history_path = plot_training_history(
            history=history, 
            model=self.model, 
            epoch=best_epoch, 
            accuracy=best_val_acc/100,
            save_path=session.history_plot_path
        )
        
        # 2. Plot and save the confusion matrix from the best epoch
        if best_confusion_matrix is not None:
            cm_path = plot_confusion_matrix(
                range(10), range(10),  # Placeholder - not used when confusion_matrix is provided
                model=self.model, 
                epoch=best_epoch, 
                accuracy=best_val_acc/100,
                save_path=session.confusion_matrix_path,
                confusion_matrix=best_confusion_matrix  # Pass the actual confusion matrix
            )
        
        # 3. Generate predictions and save them
        test_dataset = create_dataset(data_source=conf.TEST_DATA_PATH, mode='test')
        test_loader = DataLoader(test_dataset, shuffle=False, **conf.DATALOADER)
        predictions, indices = self.predict(test_loader)
        
        # 4. Save predictions to CSV
        submission = pd.DataFrame({'ID': indices, 'Label': predictions})
        submission.to_csv(session.predictions_path, index=False)
        
        # 5. Add metrics and files to session
        session.add_metrics({
            "epochs": completed_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_acc": float(best_val_acc / 100)  # Convert to 0-1 range
        })
        
        session.add_file("model_weights", session.weights_path)
        session.add_file("history_plot", session.history_plot_path)
        session.add_file("confusion_matrix", session.confusion_matrix_path)
        session.add_file("predictions", session.predictions_path)
        
        # 6. Save session data
        session_path = session.save()
        print(f"Training session data saved to {session_path}")
        
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
                indices.extend(idx.numpy())

        return np.array(predictions), np.array(indices)

    def cross_validate(self, dataset: torch.utils.data.Dataset, k_folds: int = 5) -> List[Dict]:
        """Perform k-fold cross-validation and return results for each fold."""
        # Create session tracker for cross-validation
        session = SessionTracker(self.model, "crossval")
        
        # Save the initial model state to reset for each fold
        initial_state = copy.deepcopy(self.model.state_dict())
        
        # Initialize k-fold cross validation
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_results = []
        all_matrices = []
        
        # Loop over folds
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"\n--- Fold {fold+1}/{k_folds} ---")
            
            # Reset model to initial state
            self.model.load_state_dict(initial_state)
            
            # Reset optimizer and scheduler
            self._init_optimizer_scheduler()
            
            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(
                dataset, 
                batch_size=conf.DATALOADER['batch_size'],
                sampler=train_subsampler,
                num_workers=conf.DATALOADER['num_workers'],
                pin_memory=conf.DATALOADER['pin_memory']
            )
            
            val_loader = DataLoader(
                dataset, 
                batch_size=conf.DATALOADER['batch_size'],
                sampler=val_subsampler,
                num_workers=conf.DATALOADER['num_workers'],
                pin_memory=conf.DATALOADER['pin_memory']
            )
            
            # Train model for this fold (without saving)
            fold_history = self.train(train_loader, val_loader, should_save=False)
            
            # Get best metrics
            best_val_acc = max(fold_history['val_accs'])
            best_epoch = fold_history['val_accs'].index(best_val_acc) + 1
            best_val_loss = fold_history['val_losses'][best_epoch-1]
            
            # Collect confusion matrix for this fold
            # Evaluate on validation set one more time to get confusion matrix
            _, _, fold_cm = self.val_one_epoch(val_loader)
            all_matrices.append(fold_cm)
            
            # Store results
            fold_results.append({
                'fold': fold + 1,
                'best_val_acc': best_val_acc / 100,  # Convert to 0-1 range
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'history': fold_history,
                'confusion_matrix': fold_cm
            })
            
            print(f"Fold {fold+1} - Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
        
        # Calculate average validation accuracy
        avg_val_acc = sum(r['best_val_acc'] for r in fold_results) / len(fold_results)
        std_val_acc = np.std([r['best_val_acc'] for r in fold_results])
        print(f"\nAverage Val Acc across {k_folds} folds: {avg_val_acc*100:.2f}% ± {std_val_acc*100:.2f}%")
        
        # Plot and save cross-validation history
        history_path = plot_crossval_history(
            fold_results, 
            model=self.model,
            save_path=session.history_plot_path
        )
        
        # Plot and save cross-validation confusion matrices
        cm_path = plot_crossval_confusion_matrices(
            fold_results, 
            model=self.model,
            save_path=session.confusion_matrix_path
        )
        
        # Add metrics and file paths to session
        session.add_metrics({
            "folds": k_folds,
            "avg_val_acc": float(avg_val_acc),
            "std_val_acc": float(std_val_acc),
            "fold_metrics": [
                {
                    "fold": r["fold"],
                    "best_val_acc": float(r["best_val_acc"]),
                    "best_val_loss": float(r["best_val_loss"]),
                    "best_epoch": r["best_epoch"]
                } for r in fold_results
            ]
        })
        
        session.add_file("history_plot", session.history_plot_path)
        session.add_file("confusion_matrix_plot", session.confusion_matrix_path)
        
        # Save session data
        session_path = session.save()
        print(f"Cross-validation session data saved to {session_path}")
        
        return fold_results 
