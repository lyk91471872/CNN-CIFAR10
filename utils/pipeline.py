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

class Pipeline:
    def __init__(self, model: nn.Module):
        self.model = model.to(conf.TRAIN['device'])
        self.device = conf.TRAIN['device']
        self.criterion = nn.CrossEntropyLoss()
        
        # Reset best state dict
        self.model.best_state_dict = None

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
                inputs = clean_inputs
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
            should_save: Whether to save the best model based on validation loss

        Returns:
            Dictionary containing training history
        """
        # Create a session tracker
        session = SessionTracker(self.model, "training")

        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }

        best_val_loss = float('inf')
        best_model_state = None
        best_confusion_matrix = None
        best_val_acc = 0.0
        completed_epochs = 0

        pbar = tqdm(range(conf.TRAIN['epochs']), desc="Training")
        for epoch in pbar:
            self.use_augmentation = (epoch >= conf.TRAIN['no_augmentation_epochs'])
            self._warmup_learning_rate(epoch)
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, cm = self.val_one_epoch(val_loader)

            # Only use scheduler after warmup period
            if epoch >= conf.TRAIN.get('warmup_epochs', 0):
                self.scheduler.step(val_loss)

            self.early_stopping(val_loss)  # Early stopping based on validation loss

            completed_epochs = epoch + 1  # Keep track of completed epochs

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc / 100.0  # Convert from percentage to decimal
                best_confusion_matrix = cm
                # Save the model state dictionary (not to disk yet)
                best_model_state = self.model.state_dict().copy()
                # Store the best state in the model for cross-validation
                self.model.best_state_dict = best_model_state
                print(f"Epoch {epoch+1}: New best model with validation loss {val_loss:.4f}, accuracy {val_acc:.2f}%")

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

            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # After training is complete:
        # 1. Load the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model state from epoch with val loss {best_val_loss:.4f}, accuracy {best_val_acc*100:.2f}%")

        # 2. Save the best model to disk with the number of epochs and accuracy
        if should_save:
            weights_path = self.model.save(epoch=completed_epochs, accuracy=best_val_acc)

            # 3. Plot training history and confusion matrix
            history_path = plot_training_history(
                history, 
                model=self.model, 
                epoch=completed_epochs, 
                accuracy=best_val_acc
            )

            # 4. Plot confusion matrix
            if best_confusion_matrix is not None:
                cm_path = plot_confusion_matrix(
                    confusion_matrix_data=best_confusion_matrix,
                    model=self.model, 
                    epoch=completed_epochs, 
                    accuracy=best_val_acc
                )

            # 5. Generate predictions and save them
            test_dataset = create_dataset(data_source=conf.TEST_DATA_PATH, mode='test')
            test_loader = DataLoader(test_dataset, shuffle=False, **conf.DATALOADER)
            predictions, indices = self.predict(test_loader)

            # Create prediction file with matched naming
            pred_path = get_session_filename(
                self.model, 
                epoch=completed_epochs, 
                accuracy=best_val_acc, 
                extension="csv", 
                directory=conf.PREDICTIONS_DIR
            )

            # Save predictions
            submission = pd.DataFrame({'ID': indices, 'Label': predictions})
            submission.to_csv(pred_path, index=False)
            print(f"\nTest predictions saved to {pred_path}")

            # 6. Update session tracker with all paths and metrics
            session.add_metrics({
                "epochs": completed_epochs,
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "final_train_loss": float(history['train_losses'][-1]),
                "final_train_acc": float(history['train_accs'][-1] / 100.0),
            })

            session.add_file("weights", weights_path)
            session.add_file("history_plot", history_path)
            session.add_file("confusion_matrix_plot", cm_path)
            session.add_file("predictions", pred_path)

            # 7. Save session data
            session_path = session.save()
            print(f"Training session data saved to {session_path}")

            print(f"\nTraining completed after {completed_epochs} epochs.")
            print(f"Best model saved with validation accuracy {best_val_acc*100:.2f}%")

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
        # Create a session tracker
        session = SessionTracker(self.model, "crossval")

        # Save the initial model state to reset for each fold
        initial_state = self.model.state_dict().copy()

        # Initialize list to store results for each fold
        fold_results = []

        # Create k folds
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        indices = np.arange(len(dataset))

        # Track all confusion matrices
        all_confusion_matrices = []

        # Run training and validation for each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\nFold {fold+1}/{k_folds}")

            # Reset model to initial state
            self.model.load_state_dict(initial_state)

            # Reset optimizer and early stopping
            self.optimizer = optim.SGD(self.model.parameters(), **conf.OPTIMIZER)
            self.scheduler = ReduceLROnPlateau(self.optimizer, **conf.SCHEDULER)
            self.early_stopping = EarlyStopping(
                patience=conf.TRAIN['early_stopping_patience'],
                delta=conf.TRAIN['early_stopping_min_delta'],
                verbose=True
            )

            # Create data loaders for this fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, shuffle=True, **conf.DATALOADER)
            val_loader = DataLoader(val_subset, shuffle=False, **conf.DATALOADER)

            # Train the model for this fold (without saving weights)
            history = self.train(train_loader, val_loader, should_save=False)

            # Find best validation accuracy
            best_epoch = np.argmin(history['val_losses'])
            best_val_loss = history['val_losses'][best_epoch]
            best_val_acc = history['val_accs'][best_epoch] / 100.0  # Convert percentage to decimal

            # Get the best model state from training and apply it
            if hasattr(self.model, 'best_state_dict') and self.model.best_state_dict is not None:
                self.model.load_state_dict(self.model.best_state_dict)
                print(f"Loaded best model state from epoch {best_epoch + 1}")
            else:
                print("WARNING: No best model state found. Using final model state.")

            # Compute confusion matrix for best model
            self.model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for inputs, clean_inputs, targets in val_loader:
                    # Use clean inputs for validation, not augmented ones
                    clean_inputs = clean_inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(clean_inputs)
                    _, predicted = outputs.max(1)

                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            cm = confusion_matrix(y_true, y_pred)
            all_confusion_matrices.append(cm)

            # Save fold results
            fold_results.append({
                'fold': fold + 1,
                'history': history,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch + 1,  # Convert to 1-indexed
                'confusion_matrix': cm
            })

        # Calculate average validation accuracy
        best_val_accs = [result['best_val_acc'] for result in fold_results]
        avg_val_acc = np.mean(best_val_accs)
        std_val_acc = np.std(best_val_accs)

        print(f"\nCross-validation results:")
        for result in fold_results:
            print(f"Fold {result['fold']}: Best validation accuracy = {result['best_val_acc']*100:.2f}%")

        print(f"\nAverage validation accuracy: {avg_val_acc*100:.2f}% ± {std_val_acc*100:.2f}%")

        # Plot average training history
        history_path, avg_history = plot_crossval_history(fold_results, model=self.model)

        # Plot confusion matrices
        cm_path = plot_crossval_confusion_matrices(fold_results, model=self.model)

        # Update session tracker
        session.add_metrics({
            "num_folds": k_folds,
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

        session.add_file("history_plot", history_path)
        session.add_file("confusion_matrix_plot", cm_path)

        # Save session data
        session_path = session.save()
        print(f"Cross-validation session data saved to {session_path}")

        return fold_results 
