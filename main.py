from requirements import install_requirements

try:
    install_requirements()
except Exception as e:
    print(f"Error installing requirements: {e}")
    exit(1)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import argparse

import config as conf
from dataset import CIFAR10Dataset, CIFAR10TestDataset
from models import CustomResNet18, CustomEfficientNetV2_B0
from utils.pipeline import Pipeline
from utils.visualization import plot_training_history

def parse_args():
    parser = argparse.ArgumentParser(description='Train or cross-validate a model on CIFAR-10')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model on full dataset')
    parser.add_argument('-c', '--crossval', action='store_true', help='Run cross-validation')
    return parser.parse_args()

def main():
    """Main function to run the training or cross-validation."""
    args = parse_args()
    
    if not (args.train or args.crossval):
        print("Please specify either -t (train) or -c (crossval) mode")
        return
    
    dataset = CIFAR10Dataset(data_paths=conf.TRAIN_DATA_PATHS)
    # model = CustomResNet18()
    model = CustomEfficientNetV2_B0()
    pipeline = Pipeline(model)
    
    if args.crossval:
        print("\nStarting cross-validation...")
        fold_results = pipeline.cross_validate(dataset)
        print("\nCross-validation results:")
        for result in fold_results:
            print(f"Fold {result['fold']}: Best validation accuracy = {result['best_val_acc']:.2f}%")
    
    if args.train:
        print("\nTraining on full dataset...")
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, shuffle=True, **conf.DATALOADER)
        val_loader = DataLoader(val_dataset, shuffle=False, **conf.DATALOADER)
        
        history = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        plot_training_history(history)
        
        print("\nGenerating predictions...")
        model.load()
        test_dataset = CIFAR10TestDataset(conf.TEST_DATA_PATH)
        test_loader = DataLoader(test_dataset, shuffle=False, **conf.DATALOADER)
        predictions, indices = pipeline.predict(test_loader)
        
        submission = pd.DataFrame({'ID': indices, 'Label': predictions})
        submission.to_csv('prediction.csv', index=False)
        print("\nTest predictions saved to prediction.csv")

if __name__ == "__main__":
    main()
