import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import argparse
import os
import numpy as np
import json
from datetime import datetime

import config as conf
from dataset import create_dataset
from utils.pipeline import Pipeline
from utils.visualization import plot_training_history

def parse_args():
    parser = argparse.ArgumentParser(description='Train or cross-validate a model on CIFAR-10')
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', help='Train the model on full dataset')
    group.add_argument('-c', '--crossval', action='store_true', help='Run cross-validation')
    group.add_argument('-p', '--pdf', action='store_true', help='Generate PDF of test images')
    group.add_argument('-tp', '--train-pdf', action='store_true', help='Generate PDF of training images (batch 1)')
    group.add_argument('-d', '--benchmark', action='store_true', help='Run dataloader benchmark')
    group.add_argument('-n', '--normalize', action='store_true', help='Update normalization values')
    group.add_argument('-l', '--list-sessions', action='store_true', help='List recent training/cross-validation sessions')
    
    # Optional arguments
    parser.add_argument('--model', type=str, help='Filter sessions by model name')
    parser.add_argument('--type', type=str, choices=['training', 'crossval'], help='Filter sessions by type')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of sessions to list')
    
    return parser.parse_args()

def main():
    """Main function to run the training or cross-validation."""
    args = parse_args()

    # Handle listing sessions (early return)
    if args.list_sessions:
        print("\nListing recent training/cross-validation sessions:")
        sessions = conf.SessionTracker.list_sessions(
            model_name=args.model,
            session_type=args.type,
            limit=args.limit
        )
        
        if not sessions:
            print("No sessions found matching your criteria.")
            return
        
        print(f"\nFound {len(sessions)} session(s):")
        for i, session in enumerate(sessions, 1):
            data = session.data
            print(f"\n{i}. {data['model_name']} - {data['session_type']} - {data['timestamp']}")
            
            if 'metrics' in data:
                metrics = data['metrics']
                if 'best_val_acc' in metrics:
                    print(f"   Accuracy: {metrics['best_val_acc']*100:.2f}%")
                if 'epochs' in metrics:
                    print(f"   Epochs: {metrics['epochs']}")
                if 'avg_val_acc' in metrics:
                    print(f"   Avg CV Accuracy: {metrics['avg_val_acc']*100:.2f}%")
            
            if 'files' in data:
                files = data['files']
                print(f"   Files:")
                for file_type, file_path in files.items():
                    print(f"     - {file_type}: {file_path}")
        
        return

    # Handle PDF generation (early return)
    if args.pdf:
        print("\nGenerating PDF of test images...")
        try:
            from scripts.testset2pdf import testset_to_pdf
            
            output_pdf_path = os.path.join(conf.SCRIPTS_OUTPUT_DIR, "test_images_raw.pdf")
            test_dataset = create_dataset(data_source=conf.TEST_DATA_PATH, mode='test', raw=True)
            testset_to_pdf(test_dataset, output_pdf_path, use_grayscale=False)
            
            print(f"PDF generated successfully: {output_pdf_path}")
        except Exception as e:
            print(f"Error generating PDF: {e}")
        return
    
    # Handle training PDF generation (early return)
    if args.train_pdf:
        print("\nGenerating PDF of training images (batch 1)...")
        try:
            from scripts.trainingset2pdf import CIFAR10TrainingDatasetRaw, trainingset_to_pdf
            
            training_batch_path = os.path.join(conf.DATA_DIR, 'data_batch_1')
            output_pdf_path = os.path.join(conf.SCRIPTS_OUTPUT_DIR, "training_images.pdf")
            
            training_dataset = CIFAR10TrainingDatasetRaw(training_batch_path)
            trainingset_to_pdf(training_dataset, output_pdf_path, use_grayscale=False)
            
            print(f"PDF generated successfully: {output_pdf_path}")
        except Exception as e:
            print(f"Error generating PDF: {e}")
        return

    # Handle benchmark (early return)
    if args.benchmark:
        print("\nRunning dataloader benchmark...")
        try:
            from scripts.dataloader_benchmark import main as benchmark_main
            benchmark_main()
        except Exception as e:
            print(f"Error running benchmark: {e}")
        return
    
    # Handle normalization (early return)
    if args.normalize:
        print("\nUpdating normalization values...")
        try:
            from scripts.update_normalization_values import main as normalize_main
            normalize_main()
        except Exception as e:
            print(f"Error updating normalization values: {e}")
        return

    # Handle cross-validation
    if args.crossval:
        print("\nStarting cross-validation...")
        dataset = create_dataset(data_source=conf.TRAIN_DATA_PATHS, mode='training')
        model = conf.get_model()()  # Get the model class and instantiate it
        pipeline = Pipeline(model)
        
        # Pipeline.cross_validate handles everything including plotting and tracking
        fold_results = pipeline.cross_validate(dataset)
        return

    # Handle training
    if args.train:
        print("\nTraining on full dataset...")
        dataset = create_dataset(data_source=conf.TRAIN_DATA_PATHS, mode='training')
        model = conf.get_model()()  # Get the model class and instantiate it
        pipeline = Pipeline(model)
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, shuffle=True, **conf.DATALOADER)
        val_loader = DataLoader(val_dataset, shuffle=False, **conf.DATALOADER)
        
        # Pipeline.train handles everything including plotting, prediction generation, and tracking
        history = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        return

if __name__ == "__main__":
    main()
