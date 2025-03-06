import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import argparse
import os
import numpy as np
import json
import sqlite3
from datetime import datetime

import config as conf
from dataset import create_dataset
from utils.pipeline import Pipeline
from utils.visualization import plot_training_history, plot_crossval_history
from utils.db import record_prediction, get_model_run_by_weights, init_db

def record_crossval_results(model, fold_results, avg_history_path):
    """Record cross-validation results in the database.
    
    Args:
        model: The model instance
        fold_results: List of dictionaries containing fold results
        avg_history_path: Path to the saved average history plot
    
    Returns:
        int: The ID of the inserted record
    """
    init_db()
    conn = sqlite3.connect(conf.DB_PATH)
    cursor = conn.cursor()
    
    # Check if cross_validation_results table exists, create if not
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cross_validation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        num_folds INTEGER NOT NULL,
        avg_val_acc REAL NOT NULL,
        std_val_acc REAL NOT NULL,
        history_plot_path TEXT NOT NULL,
        fold_details TEXT NOT NULL,
        model_config TEXT NOT NULL
    )
    ''')
    
    # Extract relevant data
    model_name = str(model.__class__.__name__)
    timestamp = datetime.now().isoformat()
    num_folds = len(fold_results)
    
    # Calculate statistics
    best_val_accs = [result['best_val_acc'] for result in fold_results]
    avg_val_acc = np.mean(best_val_accs)
    std_val_acc = np.std(best_val_accs)
    
    # Model configuration
    model_config = {
        'model_name': model_name,
        'model_params': model.get_config() if hasattr(model, 'get_config') else {},
        'optimizer': conf.OPTIMIZER,
        'dataloader': conf.DATALOADER,
    }
    
    # Insert record
    cursor.execute(
        '''INSERT INTO cross_validation_results 
           (model_name, timestamp, num_folds, avg_val_acc, std_val_acc, 
            history_plot_path, fold_details, model_config) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            model_name, 
            timestamp, 
            num_folds, 
            float(avg_val_acc), 
            float(std_val_acc), 
            avg_history_path, 
            json.dumps([{k: v for k, v in r.items() if k != 'history'} for r in fold_results], default=str),
            json.dumps(model_config, default=str)
        )
    )
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return run_id

def parse_args():
    parser = argparse.ArgumentParser(description='Train or cross-validate a model on CIFAR-10')
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', help='Train the model on full dataset')
    group.add_argument('-c', '--crossval', action='store_true', help='Run cross-validation')
    group.add_argument('-p', '--pdf', action='store_true', help='Generate PDF of test images')
    group.add_argument('-d', '--benchmark', action='store_true', help='Run dataloader benchmark')
    group.add_argument('-n', '--normalize', action='store_true', help='Update normalization values')
    
    return parser.parse_args()

def main():
    """Main function to run the training or cross-validation."""
    args = parse_args()

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
        
        fold_results = pipeline.cross_validate(dataset)
        
        # Display results for each fold
        print("\nCross-validation results:")
        best_val_accs = []
        for result in fold_results:
            print(f"Fold {result['fold']}: Best validation accuracy = {result['best_val_acc']:.2f}%")
            best_val_accs.append(result['best_val_acc'])
        
        # Calculate and display average performance
        avg_val_acc = np.mean(best_val_accs)
        std_val_acc = np.std(best_val_accs)
        print(f"\nAverage validation accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
        
        # Plot average training history using the visualization module
        history_path = os.path.join(conf.GRAPHS_DIR, 'crossval_history.png')
        avg_history = plot_crossval_history(fold_results, save_path=history_path)
        
        # Record cross-validation results in the database
        try:
            run_id = record_crossval_results(model, fold_results, history_path)
            print(f"Cross-validation results recorded in database with ID {run_id}")
        except Exception as e:
            print(f"Warning: Failed to record cross-validation results in database: {e}")
        
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
        
        # Train the model
        history = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        plot_training_history(history)

        print("\nGenerating predictions...")
        # Load the best model
        model.load()
        
        test_dataset = create_dataset(data_source=conf.TEST_DATA_PATH, mode='test')
        test_loader = DataLoader(test_dataset, shuffle=False, **conf.DATALOADER)
        predictions, indices = pipeline.predict(test_loader)

        # Create a timestamped prediction file path
        prediction_file = conf.get_timestamped_filename(model, 'csv', conf.PREDICTIONS_DIR)
        
        submission = pd.DataFrame({'ID': indices, 'Label': predictions})
        submission.to_csv(prediction_file, index=False)
        print(f"\nTest predictions saved to {prediction_file}")
        
        # Record the prediction in the database
        try:
            # Get model run details from the database using the model's weight path
            if model.weight_path:
                run_info = get_model_run_by_weights(model.weight_path)
                if run_info:
                    record_prediction(run_info['id'], prediction_file)
                    print(f"Recorded prediction in database, linked to model run {run_info['id']}")
                else:
                    print("Warning: Could not find model run in database")
        except Exception as e:
            print(f"Warning: Failed to record prediction in database: {e}")
        return

if __name__ == "__main__":
    main()
