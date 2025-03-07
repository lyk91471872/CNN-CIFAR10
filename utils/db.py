import os
import sqlite3
import json
import numpy as np
from datetime import datetime

from config import ROOT_DIR, DB_PATH
import config as conf  # Import config for accessing optimizer and dataloader settings

def init_db():
    """Initialize the database with the required tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for model runs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        weights_file TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        epochs INTEGER,
        config TEXT NOT NULL
    )
    ''')
    
    # Create table for predictions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        prediction_file TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES model_runs (id)
    )
    ''')
    
    # Create table for cross-validation results
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
    
    conn.commit()
    conn.close()

def record_model_run(model, weights_file, config_dict, epochs=None):
    """Record a model run in the database.
    
    Args:
        model: The model instance or class
        weights_file: Path to the saved weights file
        config_dict: Dictionary containing model configuration
        epochs: Number of epochs the model was trained for
    
    Returns:
        int: The ID of the inserted record
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    model_name = str(model).split('(')[0]  # Extract model class name
    timestamp = datetime.now().isoformat()
    config_json = json.dumps(config_dict, default=str)
    
    cursor.execute(
        'INSERT INTO model_runs (model_name, weights_file, timestamp, epochs, config) VALUES (?, ?, ?, ?, ?)',
        (model_name, weights_file, timestamp, epochs, config_json)
    )
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return run_id

def record_prediction(run_id, prediction_file):
    """Record a prediction in the database.
    
    Args:
        run_id: The ID of the model run
        prediction_file: Path to the prediction CSV file
    
    Returns:
        int: The ID of the inserted record
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute(
        'INSERT INTO predictions (run_id, prediction_file, timestamp) VALUES (?, ?, ?)',
        (run_id, prediction_file, timestamp)
    )
    
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return pred_id

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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
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

def get_model_run_by_weights(weights_file):
    """Get model run info by weights file.
    
    Args:
        weights_file: Path to the weights file
    
    Returns:
        dict: Model run information
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM model_runs WHERE weights_file = ?', (weights_file,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_predictions_by_run_id(run_id):
    """Get predictions by model run ID.
    
    Args:
        run_id: The ID of the model run
    
    Returns:
        list: List of prediction dictionaries
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM predictions WHERE run_id = ?', (run_id,))
    rows = cursor.fetchall()
    
    conn.close()
    
    return [dict(row) for row in rows]

def get_crossval_results(model_name=None, limit=10):
    """Get cross-validation results, optionally filtered by model name.
    
    Args:
        model_name: Optional filter by model name
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        list: List of cross-validation result dictionaries
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if model_name:
        cursor.execute(
            'SELECT * FROM cross_validation_results WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?', 
            (model_name, limit)
        )
    else:
        cursor.execute(
            'SELECT * FROM cross_validation_results ORDER BY timestamp DESC LIMIT ?', 
            (limit,)
        )
    
    rows = cursor.fetchall()
    
    conn.close()
    
    return [dict(row) for row in rows]

# Initialize database when module is imported
init_db() 