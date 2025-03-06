import os
import sqlite3
import json
from datetime import datetime

from config import ROOT_DIR, DB_PATH

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
    
    conn.commit()
    conn.close()

def record_model_run(model, weights_file, config_dict):
    """Record a model run in the database.
    
    Args:
        model: The model instance or class
        weights_file: Path to the saved weights file
        config_dict: Dictionary containing model configuration
    
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
        'INSERT INTO model_runs (model_name, weights_file, timestamp, config) VALUES (?, ?, ?, ?)',
        (model_name, weights_file, timestamp, config_json)
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

# Initialize database when module is imported
init_db() 