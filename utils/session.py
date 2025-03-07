import os
import json
import glob
import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

# Define paths - avoid importing from config to prevent circular imports
ROOT_DIR = Path(__file__).resolve().parent.parent
TRACKING_DIR = os.path.join(ROOT_DIR, 'tracking')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
GRAPHS_DIR = os.path.join(ROOT_DIR, 'graphs')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')

# Make sure these directories exist
os.makedirs(TRACKING_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def get_session_filename(model, epoch=None, accuracy=None, prefix=None, extension=None, directory=None):
    """
    Generate a consistent filename for all artifacts from a training/validation session.
    
    Args:
        model: The model object or name
        epoch: Number of epochs trained
        accuracy: Validation accuracy (0-1 range)
        prefix: Optional prefix to add to the filename
        extension: File extension (without dot)
        directory: Optional directory path
        
    Returns:
        The full path to the file
    """
    # Get model name
    if hasattr(model, '__class__'):
        model_name = model.__class__.__name__
    else:
        # Handle case where model is a string
        model_name = str(model).split('(')[0]
    
    # Format accuracy if provided
    acc_str = ""
    if accuracy is not None:
        acc_str = f"_A{int(100 * accuracy)}"
    
    # Format epoch if provided
    epoch_str = ""
    if epoch is not None:
        epoch_str = f"_E{epoch}"
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Build filename
    filename = f"{model_name}{epoch_str}{acc_str}_{timestamp}"
    
    # Add prefix if provided
    if prefix:
        filename = f"{prefix}_{filename}"
    
    # Add extension if provided
    if extension:
        filename = f"{filename}.{extension}"
    
    # Return full path if directory is provided
    if directory:
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    
    return filename


class SessionTracker:
    """Class to handle tracking training/validation sessions with JSON."""
    
    def __init__(self, model, session_type="training"):
        """Initialize a new session tracker.
        
        Args:
            model: The model being trained/validated
            session_type: Either "training" or "crossval"
        """
        self.model_name = model.__class__.__name__
        self.session_type = session_type
        self.timestamp = datetime.datetime.now().isoformat()
        self.session_id = f"{self.model_name}_{session_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize data structure
        self.data = {
            "model_name": self.model_name,
            "session_type": session_type,
            "timestamp": self.timestamp,
            "config": {
                "model_params": model.get_config() if hasattr(model, 'get_config') else {}
            },
            "metrics": {},
            "files": {}
        }
        
        # Import config values only when needed, not at module level
        try:
            # Import only when needed to avoid circular imports
            import config as conf
            # Add configuration details if available
            self.data["config"].update({
                "optimizer": getattr(conf, 'OPTIMIZER', {}),
                "scheduler": getattr(conf, 'SCHEDULER', {}),
                "training": getattr(conf, 'TRAIN', {})
            })
        except ImportError:
            # If config can't be imported, proceed without those values
            pass
    
    def add_metrics(self, metrics):
        """Add metrics to the session data."""
        self.data["metrics"] = {**self.data.get("metrics", {}), **metrics}
        return self
    
    def add_file(self, file_type, file_path):
        """Add a file reference to the session data."""
        self.data["files"][file_type] = file_path
        return self
    
    def save(self):
        """Save the session data to a JSON file."""
        filename = f"{self.session_id}.json"
        filepath = os.path.join(TRACKING_DIR, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
            
        print(f"Session tracking data saved to {filepath}")
        return filepath
    
    @staticmethod
    def load(filepath):
        """Load session data from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create a new SessionTracker and fill it with loaded data
        tracker = SessionTracker.__new__(SessionTracker)
        tracker.data = data
        tracker.model_name = data["model_name"]
        tracker.session_type = data["session_type"]
        tracker.timestamp = data["timestamp"]
        tracker.session_id = os.path.splitext(os.path.basename(filepath))[0]
        
        return tracker
    
    @staticmethod
    def list_sessions(model_name=None, session_type=None, limit=10):
        """List available sessions, optionally filtered by model name and/or type."""
        sessions = []
        
        for filename in os.listdir(TRACKING_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(TRACKING_DIR, filename)
                try:
                    tracker = SessionTracker.load(filepath)
                    
                    # Apply filters
                    if model_name and tracker.data["model_name"] != model_name:
                        continue
                    if session_type and tracker.data["session_type"] != session_type:
                        continue
                    
                    sessions.append(tracker)
                except Exception as e:
                    print(f"Error loading session from {filename}: {e}")
        
        # Sort by timestamp (newest first) and limit results
        sessions.sort(key=lambda x: x.data["timestamp"], reverse=True)
        return sessions[:limit]