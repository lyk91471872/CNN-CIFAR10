import os
import json
import glob
import datetime
from typing import Optional, Dict, List, Any

# Define paths - avoid importing from config to prevent circular imports
TRACKING_DIR = 'tracking'
WEIGHTS_DIR = 'weights'
GRAPHS_DIR = 'graphs' 
RESULTS_DIR = 'results'

# Make sure these directories exist
os.makedirs(TRACKING_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_session_filename(model, epoch=None, accuracy=None, prefix=None, extension=None, directory=None):
    """
    Get a filename for a session-based artifact.
    
    Args:
        model: The model being trained
        epoch: Number of epochs trained
        accuracy: Validation accuracy
        prefix: Optional prefix to add to the filename
        extension: File extension (without dot)
        directory: Directory to save the file in
        
    Returns:
        Full path to the file
    """
    # Get base name
    model_name = model.__class__.__name__
    
    # Generate timestamp part
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Build filename
    parts = []
    if prefix:
        parts.append(prefix)
        
    parts.append(model_name)
    
    if epoch is not None:
        parts.append(f"E{epoch}")
        
    if accuracy is not None:
        parts.append(f"A{int(100*accuracy)}")
        
    parts.append(timestamp)
    
    filename = "_".join(parts)
    if extension:
        filename = f"{filename}.{extension}"
        
    # Return full path if directory provided
    if directory:
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    
    return filename


class SessionTracker:
    """Tracks training and cross-validation sessions, saving data to JSON files."""
    
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
        """Load a session from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        session = SessionTracker.__new__(SessionTracker)
        session.data = data
        session.model_name = data["model_name"]
        session.session_type = data["session_type"]
        session.timestamp = data["timestamp"]
        session.session_id = os.path.splitext(os.path.basename(filepath))[0]
        
        return session
    
    @staticmethod
    def list_sessions(model_name=None, session_type=None, limit=10):
        """List all sessions, optionally filtered by model and session type."""
        # Create tracking directory if it doesn't exist
        os.makedirs(TRACKING_DIR, exist_ok=True)
        
        # Find all JSON files in tracking directory
        json_files = glob.glob(os.path.join(TRACKING_DIR, "*.json"))
        
        # Load each file and filter
        sessions = []
        for filepath in json_files:
            try:
                session = SessionTracker.load(filepath)
                
                # Filter by model_name if specified
                if model_name and session.model_name != model_name:
                    continue
                    
                # Filter by session_type if specified
                if session_type and session.session_type != session_type:
                    continue
                    
                sessions.append(session)
            except Exception as e:
                print(f"Error loading session from {filepath}: {e}")
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda s: s.timestamp, reverse=True)
        
        # Limit number of results
        if limit:
            sessions = sessions[:limit]
            
        return sessions 