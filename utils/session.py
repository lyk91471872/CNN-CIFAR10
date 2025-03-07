import os
import json
import glob
import datetime
from typing import Optional, Dict, List, Any, Union

# Import config
import config as conf

# Create necessary directories
def ensure_directories_exist():
    """Create all necessary directories for storing artifacts."""
    os.makedirs(conf.WEIGHTS_DIR, exist_ok=True)
    os.makedirs(conf.GRAPHS_DIR, exist_ok=True)
    os.makedirs(conf.SCRIPTS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(conf.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(conf.TRACKING_DIR, exist_ok=True)

# Create directories when module is imported
ensure_directories_exist()

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
    if hasattr(model, '__class__'):
        model_name = model.__class__.__name__
    else:
        # Handle case where model is a string
        model_name = str(model).split('(')[0]
    
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
    """Tracks training and cross-validation sessions, with built-in path handling."""
    
    def __init__(self, model, session_type="training"):
        """Initialize a new session tracker.
        
        Args:
            model: The model being trained/validated
            session_type: Either "training" or "crossval"
        """
        # Core session info
        self.model = model
        self.model_name = model.__class__.__name__
        self.session_type = session_type
        self.timestamp = datetime.datetime.now().isoformat()
        self.session_id = f"{self.model_name}_{session_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Tracking fields that will be updated during training
        self.epoch = 0
        self.accuracy = 0.0
        self.best_epoch = 0
        self.best_accuracy = 0.0
        self.confusion_matrix = None
        
        # Initialize data structure
        self.data = {
            "model_name": self.model_name,
            "session_type": session_type,
            "timestamp": self.timestamp,
            "config": {
                "optimizer": conf.OPTIMIZER,
                "scheduler": conf.SCHEDULER,
                "training": conf.TRAIN,
                "model_params": model.get_config() if hasattr(model, 'get_config') else {}
            },
            "metrics": {},
            "files": {}
        }
    
    def update_epoch(self, epoch: int, accuracy: float) -> None:
        """Update the current epoch and accuracy."""
        self.epoch = epoch
        self.accuracy = accuracy
        
        # Update metrics in data structure
        self.data["metrics"]["current_epoch"] = epoch
        self.data["metrics"]["current_accuracy"] = float(accuracy)
        
        # Update best values if current accuracy is better
        if accuracy > self.best_accuracy:
            self.best_epoch = epoch
            self.best_accuracy = accuracy
            self.data["metrics"]["best_epoch"] = epoch
            self.data["metrics"]["best_accuracy"] = float(accuracy)
    
    def update_confusion_matrix(self, confusion_matrix) -> None:
        """Update the confusion matrix for the current session."""
        self.confusion_matrix = confusion_matrix
    
    def get_filename(self, prefix: Optional[str] = None, extension: Optional[str] = None) -> str:
        """
        Get a consistent filename for session artifacts.
        
        Args:
            prefix: Optional prefix to add to the filename
            extension: File extension (without dot)
            
        Returns:
            Formatted filename string
        """
        # Build filename components
        parts = []
        if prefix:
            parts.append(prefix)
            
        parts.append(self.model_name)
        
        # Use best values for filenames
        if self.best_epoch > 0:
            parts.append(f"E{self.best_epoch}")
            
        if self.best_accuracy > 0:
            parts.append(f"A{int(100*self.best_accuracy)}")
            
        # Add timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        parts.append(timestamp)
        
        # Combine parts
        filename = "_".join(parts)
        
        # Add extension if provided
        if extension:
            filename = f"{filename}.{extension}"
            
        return filename
    
    def get_path(self, directory: str, prefix: Optional[str] = None, extension: Optional[str] = None) -> str:
        """
        Get full path for a session artifact.
        
        Args:
            directory: Directory to save in
            prefix: Optional prefix for filename
            extension: File extension
            
        Returns:
            Full path to the file
        """
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, self.get_filename(prefix, extension))
    
    @property
    def weights_path(self) -> str:
        """Get path for model weights file."""
        return self.get_path(conf.WEIGHTS_DIR, prefix=None, extension="pth")
    
    @property
    def history_plot_path(self) -> str:
        """Get path for training history plot."""
        return self.get_path(conf.GRAPHS_DIR, prefix="history", extension="png")
    
    @property
    def confusion_matrix_path(self) -> str:
        """Get path for confusion matrix plot."""
        return self.get_path(conf.GRAPHS_DIR, prefix="confusion", extension="png")
    
    @property
    def predictions_path(self) -> str:
        """Get path for prediction CSV file."""
        return self.get_path(conf.PREDICTIONS_DIR, prefix="predictions", extension="csv")
    
    @property
    def tracking_path(self) -> str:
        """Get path for session tracking JSON file."""
        filename = f"{self.session_id}.json"
        return os.path.join(conf.TRACKING_DIR, filename)
    
    def add_metrics(self, metrics: Dict) -> 'SessionTracker':
        """Add metrics to the session data."""
        self.data["metrics"] = {**self.data.get("metrics", {}), **metrics}
        return self
    
    def add_file(self, file_type: str, file_path: str) -> 'SessionTracker':
        """Add a file reference to the session data."""
        self.data["files"][file_type] = file_path
        return self
    
    def save(self) -> str:
        """Save the session data to a JSON file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.tracking_path), exist_ok=True)
        
        with open(self.tracking_path, 'w') as f:
            json.dump(self.data, f, indent=2)
            
        print(f"Session tracking data saved to {self.tracking_path}")
        return self.tracking_path
    
    @staticmethod
    def load(filepath: str) -> 'SessionTracker':
        """Load a session from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create a new instance without calling __init__
        session = SessionTracker.__new__(SessionTracker)
        
        # Set the basic attributes
        session.data = data
        session.model_name = data["model_name"]
        session.session_type = data["session_type"]
        session.timestamp = data["timestamp"]
        session.session_id = os.path.splitext(os.path.basename(filepath))[0]
        
        # Set the tracking attributes from metrics if available
        metrics = data.get("metrics", {})
        session.epoch = metrics.get("current_epoch", 0)
        session.accuracy = metrics.get("current_accuracy", 0.0)
        session.best_epoch = metrics.get("best_epoch", 0)
        session.best_accuracy = metrics.get("best_accuracy", 0.0)
        
        # Model will be None for loaded sessions
        session.model = None
        session.confusion_matrix = None
        
        return session
    
    @staticmethod
    def list_sessions(model_name: Optional[str] = None, 
                      session_type: Optional[str] = None, 
                      limit: Optional[int] = 10) -> List['SessionTracker']:
        """List all sessions, optionally filtered by model and session type."""
        # Create tracking directory if it doesn't exist
        os.makedirs(conf.TRACKING_DIR, exist_ok=True)
        
        sessions = []
        
        for filename in os.listdir(conf.TRACKING_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(conf.TRACKING_DIR, filename)
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
        sessions.sort(key=lambda s: s.data["timestamp"], reverse=True)
        
        # Limit number of results
        if limit:
            sessions = sessions[:limit]
                
        return sessions