"""
Helper Utilities - General helper functions for VoiceGuard AI
Provides common utility functions used across the system
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib


class Helper:
    """
    General helper utilities for VoiceGuard AI.
    
    This class provides:
    - File and path utilities
    - Data validation helpers
    - Format conversion utilities
    - Logging and timing utilities
    - Configuration management
    """
    
    @staticmethod
    def generate_unique_id(prefix: str = "VG") -> str:
        """
        Generate a unique identifier.
        
        Args:
            prefix: Prefix for the ID
            
        Returns:
            Unique ID string
        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:4]
        return f"{prefix}-{timestamp}-{random_suffix}"
    
    @staticmethod
    def ensure_directory_exists(directory: str) -> str:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            directory: Directory path
            
        Returns:
            Absolute path to directory
        """
        abs_path = os.path.abspath(directory)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get file extension in lowercase.
        
        Args:
            file_path: Path to file
            
        Returns:
            File extension with dot (e.g., '.wav')
        """
        return os.path.splitext(file_path)[1].lower()
    
    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        """
        Check if file is a supported audio format.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is audio format
        """
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma', '.aac']
        return Helper.get_file_extension(file_path) in audio_extensions
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in seconds to human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., "3m 45s")
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
    
    @staticmethod
    def format_file_size(bytes_size: int) -> str:
        """
        Format file size in bytes to human-readable string.
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted size string (e.g., "5.2 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Path to save file
            indent: JSON indentation
            
        Returns:
            True if successful
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory:
                Helper.ensure_directory_exists(directory)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data or None if error
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None
    
    @staticmethod
    def validate_dict_keys(data: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
        """
        Validate that dictionary contains required keys.
        
        Args:
            data: Dictionary to validate
            required_keys: List of required keys
            
        Returns:
            Validation result dictionary
        """
        missing_keys = [key for key in required_keys if key not in data]
        
        return {
            "valid": len(missing_keys) == 0,
            "missing_keys": missing_keys,
            "available_keys": list(data.keys())
        }
    
    @staticmethod
    def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two dictionaries recursively.
        
        Args:
            base: Base dictionary
            override: Dictionary with values to override
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Helper.merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp as ISO format string.
        
        Returns:
            Timestamp string
        """
        return datetime.now().isoformat()
    
    @staticmethod
    def calculate_percentage(part: float, total: float) -> float:
        """
        Calculate percentage safely.
        
        Args:
            part: Part value
            total: Total value
            
        Returns:
            Percentage (0-100)
        """
        if total == 0:
            return 0.0
        return (part / total) * 100
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """
        Clamp value between min and max.
        
        Args:
            value: Value to clamp
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clamped value
        """
        return max(min_val, min(value, max_val))
    
    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize score to 0-1 range.
        
        Args:
            score: Score to normalize
            min_val: Minimum possible value
            max_val: Maximum possible value
            
        Returns:
            Normalized score (0-1)
        """
        if max_val == min_val:
            return 0.5
        normalized = (score - min_val) / (max_val - min_val)
        return Helper.clamp(normalized, 0.0, 1.0)
    
    @staticmethod
    def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
        """
        Split list into batches.
        
        Args:
            items: List of items
            batch_size: Size of each batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches
    
    @staticmethod
    def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', 
                     sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            nested_dict: Nested dictionary
            parent_key: Parent key prefix
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(Helper.flatten_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)
    
    @staticmethod
    def timer(func):
        """
        Decorator to time function execution.
        
        Args:
            func: Function to time
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dictionary with system info
        """
        import platform
        
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine()
        }
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255 - len(ext)] + ext
        
        return filename


# Convenience functions
def save_results(results: Dict[str, Any], output_dir: str = "results", 
                 filename: Optional[str] = None) -> str:
    """
    Save analysis results to file.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        Path to saved file
    """
    Helper.ensure_directory_exists(output_dir)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_{timestamp}.json"
    
    file_path = os.path.join(output_dir, filename)
    Helper.save_json(results, file_path)
    
    return file_path


def print_section(title: str, char: str = "=", length: int = 80):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character to use for separator
        length: Length of separator
    """
    print(f"\n{char * length}")
    print(f"  {title}")
    print(f"{char * length}")


def validate_analysis_input(audio_data: Any, features: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Validate input for analysis pipeline.
    
    Args:
        audio_data: Audio data (numpy array or file path)
        features: Optional pre-extracted features
        
    Returns:
        Validation result
    """
    import numpy as np
    
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check audio data
    if audio_data is None:
        validation["valid"] = False
        validation["errors"].append("Audio data is None")
    elif isinstance(audio_data, str):
        # File path
        if not os.path.exists(audio_data):
            validation["valid"] = False
            validation["errors"].append(f"Audio file not found: {audio_data}")
    elif isinstance(audio_data, np.ndarray):
        # Numpy array
        if len(audio_data) == 0:
            validation["valid"] = False
            validation["errors"].append("Audio array is empty")
    else:
        validation["valid"] = False
        validation["errors"].append(f"Unsupported audio data type: {type(audio_data)}")
    
    # Check features if provided
    if features is not None:
        if not isinstance(features, dict):
            validation["valid"] = False
            validation["errors"].append("Features must be a dictionary")
        elif len(features) == 0:
            validation["warnings"].append("Features dictionary is empty")
    
    return validation