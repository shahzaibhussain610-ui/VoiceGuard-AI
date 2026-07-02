"""
VoiceGuard AI - Configuration Management
Centralized configuration for the deepfake detection system
"""

from typing import Dict, Any, Optional
import os
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    target_sr: int = 16000
    min_duration: float = 1.0
    max_duration: float = 30.0
    mono: bool = True
    normalize: bool = True
    trim_silence: bool = True
    silence_threshold_db: int = 20


@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    n_chroma: int = 12
    n_mels: int = 128
    normalize: bool = True
    normalization_method: str = "zscore"  # 'zscore', 'minmax', 'robust'


@dataclass
class DetectionConfig:
    """Detection model configuration"""
    model_type: str = "ensemble"  # 'random_forest', 'svm', 'gradient_boosting', 'ensemble'
    test_size: float = 0.2
    random_state: int = 42
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 20
    rf_min_samples_split: int = 10
    rf_min_samples_leaf: int = 5
    
    # SVM parameters
    svm_kernel: str = "rbf"
    svm_c: float = 1.0
    svm_gamma: str = "scale"
    
    # Gradient Boosting parameters
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 5


@dataclass
class ReportConfig:
    """Report generation configuration"""
    output_dir: str = "output"
    export_formats: list = field(default_factory=lambda: ["json", "text", "html"])
    include_visualizations: bool = False
    detailed: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reporting: ReportConfig = field(default_factory=ReportConfig)
    
    # General settings
    verbose: bool = True
    log_level: str = "INFO"
    cache_dir: str = "cache"
    model_dir: str = "models"


class ConfigManager:
    """
    Configuration manager for VoiceGuard AI.
    
    This class provides:
    - Centralized configuration management
    - Configuration file loading/saving
    - Environment variable support
    - Configuration validation
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> PipelineConfig:
        """
        Load configuration from file or use defaults.
        
        Returns:
            PipelineConfig object
        """
        if self.config_file and os.path.exists(self.config_file):
            return self._load_from_file(self.config_file)
        
        # Check for environment variables
        config = self._load_from_env()
        
        return config
    
    def _load_from_file(self, file_path: str) -> PipelineConfig:
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            PipelineConfig object
        """
        import json
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return self._dict_to_config(config_dict)
    
    def _load_from_env(self) -> PipelineConfig:
        """
        Load configuration from environment variables.
        
        Returns:
            PipelineConfig object
        """
        # Start with defaults
        config = PipelineConfig()
        
        # Override with environment variables if present
        if os.getenv("VOICEGUARD_TARGET_SR"):
            config.audio.target_sr = int(os.getenv("VOICEGUARD_TARGET_SR"))
        
        if os.getenv("VOICEGUARD_MODEL_TYPE"):
            config.detection.model_type = os.getenv("VOICEGUARD_MODEL_TYPE")
        
        if os.getenv("VOICEGUARD_OUTPUT_DIR"):
            config.reporting.output_dir = os.getenv("VOICEGUARD_OUTPUT_DIR")
        
        if os.getenv("VOICEGUARD_VERBOSE"):
            config.verbose = os.getenv("VOICEGUARD_VERBOSE").lower() == "true"
        
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Convert dictionary to PipelineConfig.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            PipelineConfig object
        """
        # Extract nested configurations
        audio_dict = config_dict.get("audio", {})
        features_dict = config_dict.get("features", {})
        detection_dict = config_dict.get("detection", {})
        reporting_dict = config_dict.get("reporting", {})
        
        # Create config objects
        audio_config = AudioConfig(**audio_dict)
        features_config = FeatureConfig(**features_dict)
        detection_config = DetectionConfig(**detection_dict)
        reporting_config = ReportConfig(**reporting_dict)
        
        # Extract general settings
        general_dict = {k: v for k, v in config_dict.items() 
                       if k not in ["audio", "features", "detection", "reporting"]}
        
        return PipelineConfig(
            audio=audio_config,
            features=features_config,
            detection=detection_config,
            reporting=reporting_config,
            **general_dict
        )
    
    def _config_to_dict(self, config: PipelineConfig) -> Dict[str, Any]:
        """
        Convert PipelineConfig to dictionary.
        
        Args:
            config: PipelineConfig object
            
        Returns:
            Configuration dictionary
        """
        from dataclasses import asdict
        
        return asdict(config)
    
    def save_config(self, file_path: str):
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
        """
        import json
        
        config_dict = self._config_to_dict(self.config)
        
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_config(self) -> PipelineConfig:
        """
        Get current configuration.
        
        Returns:
            PipelineConfig object
        """
        return self.config
    
    def update_config(self, **kwargs):
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown config key '{key}'")
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration values.
        
        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []
        
        # Validate audio config
        if self.config.audio.target_sr <= 0:
            errors.append("target_sr must be positive")
        
        if self.config.audio.min_duration <= 0:
            errors.append("min_duration must be positive")
        
        if self.config.audio.max_duration <= self.config.audio.min_duration:
            errors.append("max_duration must be greater than min_duration")
        
        # Validate feature config
        if self.config.features.n_mfcc <= 0:
            errors.append("n_mfcc must be positive")
        
        if self.config.features.n_fft <= 0:
            errors.append("n_fft must be positive")
        
        if self.config.features.hop_length <= 0:
            errors.append("hop_length must be positive")
        
        # Validate detection config
        valid_model_types = ['random_forest', 'svm', 'gradient_boosting', 'ensemble']
        if self.config.detection.model_type not in valid_model_types:
            errors.append(f"model_type must be one of {valid_model_types}")
        
        if not 0 < self.config.detection.test_size < 1:
            errors.append("test_size must be between 0 and 1")
        
        # Validate report config
        valid_formats = ['json', 'text', 'html']
        for fmt in self.config.reporting.export_formats:
            if fmt not in valid_formats:
                warnings.append(f"Unknown export format: {fmt}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# Preset configurations
def get_fast_config() -> PipelineConfig:
    """
    Get configuration optimized for speed.
    
    Returns:
        PipelineConfig for fast processing
    """
    return PipelineConfig(
        audio=AudioConfig(target_sr=16000, min_duration=1.0, max_duration=10.0),
        features=FeatureConfig(n_mfcc=13, n_fft=1024, hop_length=256),
        detection=DetectionConfig(model_type="random_forest"),
        reporting=ReportConfig(export_formats=["json"])
    )


def get_accurate_config() -> PipelineConfig:
    """
    Get configuration optimized for accuracy.
    
    Returns:
        PipelineConfig for high accuracy
    """
    return PipelineConfig(
        audio=AudioConfig(target_sr=22050, min_duration=2.0, max_duration=30.0),
        features=FeatureConfig(n_mfcc=20, n_fft=2048, hop_length=512),
        detection=DetectionConfig(model_type="ensemble"),
        reporting=ReportConfig(export_formats=["json", "text", "html"], detailed=True)
    )


def get_default_config() -> PipelineConfig:
    """
    Get default balanced configuration.
    
    Returns:
        PipelineConfig with default settings
    """
    return PipelineConfig()


# Convenience function
def load_config(config_file: str) -> PipelineConfig:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        PipelineConfig object
    """
    manager = ConfigManager(config_file)
    return manager.get_config()


def create_default_config_file(output_path: str = "config.json"):
    """
    Create a default configuration file.
    
    Args:
        output_path: Path to save configuration file
    """
    config = get_default_config()
    manager = ConfigManager()
    manager.config = config
    manager.save_config(output_path)
    print(f"Default configuration saved to: {output_path}")