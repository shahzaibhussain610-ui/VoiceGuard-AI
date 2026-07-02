"""
VoiceGuard AI - Model Training Script
Train the detection model with real audio data for accurate deepfake detection

This script provides:
- Dataset preparation from labeled audio files
- Feature extraction and training
- Model evaluation and saving
- Support for custom datasets
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.input_agent import InputAgent
from app.agents.feature_agent import FeatureAgent
from app.agents.detection_agent import DetectionAgent
from app.utils.helper import Helper, print_section


class ModelTrainer:
    """
    Trainer class for VoiceGuard AI detection models.
    
    This class handles:
    - Loading and processing audio datasets
    - Feature extraction from audio files
    - Model training and validation
    - Performance evaluation
    - Model saving and metadata export
    """
    
    def __init__(self, 
                 model_dir: str = "models",
                 output_dir: str = "output",
                 config: Optional[Dict] = None):
        """
        Initialize the model trainer.
        
        Args:
            model_dir: Directory to save trained models
            output_dir: Directory for training reports
            config: Optional configuration dictionary
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.config = config or self._get_default_config()
        
        # Initialize agents
        self.input_agent = InputAgent(
            target_sr=self.config["audio"]["target_sr"],
            min_duration=self.config["audio"]["min_duration"],
            max_duration=self.config["audio"]["max_duration"]
        )
        
        self.feature_agent = FeatureAgent(
            sr=self.config["audio"]["target_sr"],
            n_mfcc=self.config["features"]["n_mfcc"],
            n_fft=self.config["features"]["n_fft"],
            hop_length=self.config["features"]["hop_length"]
        )
        
        self.detection_agent = DetectionAgent(
            model_type=self.config["detection"]["model_type"]
        )
        
        # Create directories
        Helper.ensure_directory_exists(model_dir)
        Helper.ensure_directory_exists(output_dir)
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            "audio": {
                "target_sr": 16000,
                "min_duration": 1.0,
                "max_duration": 30.0
            },
            "features": {
                "n_mfcc": 13,
                "n_fft": 2048,
                "hop_length": 512,
                "n_chroma": 12
            },
            "detection": {
                "model_type": "ensemble"
            },
            "training": {
                "test_size": 0.2,
                "cross_validate": True,
                "cv_folds": 5
            }
        }
    
    def prepare_dataset(self, 
                       real_audio_files: List[str],
                       fake_audio_files: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training dataset from audio files.
        
        Args:
            real_audio_files: List of paths to real audio files
            fake_audio_files: List of paths to fake/deepfake audio files
            
        Returns:
            Tuple of (X, y, failed_files) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Labels (0 for real, 1 for fake)
            - failed_files: List of files that failed to process
        """
        print_section("Preparing Dataset", "=", 80)
        
        X_list = []
        y_list = []
        failed_files = []
        file_names = []
        
        # Process real audio files (label = 0)
        print(f"\nProcessing {len(real_audio_files)} real audio files...")
        for i, audio_file in enumerate(real_audio_files, 1):
            print(f"  [{i}/{len(real_audio_files)}] {os.path.basename(audio_file)}...", end=" ")
            
            try:
                # Load and process audio
                input_result = self.input_agent.process(audio_file)
                
                if not input_result["success"]:
                    print(f"FAILED - {input_result.get('error')}")
                    failed_files.append((audio_file, input_result.get('error')))
                    continue
                
                # Extract features
                audio = input_result["audio"]
                feature_result = self.feature_agent.process(audio, normalize=True)
                
                if not feature_result["success"]:
                    print(f"FAILED - {feature_result.get('error')}")
                    failed_files.append((audio_file, feature_result.get('error')))
                    continue
                
                # Get feature vector
                feature_vector = self.feature_agent.prepare_feature_vector(
                    feature_result["features"]
                )
                
                X_list.append(feature_vector)
                y_list.append(0)  # Real = 0
                file_names.append(audio_file)
                print("OK")
                
            except Exception as e:
                print(f"ERROR - {str(e)}")
                failed_files.append((audio_file, str(e)))
        
        # Process fake audio files (label = 1)
        print(f"\nProcessing {len(fake_audio_files)} fake audio files...")
        for i, audio_file in enumerate(fake_audio_files, 1):
            print(f"  [{i}/{len(fake_audio_files)}] {os.path.basename(audio_file)}...", end=" ")
            
            try:
                # Load and process audio
                input_result = self.input_agent.process(audio_file)
                
                if not input_result["success"]:
                    print(f"FAILED - {input_result.get('error')}")
                    failed_files.append((audio_file, input_result.get('error')))
                    continue
                
                # Extract features
                audio = input_result["audio"]
                feature_result = self.feature_agent.process(audio, normalize=True)
                
                if not feature_result["success"]:
                    print(f"FAILED - {feature_result.get('error')}")
                    failed_files.append((audio_file, feature_result.get('error')))
                    continue
                
                # Get feature vector
                feature_vector = self.feature_agent.prepare_feature_vector(
                    feature_result["features"]
                )
                
                X_list.append(feature_vector)
                y_list.append(1)  # Fake = 1
                file_names.append(audio_file)
                print("OK")
                
            except Exception as e:
                print(f"ERROR - {str(e)}")
                failed_files.append((audio_file, str(e)))
        
        # Convert to numpy arrays
        if not X_list:
            raise RuntimeError("No valid audio files processed. Cannot train model.")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"\n{'='*80}")
        print(f"Dataset Summary:")
        print(f"  Total samples: {len(y)}")
        print(f"  Real samples: {np.sum(y == 0)}")
        print(f"  Fake samples: {np.sum(y == 1)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(f"  Failed files: {len(failed_files)}")
        print(f"{'='*80}\n")
        
        return X, y, failed_files
    
    def train_model(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    save_model: bool = True,
                    model_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the detection model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0 for real, 1 for fake)
            save_model: Whether to save the trained model
            model_filename: Optional custom filename for saved model
            
        Returns:
            Training results dictionary
        """
        print_section("Training Model", "=", 80)
        
        # Train the model
        training_config = self.config.get("training", {})
        results = self.detection_agent.train(
            X, y,
            test_size=training_config.get("test_size", 0.2),
            cross_validate=training_config.get("cross_validate", True),
            cv_folds=training_config.get("cv_folds", 5)
        )
        
        if results["success"]:
            print(f"\n✓ Training completed successfully!")
            print(f"\nModel Performance:")
            for model_name, scores in results["model_scores"].items():
                print(f"\n  {model_name.upper()}:")
                print(f"    Accuracy:  {scores['accuracy']:.4f}")
                print(f"    Precision: {scores['precision']:.4f}")
                print(f"    Recall:    {scores['recall']:.4f}")
                print(f"    F1 Score:  {scores['f1_score']:.4f}")
            
            # Save model if requested
            if save_model:
                if model_filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"voiceguard_model_{timestamp}.joblib"
                
                model_path = os.path.join(self.model_dir, model_filename)
                self.detection_agent.save_model(model_path)
                results["model_path"] = model_path
                
                # Also save as default model
                default_model_path = os.path.join(self.model_dir, "voiceguard_model.joblib")
                self.detection_agent.save_model(default_model_path)
                results["default_model_path"] = default_model_path
                
                # Export metadata
                metadata_path = os.path.join(
                    self.output_dir, 
                    f"training_metadata_{timestamp}.json"
                )
                self._export_training_metadata(results, metadata_path)
                results["metadata_path"] = metadata_path
        
        return results
    
    def _export_training_metadata(self, training_results: Dict, metadata_path: str):
        """Export training metadata to JSON file"""
        metadata = {
            "training_timestamp": datetime.now().isoformat(),
            "training_results": training_results,
            "config": self.config,
            "model_info": self.detection_agent.get_model_info()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\nTraining metadata exported to: {metadata_path}")
    
    def evaluate_model(self, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Evaluation results dictionary
        """
        print_section("Model Evaluation", "=", 80)
        
        if not self.detection_agent.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        results = self.detection_agent.evaluate_model(X_test, y_test)
        
        if results["success"]:
            print(f"\n✓ Evaluation completed!")
            print(f"\nTest Set Performance:")
            for model_name, scores in results["model_scores"].items():
                print(f"\n  {model_name.upper()}:")
                print(f"    Accuracy:  {scores['accuracy']:.4f}")
                print(f"    Precision: {scores['precision']:.4f}")
                print(f"    Recall:    {scores['recall']:.4f}")
                print(f"    F1 Score:  {scores['f1_score']:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Feature importance dictionary
        """
        return self.detection_agent.get_feature_importance(feature_names)


def create_dataset_from_directories(real_dir: str, 
                                    fake_dir: str,
                                    extensions: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Create dataset from directory structure.
    
    Args:
        real_dir: Directory containing real audio files
        fake_dir: Directory containing fake audio files
        extensions: List of file extensions to include (default: common audio formats)
        
    Returns:
        Tuple of (real_files, fake_files)
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    real_files = []
    fake_files = []
    
    # Get real audio files
    if os.path.exists(real_dir):
        for file in os.listdir(real_dir):
            if any(file.lower().endswith(ext) for ext in extensions):
                real_files.append(os.path.join(real_dir, file))
    
    # Get fake audio files
    if os.path.exists(fake_dir):
        for file in os.listdir(fake_dir):
            if any(file.lower().endswith(ext) for ext in extensions):
                fake_files.append(os.path.join(fake_dir, file))
    
    return real_files, fake_files


def main():
    """
    Main function demonstrating model training workflow.
    """
    print(f"\n{'='*80}")
    print("  VoiceGuard AI - Model Training")
    print(f"{'='*80}\n")
    
    # Configuration
    MODEL_DIR = "models"
    OUTPUT_DIR = "output"
    
    # Dataset paths - MODIFY THESE TO YOUR ACTUAL DATASET LOCATIONS
    REAL_AUDIO_DIR = "app/data/real"  # Directory with real audio files
    FAKE_AUDIO_DIR = "app/data/fake"  # Directory with fake/deepfake audio files
    
    # Alternative: Specify individual files
    # real_files = ["path/to/real1.wav", "path/to/real2.wav", ...]
    # fake_files = ["path/to/fake1.wav", "path/to/fake2.wav", ...]
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Load dataset
    print_section("Loading Dataset", "=", 80)
    
    # Try to load from directories
    if os.path.exists(REAL_AUDIO_DIR) and os.path.exists(FAKE_AUDIO_DIR):
        print(f"\nLoading from directories:")
        print(f"  Real audio: {REAL_AUDIO_DIR}")
        print(f"  Fake audio: {FAKE_AUDIO_DIR}")
        
        real_files, fake_files = create_dataset_from_directories(
            REAL_AUDIO_DIR, 
            FAKE_AUDIO_DIR
        )
    else:
        print(f"\nWARNING: Dataset directories not found!")
        print(f"  Expected: {REAL_AUDIO_DIR}")
        print(f"  Expected: {FAKE_AUDIO_DIR}")
        print(f"\nPlease either:")
        print(f"  1. Create these directories and add audio files")
        print(f"  2. Modify the paths in this script")
        print(f"  3. Use the interactive mode below\n")
        
        # Interactive mode
        use_demo = input("Would you like to use synthetic demo data? (y/n): ").lower()
        
        if use_demo == 'y':
            print("\nGenerating synthetic training data...")
            from main import create_demo_audio
            
            real_files = []
            fake_files = []
            
            # Generate synthetic real audio (natural-sounding)
            np.random.seed(42)
            for i in range(20):
                freq = np.random.randint(200, 800)
                duration = np.random.uniform(2.0, 5.0)
                audio, sr = create_demo_audio(duration=duration, frequency=freq, sr=16000)
                
                # Add some natural variation
                audio += 0.1 * np.random.randn(len(audio))
                filepath = f"demo_real_{i}.wav"
                import soundfile as sf
                sf.write(filepath, audio, sr)
                real_files.append(filepath)
            
            # Generate synthetic fake audio (more regular patterns)
            for i in range(20):
                freq = np.random.randint(200, 800)
                duration = np.random.uniform(2.0, 5.0)
                audio, sr = create_demo_audio(duration=duration, frequency=freq, sr=16000)
                
                # Add less variation (more synthetic)
                audio += 0.05 * np.random.randn(len(audio))
                filepath = f"demo_fake_{i}.wav"
                import soundfile as sf
                sf.write(filepath, audio, sr)
                fake_files.append(filepath)
            
            print(f"  Generated {len(real_files)} real samples")
            print(f"  Generated {len(fake_files)} fake samples")
        else:
            print("\nExiting. Please prepare your dataset first.")
            return
    
    # Check if we have files
    if not real_files and not fake_files:
        print("\nERROR: No audio files found. Cannot train model.")
        return
    
    if len(real_files) < 2 or len(fake_files) < 2:
        print("\nWARNING: Very small dataset. Results may not be reliable.")
        print(f"  Real files: {len(real_files)}")
        print(f"  Fake files: {len(fake_files)}")
        print("  Recommended: At least 10-20 samples per class")
    
    # Prepare dataset
    try:
        X, y, failed_files = trainer.prepare_dataset(real_files, fake_files)
    except Exception as e:
        print(f"\nERROR: Failed to prepare dataset: {e}")
        return
    
    if len(X) < 4:
        print("\nERROR: Too few valid samples for training. Need at least 4 samples.")
        return
    
    # Train model
    try:
        training_results = trainer.train_model(X, y, save_model=True)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        return
    
    if not training_results["success"]:
        print(f"\nERROR: Training failed: {training_results.get('error')}")
        return
    
    # Display results
    print_section("Training Complete!", "=", 80)
    print(f"\n✓ Model trained and saved successfully!")
    
    if "model_path" in training_results:
        print(f"\nModel saved to:")
        print(f"  {training_results['model_path']}")
    
    if "default_model_path" in training_results:
        print(f"\nDefault model saved to:")
        print(f"  {training_results['default_model_path']}")
    
    print(f"\nNext steps:")
    print(f"  1. Test the model on your audio files")
    print(f"  2. If accuracy is low, add more training data")
    print(f"  3. Consider fine-tuning thresholds in explain_agent.py")
    print(f"  4. Run: streamlit run app.py")
    
    # Cleanup demo files if created
    if 'use_demo' in locals() and use_demo == 'y':
        print(f"\nCleaning up demo files...")
        for f in real_files + fake_files:
            if os.path.exists(f):
                os.remove(f)
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()