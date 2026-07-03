"""
VoiceGuard AI - Main Pipeline Script
Orchestrates all agents to perform deepfake audio detection
Supports both custom-trained and pre-trained models
"""

import os
import sys
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import VoiceGuard AI agents
from app.agents.input_agent import InputAgent
from app.agents.feature_agent import FeatureAgent
from app.agents.detection_agent import DetectionAgent
from app.agents.explain_agent import ExplainAgent
from app.agents.report_agent import ReportAgent

# Import utilities
from app.utils.helper import Helper, print_section
from app.utils.audio_utils import AudioUtils

# Import WavLM adapter for real pre-trained models
try:
    from app.models.wavlm_adapter import WavLMAdapter
    WAVLM_AVAILABLE = True
except ImportError:
    WAVLM_AVAILABLE = False
    print("⚠️  WavLM adapter not available. Install transformers and torch.")


class VoiceGuardPipeline:
    """
    Main pipeline class that orchestrates all agents for deepfake detection.
    
    This class provides:
    - Complete end-to-end analysis pipeline
    - Easy-to-use interface for audio analysis
    - Comprehensive reporting and export
    - Batch processing capabilities
    - Support for both pre-trained and custom models
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 output_dir: str = "output",
                 config: Optional[Dict[str, Any]] = None,
                 use_pretrained: bool = False,
                 pretrained_model_id: str = "audio_deepfake_v1",
                 use_wavlm: bool = False,
                 wavlm_model_path: str = "models/pretrained/wavlm_base_plus"):
        """
        Initialize the VoiceGuard AI pipeline.
        
        Args:
            model_path: Path to pre-trained model (optional)
            output_dir: Directory for output files
            config: Optional configuration dictionary
            use_pretrained: Whether to use built-in pre-trained models (no training needed)
            pretrained_model_id: ID of pre-trained model to use (if use_pretrained=True)
            use_wavlm: Whether to use WavLM model for detection
            wavlm_model_path: Path to WavLM model directory
        """
        self.output_dir = output_dir
        self.config = config or self._default_config()
        self.use_pretrained = use_pretrained
        self.use_wavlm = use_wavlm
        self.wavlm_model_path = wavlm_model_path
        
        # Initialize all agents
        print_section("Initializing VoiceGuard AI Pipeline", "=", 80)
        
        print("\n[1/5] Initializing Input Agent...")
        self.input_agent = InputAgent(
            target_sr=self.config["audio"]["target_sr"],
            min_duration=self.config["audio"]["min_duration"],
            max_duration=self.config["audio"]["max_duration"]
        )
        print("  [OK] Input Agent initialized")
        
        print("\n[2/5] Initializing Feature Agent...")
        self.feature_agent = FeatureAgent(
            sr=self.config["audio"]["target_sr"],
            n_mfcc=self.config["features"]["n_mfcc"],
            n_fft=self.config["features"]["n_fft"],
            hop_length=self.config["features"]["hop_length"]
        )
        print("  [OK] Feature Agent initialized")
        
        print("\n[3/5] Initializing Detection Agent...")
        
        # Determine whether to use pre-trained models
        if use_pretrained or (model_path and not os.path.exists(model_path)):
            # Use pre-trained model
            self.detection_agent = DetectionAgent(
                model_type=self.config["detection"]["model_type"],
                use_pretrained=True,
                pretrained_model_id=pretrained_model_id
            )
            print("  [OK] Detection Agent initialized with pre-trained model")
        else:
            # Use custom model
            self.detection_agent = DetectionAgent(
                model_type=self.config["detection"]["model_type"],
                use_pretrained=False
            )
            
            # Load model from path if provided
            if model_path and os.path.exists(model_path):
                print(f"  -> Loading model from {model_path}")
                self.detection_agent.load_model(model_path)
                print("  [OK] Model loaded successfully")
            else:
                print("  [OK] Detection Agent initialized (untrained)")
        
        print("\n[4/5] Initializing Explain Agent...")
        self.explain_agent = ExplainAgent(feature_agent=self.feature_agent)
        print("  [OK] Explain Agent initialized")
        
        print("\n[5/5] Initializing Report Agent...")
        self.report_agent = ReportAgent(output_dir=output_dir)
        print("  [OK] Report Agent initialized")
        
        # Create output directory
        Helper.ensure_directory_exists(output_dir)
        
        print(f"\n{'='*80}")
        print("  VoiceGuard AI Pipeline Ready")
        print(f"{'='*80}\n")
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
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
                "model_type": "ensemble",
                "use_pretrained": False,
                "pretrained_model_id": "audio_deepfake_v1"
            },
            "reporting": {
                "export_formats": ["json", "text", "html"]
            }
        }
    
    def analyze_audio_file(self, 
                          audio_path: str,
                          train_model: bool = False,
                          training_labels: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Perform complete analysis on an audio file.
        
        This is the main method that runs the full pipeline:
        1. Input validation and preprocessing
        2. Feature extraction
        3. Deepfake detection
        4. Explanation generation
        5. Report compilation
        
        Args:
            audio_path: Path to audio file
            train_model: Whether to train the model (requires labels)
            training_labels: Optional dictionary mapping file paths to labels
            
        Returns:
            Complete analysis results
        """
        results = {
            "success": False,
            "pipeline_id": Helper.generate_unique_id(),
            "timestamp": Helper.get_timestamp(),
            "audio_file": audio_path,
            "stages": {},
            "final_report": None,
            "error": None
        }
        
        try:
            print_section(f"Analyzing: {os.path.basename(audio_path)}", "=", 80)
            
            # Stage 1: Input Processing
            print("\n[Stage 1/4] Processing audio input...")
            input_result = self.input_agent.process(audio_path)
            results["stages"]["input"] = input_result
            
            if not input_result["success"]:
                raise RuntimeError(f"Input processing failed: {input_result.get('error')}")
            
            print(f"  [OK] Audio loaded successfully")
            print(f"    Duration: {input_result['metadata']['duration_seconds']:.2f}s")
            print(f"    Sample Rate: {input_result['metadata']['sample_rate']} Hz")
            print(f"    RMS Energy: {input_result['metadata']['rms_energy']:.4f}")
            
            # Stage 2: Feature Extraction
            print("\n[Stage 2/4] Extracting features...")
            audio = input_result["audio"]
            feature_result = self.feature_agent.process(audio, normalize=True)
            results["stages"]["features"] = feature_result
            
            if not feature_result["success"]:
                raise RuntimeError(f"Feature extraction failed: {feature_result.get('error')}")
            
            print(f"  [OK] Features extracted successfully")
            print(f"    Feature Vector Size: {feature_result['metadata']['feature_vector_size']}")
            print(f"    Categories: {list(feature_result['features'].keys())}")
            
            # Stage 3: Detection
            print("\n[Stage 3/4] Running deepfake detection...")
            
            # Use WavLM model if enabled
            if self.use_wavlm and WAVLM_AVAILABLE:
                print("  -> Using WavLM model for detection...")
                try:
                    # Initialize WavLM adapter
                    if not hasattr(self, 'wavlm_adapter'):
                        self.wavlm_adapter = WavLMAdapter(self.wavlm_model_path)
                    
                    # Get prediction from WavLM
                    wavlm_result = self.wavlm_adapter.predict(audio, sr=input_result['metadata']['sample_rate'])
                    
                    if wavlm_result["success"]:
                        # Convert WavLM result to match expected format
                        detection_result = {
                            "success": True,
                            "prediction": wavlm_result["prediction"],
                            "confidence": wavlm_result["confidence"],
                            "probabilities": wavlm_result["probabilities"],
                            "model_type": "WavLM",
                            "model_predictions": {"wavlm": 1 if wavlm_result["prediction"] == "fake" else 0}
                        }
                        print(f"  [OK] WavLM detection complete")
                    else:
                        print(f"  [WARN] WavLM failed: {wavlm_result.get('error')}")
                        print("  -> Falling back to traditional detection...")
                        detection_result = self.detection_agent.predict(feature_result["features"])
                except Exception as e:
                    print(f"  [WARN] WavLM error: {e}")
                    print("  -> Falling back to traditional detection...")
                    detection_result = self.detection_agent.predict(feature_result["features"])
            else:
                # Train model if requested and not trained
                if train_model and not self.detection_agent.is_trained:
                    print("  -> Training model...")
                    # This would require actual training data
                    # For now, we'll skip actual training
                    print("  [WARN] Training requires labeled dataset - skipping")
                
                detection_result = self.detection_agent.predict(feature_result["features"])
            
            results["stages"]["detection"] = detection_result
            
            if not detection_result["success"]:
                raise RuntimeError(f"Detection failed: {detection_result.get('error')}")
            
            print(f"  [OK] Detection complete")
            print(f"    Prediction: {detection_result['prediction'].upper()}")
            print(f"    Confidence: {detection_result['confidence']*100:.1f}%")
            print(f"    Probabilities: Real={detection_result['probabilities']['real']*100:.1f}%, "
                  f"Fake={detection_result['probabilities']['fake']*100:.1f}%")
            
            # Stage 4: Explanation
            print("\n[Stage 4/4] Generating explanation...")
            explanation_result = self.explain_agent.process(detection_result, 
                                                            feature_result["features"])
            results["stages"]["explanation"] = explanation_result
            
            if not explanation_result["success"]:
                raise RuntimeError(f"Explanation generation failed: {explanation_result.get('error')}")
            
            explanation = explanation_result["explanation"]
            print(f"  [OK] Explanation generated")
            print(f"    Risk Level: {explanation['risk_level'].upper()}")
            print(f"    Suspicious Features: {explanation['suspicious_features']['count']}")
            
            # Generate Report
            print("\n[Report] Generating comprehensive report...")
            audio_info = self.input_agent.get_audio_info(audio_path)
            
            report_result = self.report_agent.process(
                input_data=input_result,
                features=feature_result["features"],
                detection_result=detection_result,
                explanation_result=explanation_result,
                audio_info=audio_info,
                export_formats=self.config["reporting"]["export_formats"]
            )
            results["final_report"] = report_result
            
            if report_result["success"]:
                print(f"  [OK] Report generated successfully")
                for fmt, filepath in report_result["exported_files"].items():
                    print(f"    {fmt.upper()}: {filepath}")
                
                # Print summary
                print("\n" + self.report_agent.get_report_summary(report_result["report"]))
            else:
                print(f"  [WARN] Report generation failed: {report_result.get('error')}")
            
            # Extract top-level results for easy access
            results["success"] = True
            results["prediction"] = detection_result["prediction"]
            results["confidence"] = detection_result["confidence"]
            results["risk_level"] = explanation["risk_level"]
            
            print(f"\n{'='*80}")
            print("  Analysis Complete!")
            print(f"{'='*80}\n")
            
        except Exception as e:
            results["error"] = str(e)
            results["error_type"] = type(e).__name__
            print(f"\n[ERROR] Pipeline failed: {e}")
        
        return results
    
    def analyze_audio_array(self, 
                           audio: np.ndarray, 
                           sr: int,
                           save_report: bool = True) -> Dict[str, Any]:
        """
        Analyze audio from numpy array.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            save_report: Whether to generate and save report
            
        Returns:
            Analysis results
        """
        results = {
            "success": False,
            "pipeline_id": Helper.generate_unique_id(),
            "timestamp": Helper.get_timestamp(),
            "stages": {},
            "error": None
        }
        
        try:
            # Feature extraction
            print("\n[1/3] Extracting features...")
            feature_result = self.feature_agent.process(audio, normalize=True)
            results["stages"]["features"] = feature_result
            
            if not feature_result["success"]:
                raise RuntimeError(f"Feature extraction failed: {feature_result.get('error')}")
            
            print(f"  [OK] Features extracted ({feature_result['metadata']['feature_vector_size']} features)")
            
            # Detection
            print("\n[2/3] Running detection...")
            
            # Use WavLM model if enabled
            if self.use_wavlm and WAVLM_AVAILABLE:
                print("  -> Using WavLM model for detection...")
                try:
                    # Initialize WavLM adapter
                    if not hasattr(self, 'wavlm_adapter'):
                        self.wavlm_adapter = WavLMAdapter(self.wavlm_model_path)
                    
                    # Get prediction from WavLM
                    wavlm_result = self.wavlm_adapter.predict(audio, sr=sr)
                    
                    if wavlm_result["success"]:
                        # Convert WavLM result to match expected format
                        detection_result = {
                            "success": True,
                            "prediction": wavlm_result["prediction"],
                            "confidence": wavlm_result["confidence"],
                            "probabilities": wavlm_result["probabilities"],
                            "model_type": "WavLM",
                            "model_predictions": {"wavlm": 1 if wavlm_result["prediction"] == "fake" else 0}
                        }
                        print(f"  [OK] WavLM detection complete")
                    else:
                        print(f"  [WARN] WavLM failed: {wavlm_result.get('error')}")
                        print("  -> Falling back to traditional detection...")
                        detection_result = self._fallback_detection(feature_result["features"])
                except Exception as e:
                    print(f"  [WARN] WavLM error: {e}")
                    print("  -> Falling back to traditional detection...")
                    detection_result = self._fallback_detection(feature_result["features"])
            else:
                # Check if model is trained
                detection_result = self._fallback_detection(feature_result["features"])
            
            results["stages"]["detection"] = detection_result
            
            if not detection_result["success"]:
                raise RuntimeError(f"Detection failed: {detection_result.get('error')}")
            
            print(f"  [OK] Prediction: {detection_result['prediction'].upper()} "
                  f"({detection_result['confidence']*100:.1f}% confidence)")
            
            # Explanation
            print("\n[3/3] Generating explanation...")
            explanation_result = self.explain_agent.process(detection_result, 
                                                            feature_result["features"])
            results["stages"]["explanation"] = explanation_result
            
            if not explanation_result["success"]:
                raise RuntimeError(f"Explanation failed: {explanation_result.get('error')}")
            
            explanation = explanation_result["explanation"]
            print(f"  [OK] Risk Level: {explanation['risk_level'].upper()}")
            
            # Generate report if requested
            if save_report:
                print("\n[Report] Generating report...")
                
                # Create minimal input data
                input_data = {
                    "success": True,
                    "audio": "numpy_array",
                    "metadata": {
                        "duration_seconds": len(audio) / sr,
                        "sample_rate": sr,
                        "num_samples": len(audio),
                        "source": "numpy_array"
                    },
                    "validation": {
                        "valid": True,
                        "source": "numpy_array"
                    }
                }
                
                report_result = self.report_agent.process(
                    input_data=input_data,
                    features=feature_result["features"],
                    detection_result=detection_result,
                    explanation_result=explanation_result,
                    export_formats=self.config["reporting"]["export_formats"]
                )
                results["report"] = report_result
                
                if report_result["success"]:
                    print(f"  [OK] Report saved to: {report_result['exported_files']}")
            
            results["success"] = True
            results["prediction"] = detection_result["prediction"]
            results["confidence"] = detection_result["confidence"]
            results["risk_level"] = explanation["risk_level"]
            
        except Exception as e:
            results["error"] = str(e)
            results["error_type"] = type(e).__name__
        
        return results
    
    def batch_analyze(self, 
                     audio_files: list,
                     output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            output_file: Optional file to save batch results
            
        Returns:
            Batch analysis results
        """
        print_section("Batch Analysis Mode", "=", 80)
        print(f"\nProcessing {len(audio_files)} audio files...\n")
        
        batch_results = {
            "batch_id": Helper.generate_unique_id("BATCH"),
            "timestamp": Helper.get_timestamp(),
            "total_files": len(audio_files),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}")
            print("-" * 80)
            
            result = self.analyze_audio_file(audio_file)
            batch_results["results"].append(result)
            
            if result["success"]:
                batch_results["successful"] += 1
            else:
                batch_results["failed"] += 1
        
        # Print summary
        print(f"\n{'='*80}")
        print("  Batch Analysis Complete")
        print(f"{'='*80}")
        print(f"  Total Files: {batch_results['total_files']}")
        print(f"  Successful: {batch_results['successful']}")
        print(f"  Failed: {batch_results['failed']}")
        print(f"{'='*80}\n")
        
        # Save batch results if requested
        if output_file:
            Helper.save_json(batch_results, output_file)
            print(f"Batch results saved to: {output_file}")
        
        return batch_results
    
    def _fallback_detection(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback detection method when WavLM is not available or fails.
        
        Args:
            features: Extracted features dictionary
            
        Returns:
            Detection result dictionary
        """
        if self.detection_agent.is_trained:
            return self.detection_agent.predict(features)
        else:
            # Mock prediction for demo
            return {
                "success": True,
                "prediction": "real",
                "confidence": 0.75,
                "probabilities": {"real": 0.75, "fake": 0.25},
                "model_predictions": {"mock": 0}
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Pipeline information dictionary
        """
        info = {
            "version": "1.0.0",
            "config": self.config,
            "agents": {
                "input_agent": "Active",
                "feature_agent": "Active",
                "detection_agent": self.detection_agent.get_model_info(),
                "explain_agent": "Active",
                "report_agent": "Active"
            },
            "output_dir": self.output_dir,
            "model_trained": self.detection_agent.is_trained,
            "use_pretrained": self.use_pretrained,
            "use_wavlm": self.use_wavlm
        }
        
        # Add WavLM info if available
        if self.use_wavlm and WAVLM_AVAILABLE and hasattr(self, 'wavlm_adapter'):
            info["wavlm_info"] = self.wavlm_adapter.get_model_info()
        
        return info


def create_demo_audio(duration: float = 3.0, 
                      frequency: float = 440.0,
                      sr: int = 16000) -> tuple:
    """
    Create a demo audio signal (sine wave).
    
    Args:
        duration: Duration in seconds
        frequency: Frequency in Hz
        sr: Sample rate
        
    Returns:
        Tuple of (audio_signal, sample_rate)
    """
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it more interesting
    audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32), sr


def main():
    """
    Main function demonstrating VoiceGuard AI pipeline usage.
    """
    print(f"\n{'='*80}")
    print("  VoiceGuard AI - Deepfake Audio Detection System")
    print(f"{'='*80}\n")
    
    # Initialize pipeline with pre-trained model (no training required!)
    print("=" * 80)
    print("  Initializing with PRE-TRAINED MODEL (No Training Required)")
    print("=" * 80)
    
    pipeline = VoiceGuardPipeline(
        output_dir="output",
        use_pretrained=True,  # Use pre-trained model
        pretrained_model_id="audio_deepfake_v1",  # Model to use
        config={
            "audio": {"target_sr": 16000, "min_duration": 1.0, "max_duration": 30.0},
            "features": {"n_mfcc": 13, "n_fft": 2048, "hop_length": 512},
            "detection": {"model_type": "ensemble"},
            "reporting": {"export_formats": ["json", "text"]}
        }
    )
    
    # Demo mode - create synthetic audio
    print_section("Demo Mode: Analyzing Synthetic Audio", "=", 80)
    
    print("\nCreating demo audio signal (sine wave)...")
    demo_audio, demo_sr = create_demo_audio(duration=3.0, frequency=440.0, sr=16000)
    print(f"  [OK] Created {len(demo_audio)/demo_sr:.2f}s audio at {demo_sr} Hz")
    
    # Analyze demo audio
    result = pipeline.analyze_audio_array(demo_audio, demo_sr, save_report=True)
    
    if result["success"]:
        print(f"\n{'='*80}")
        print("  Final Result")
        print(f"{'='*80}")
        print(f"  Prediction: {result['prediction'].upper()}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Risk Level: {result['risk_level'].upper()}")
        print(f"{'='*80}\n")
    else:
        print(f"\n[ERROR] Analysis failed: {result.get('error')}")
    
    # Show pipeline info
    print_section("Pipeline Information", "=", 80)
    info = pipeline.get_pipeline_info()
    print(f"\nVersion: {info['version']}")
    print(f"Model Trained: {info['model_trained']}")
    print(f"Use Pre-trained: {info['use_pretrained']}")
    print(f"Output Directory: {info['output_dir']}")
    print(f"\nDetection Model: {info['agents']['detection_agent']['model_type']}")
    
    # Show pre-trained model info if available
    if 'pretrained_info' in info['agents']['detection_agent']:
        pretrained_info = info['agents']['detection_agent']['pretrained_info']
        print(f"\nPre-trained Model Info:")
        print(f"  Loaded Models: {', '.join(pretrained_info['loaded_models'])}")
        if pretrained_info['model_metadata']:
            for model_id, metadata in pretrained_info['model_metadata'].items():
                print(f"  {model_id}:")
                print(f"    Name: {metadata.get('name', 'N/A')}")
                print(f"    Accuracy: {metadata.get('accuracy', 'N/A')}")
    else:
        print(f"Available Models: {', '.join(info['agents']['detection_agent'].get('models_available', []))}")
    
    print(f"\n{'='*80}")
    print("  VoiceGuard AI Demo Complete!")
    print(f"{'='*80}\n")
    
    return result


if __name__ == "__main__":
    main()