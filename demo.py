"""
VoiceGuard AI - Demo and Test Script
Demonstrates the complete pipeline functionality and runs tests
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.input_agent import InputAgent
from app.agents.feature_agent import FeatureAgent
from app.agents.detection_agent import DetectionAgent
from app.agents.explain_agent import ExplainAgent
from app.agents.report_agent import ReportAgent
from app.utils.helper import Helper, print_section
from app.utils.audio_utils import AudioUtils
from app.utils import audio_utils


def test_individual_agents():
    """Test each agent individually"""
    print_section("Testing Individual Agents", "=", 80)
    
    # Create demo audio
    print("\n[1/6] Creating demo audio...")
    demo_audio, sr = audio_utils.create_demo_audio(duration=3.0, frequency=440.0, sr=16000)
    print(f"  [OK] Created {len(demo_audio)/sr:.2f}s audio at {sr} Hz")
    
    # Test Input Agent
    print("\n[2/6] Testing Input Agent...")
    try:
        input_agent = InputAgent()
        # Test with numpy array
        input_result = input_agent.process(demo_audio, sr=sr)
        if input_result["success"]:
            print(f"  [OK] Input Agent working - Duration: {input_result['metadata']['duration_seconds']:.2f}s")
        else:
            print(f"  [FAIL] Input Agent failed: {input_result.get('error')}")
    except Exception as e:
        print(f"  [FAIL] Input Agent error: {e}")
    
    # Test Feature Agent
    print("\n[3/6] Testing Feature Agent...")
    try:
        feature_agent = FeatureAgent()
        feature_result = feature_agent.process(demo_audio, normalize=True)
        if feature_result["success"]:
            print(f"  [OK] Feature Agent working - {feature_result['metadata']['feature_vector_size']} features")
        else:
            print(f"  [FAIL] Feature Agent failed: {feature_result.get('error')}")
    except Exception as e:
        print(f"  [FAIL] Feature Agent error: {e}")
    
    # Test Detection Agent
    print("\n[4/6] Testing Detection Agent...")
    try:
        detection_agent = DetectionAgent()
        # Note: Detection will fail because model is not trained
        detection_result = detection_agent.predict(feature_result["features"])
        if not detection_result["success"]:
            print(f"  [OK] Detection Agent working (expected error: {detection_result.get('error')})")
        else:
            print(f"  [OK] Detection Agent working - Prediction: {detection_result['prediction']}")
    except Exception as e:
        print(f"  [FAIL] Detection Agent error: {e}")
    
    # Test Explain Agent
    print("\n[5/6] Testing Explain Agent...")
    try:
        explain_agent = ExplainAgent()
        # Create mock detection result
        mock_detection = {
            "prediction": "fake",
            "confidence": 0.85,
            "probabilities": {"real": 0.15, "fake": 0.85}
        }
        explanation_result = explain_agent.process(mock_detection, feature_result["features"])
        if explanation_result["success"]:
            print(f"  [OK] Explain Agent working - Risk: {explanation_result['explanation']['risk_level']}")
        else:
            print(f"  [FAIL] Explain Agent failed: {explanation_result.get('error')}")
    except Exception as e:
        print(f"  [FAIL] Explain Agent error: {e}")
    
    # Test Report Agent
    print("\n[6/6] Testing Report Agent...")
    try:
        report_agent = ReportAgent(output_dir="test_reports")
        
        # Create mock data
        mock_input = {
            "success": True,
            "audio": "test_audio.wav",
            "metadata": {"duration_seconds": 3.0, "sample_rate": 16000},
            "validation": {"valid": True}
        }
        
        mock_explanation = {
            "success": True,
            "explanation": {
                "prediction": "fake",
                "confidence": 0.85,
                "risk_level": "high",
                "primary_reasons": ["Test reason"],
                "suspicious_features": {"count": 1, "features": [], "severity": "high"},
                "feature_analysis": {},
                "recommendations": ["Test recommendation"],
                "explanation_text": "Test explanation"
            }
        }
        
        report_result = report_agent.process(
            mock_input, 
            feature_result["features"],
            mock_detection,
            mock_explanation,
            export_formats=["json"]
        )
        
        if report_result["success"]:
            print(f"  [OK] Report Agent working - Report generated")
        else:
            print(f"  [FAIL] Report Agent failed: {report_result.get('error')}")
    except Exception as e:
        print(f"  [FAIL] Report Agent error: {e}")


def test_utility_functions():
    """Test utility functions"""
    print_section("Testing Utility Functions", "=", 80)
    
    # Test Helper class
    print("\n[1/3] Testing Helper utilities...")
    try:
        # Test unique ID generation
        uid = Helper.generate_unique_id()
        print(f"  [OK] Generated unique ID: {uid}")
        
        # Test directory creation
        test_dir = Helper.ensure_directory_exists("test_dir")
        print(f"  [OK] Created directory: {test_dir}")
        
        # Test file size formatting
        size_str = Helper.format_file_size(1024000)
        print(f"  [OK] Formatted file size: {size_str}")
        
        # Test duration formatting
        dur_str = Helper.format_duration(125.5)
        print(f"  [OK] Formatted duration: {dur_str}")
        
    except Exception as e:
        print(f"  [FAIL] Helper utilities error: {e}")
    
    # Test AudioUtils class
    print("\n[2/3] Testing AudioUtils...")
    try:
        utils = AudioUtils()
        
        # Create test audio
        test_audio = np.random.randn(16000)
        
        # Test normalization
        normalized = utils.normalize_audio(test_audio)
        print(f"  [OK] Audio normalization working")
        
        # Test statistics
        stats = utils.get_audio_statistics(test_audio)
        print(f"  [OK] Audio statistics: {stats['length']} samples, RMS={stats['rms']:.4f}")
        
        # Test quality detection
        quality = utils.detect_audio_issues(test_audio, 16000)
        print(f"  [OK] Quality detection: {quality['issues_detected']} issues found")
        
    except Exception as e:
        print(f"  [FAIL] AudioUtils error: {e}")
    
    # Test configuration
    print("\n[3/3] Testing Configuration...")
    try:
        from config import ConfigManager, get_default_config
        
        # Test default config
        config = get_default_config()
        print(f"  [OK] Default config loaded")
        print(f"    - Target SR: {config.audio.target_sr}")
        print(f"    - Model type: {config.detection.model_type}")
        
        # Test config manager
        manager = ConfigManager()
        validation = manager.validate_config()
        print(f"  [OK] Config validation: {'Valid' if validation['valid'] else 'Invalid'}")
        
    except Exception as e:
        print(f"  [FAIL] Configuration error: {e}")


def test_model_training():
    """Test model training with synthetic data"""
    print_section("Testing Model Training", "=", 80)
    
    try:
        from app.models.model_manager import ModelManager
        
        print("\nGenerating synthetic training data...")
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        
        # Create synthetic features
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        print(f"  [OK] Generated {n_samples} samples with {n_features} features")
        print(f"    Real samples: {np.sum(y == 0)}, Fake samples: {np.sum(y == 1)}")
        
        # Train model
        print("\nTraining model...")
        manager = ModelManager(model_dir="test_models")
        manager.initialize_models(['random_forest', 'svm'])
        
        training_results = manager.train(X, y, test_size=0.2, cross_validate=False)
        
        if training_results["success"]:
            print(f"  [OK] Training successful!")
            print(f"    Models trained: {', '.join(training_results['metrics']['models_trained'])}")
            
            for model_name, scores in training_results["model_scores"].items():
                print(f"    {model_name}: Accuracy={scores['accuracy']:.3f}, F1={scores['f1_score']:.3f}")
            
            # Test prediction
            print("\nTesting prediction...")
            test_sample = np.random.randn(n_features)
            pred_result = manager.predict(test_sample)
            
            if pred_result["success"]:
                print(f"  [OK] Prediction: {pred_result['prediction']} ({pred_result['confidence']*100:.1f}%)")
            
            # Test saving
            print("\nTesting model save/load...")
            model_path = "test_models/test_model.joblib"
            manager.save_model(model_path, include_history=False)
            print(f"  [OK] Model saved to {model_path}")
            
            # Test loading
            manager2 = ModelManager()
            load_info = manager2.load_model(model_path)
            print(f"  [OK] Model loaded: {load_info['models']}")
            
        else:
            print(f"  [FAIL] Training failed: {training_results.get('error')}")
    
    except Exception as e:
        print(f"  [FAIL] Model training error: {e}")


def test_complete_pipeline():
    """Test the complete pipeline"""
    print_section("Testing Complete Pipeline", "=", 80)
    
    try:
        from main import VoiceGuardPipeline
        
        print("\nInitializing pipeline...")
        pipeline = VoiceGuardPipeline(
            output_dir="test_output",
            config={
                "audio": {"target_sr": 16000, "min_duration": 1.0, "max_duration": 30.0},
                "features": {"n_mfcc": 13, "n_fft": 2048, "hop_length": 512},
                "detection": {"model_type": "ensemble"},
                "reporting": {"export_formats": ["json"]}
            }
        )
        print("  [OK] Pipeline initialized")
        
        # Test with numpy array
        print("\nAnalyzing demo audio...")
        demo_audio, demo_sr = audio_utils.create_demo_audio(duration=3.0, frequency=440.0, sr=16000)
        
        result = pipeline.analyze_audio_array(demo_audio, demo_sr, save_report=True)
        
        if result["success"]:
            print(f"\n  [OK] Pipeline test successful!")
            print(f"    Prediction: {result['prediction'].upper()}")
            print(f"    Confidence: {result['confidence']*100:.1f}%")
            print(f"    Risk Level: {result['risk_level'].upper()}")
        else:
            print(f"\n  [FAIL] Pipeline test failed: {result.get('error')}")
    
    except Exception as e:
        print(f"  [FAIL] Pipeline test error: {e}")


def run_all_tests():
    """Run all tests"""
    print(f"\n{'='*80}")
    print("  VoiceGuard AI - Test Suite")
    print(f"{'='*80}\n")
    
    # Run tests
    test_individual_agents()
    test_utility_functions()
    test_model_training()
    test_complete_pipeline()
    
    # Summary
    print(f"\n{'='*80}")
    print("  Test Suite Complete")
    print(f"{'='*80}\n")
    
    print("All tests completed. Check output above for any errors.")


if __name__ == "__main__":
    run_all_tests()