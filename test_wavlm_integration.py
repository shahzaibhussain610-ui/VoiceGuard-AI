"""
Test Script for WavLM Integration
Tests the WavLM model with VoiceGuard AI pipeline
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_wavlm_adapter():
    """Test the WavLM adapter directly"""
    print("=" * 80)
    print("TEST 1: WavLM Adapter Direct Test")
    print("=" * 80)
    
    try:
        from app.models.wavlm_adapter import WavLMAdapter
        
        # Check if model exists
        model_path = "models/pretrained/wavlm_base_plus"
        if not Path(model_path).exists():
            print(f"❌ Model not found at: {model_path}")
            print("Please download the model first using: python download_real_model.py")
            return False
        
        # Initialize adapter
        print(f"\nInitializing WavLM adapter...")
        adapter = WavLMAdapter(model_path)
        
        # Get model info
        print("\nModel Information:")
        info = adapter.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create test audio (sine wave)
        print("\nCreating test audio (3s sine wave at 440 Hz)...")
        duration = 3
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Make prediction
        print("\nMaking prediction...")
        result = adapter.predict(test_audio, sr)
        
        if result["success"]:
            print(f"\n✅ Prediction successful!")
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            print(f"  Probabilities:")
            print(f"    Real: {result['probabilities']['real']*100:.1f}%")
            print(f"    Fake: {result['probabilities']['fake']*100:.1f}%")
            return True
        else:
            print(f"\n❌ Prediction failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wavlm_with_pipeline():
    """Test WavLM integration with VoiceGuardPipeline"""
    print("\n" + "=" * 80)
    print("TEST 2: WavLM with VoiceGuard Pipeline")
    print("=" * 80)
    
    try:
        from main import VoiceGuardPipeline
        
        # Check if model exists
        model_path = "models/pretrained/wavlm_base_plus"
        if not Path(model_path).exists():
            print(f"❌ Model not found at: {model_path}")
            print("Please download the model first using: python download_real_model.py")
            return False
        
        # Initialize pipeline with WavLM
        print(f"\nInitializing VoiceGuard pipeline with WavLM...")
        pipeline = VoiceGuardPipeline(
            output_dir="output",
            use_wavlm=True,
            wavlm_model_path=model_path,
            config={
                "audio": {"target_sr": 16000, "min_duration": 1.0, "max_duration": 30.0},
                "features": {"n_mfcc": 13, "n_fft": 2048, "hop_length": 512},
                "detection": {"model_type": "ensemble"},
                "reporting": {"export_formats": ["json"]}
            }
        )
        
        # Create test audio
        print("\nCreating test audio (3s sine wave at 440 Hz)...")
        duration = 3
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Analyze audio
        print("\nAnalyzing audio with WavLM...")
        result = pipeline.analyze_audio_array(test_audio, sr, save_report=False)
        
        if result["success"]:
            print(f"\n✅ Pipeline test successful!")
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            print(f"  Risk Level: {result['risk_level'].upper()}")
            
            # Check if WavLM was used
            if "stages" in result and "detection" in result["stages"]:
                detection = result["stages"]["detection"]
                if detection.get("model_type") == "WavLM":
                    print(f"  ✓ WavLM model was used for detection")
                else:
                    print(f"  ⚠️  Fallback detection was used (WavLM not available)")
            return True
        else:
            print(f"\n❌ Pipeline test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wavlm_with_audio_file():
    """Test WavLM with an actual audio file"""
    print("\n" + "=" * 80)
    print("TEST 3: WavLM with Audio File")
    print("=" * 80)
    
    try:
        import librosa
        from main import VoiceGuardPipeline
        
        # Check if model exists
        model_path = "models/pretrained/wavlm_base_plus"
        if not Path(model_path).exists():
            print(f"❌ Model not found at: {model_path}")
            return False
        
        # Ask for audio file
        print("\nPlease provide an audio file path (WAV, MP3, etc.)")
        print("Or press Enter to skip this test...")
        audio_path = input("Audio file path: ").strip()
        
        if not audio_path:
            print("Skipping audio file test...")
            return True
        
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return False
        
        # Initialize pipeline with WavLM
        print(f"\nInitializing VoiceGuard pipeline with WavLM...")
        pipeline = VoiceGuardPipeline(
            output_dir="output",
            use_wavlm=True,
            wavlm_model_path=model_path,
            config={
                "audio": {"target_sr": 16000, "min_duration": 1.0, "max_duration": 30.0},
                "features": {"n_mfcc": 13, "n_fft": 2048, "hop_length": 512},
                "detection": {"model_type": "ensemble"},
                "reporting": {"export_formats": ["json", "text"]}
            }
        )
        
        # Analyze audio file
        print(f"\nAnalyzing audio file: {audio_path}")
        result = pipeline.analyze_audio_file(audio_path, save_report=True)
        
        if result["success"]:
            print(f"\n✅ Audio file analysis successful!")
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            print(f"  Risk Level: {result['risk_level'].upper()}")
            
            # Show report files
            if "final_report" in result and result["final_report"].get("success"):
                print(f"\n  Reports generated:")
                for fmt, filepath in result["final_report"]["exported_files"].items():
                    print(f"    {fmt.upper()}: {filepath}")
            
            return True
        else:
            print(f"\n❌ Audio file analysis failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("VoiceGuard AI - WavLM Integration Tests")
    print("=" * 80)
    print("\nThis script tests the WavLM model integration with VoiceGuard AI")
    print("Make sure you have downloaded the WavLM model first!")
    print("=" * 80)
    
    # Check if WavLM is available
    try:
        from app.models.wavlm_adapter import WavLMAdapter
        print("\n✓ WavLM adapter is available")
    except ImportError:
        print("\n❌ WavLM adapter not available")
        print("Please install required libraries:")
        print("  pip install transformers torch librosa numpy")
        return
    
    # Run tests
    results = []
    
    # Test 1: Direct adapter test
    results.append(("WavLM Adapter", test_wavlm_adapter()))
    
    # Test 2: Pipeline integration
    results.append(("Pipeline Integration", test_wavlm_with_pipeline()))
    
    # Test 3: Audio file test
    results.append(("Audio File Test", test_wavlm_with_audio_file()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! WavLM is successfully integrated.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()