"""
Test script to verify the confidence calculation fix
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.detection_agent import DetectionAgent
from app.agents.feature_agent import FeatureAgent
from app.utils import audio_utils

print("="*80)
print("Testing Confidence Calculation Fix")
print("="*80)

# Create demo audio
print("\n[1/4] Creating demo audio...")
demo_audio, sr = audio_utils.create_demo_audio(duration=3.0, frequency=440.0, sr=16000)
print(f"  [OK] Created {len(demo_audio)/sr:.2f}s audio at {sr} Hz")

# Extract features
print("\n[2/4] Extracting features...")
feature_agent = FeatureAgent()
feature_result = feature_agent.process(demo_audio, normalize=True)
print(f"  [OK] Extracted {feature_result['metadata']['feature_vector_size']} features")

# Train model
print("\n[3/4] Training detection model...")
np.random.seed(42)
n_samples = 200

# Get actual feature vector size
n_features = feature_result['metadata']['feature_vector_size']
print(f"  -> Feature vector size: {n_features}")

# Create training data
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

print(f"  -> Training with {n_samples} samples")
print(f"  -> Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")

# Initialize and train detection agent
detection_agent = DetectionAgent(model_type='ensemble')
training_result = detection_agent.train(X, y, test_size=0.2)

if training_result["success"]:
    print(f"  [OK] Model trained successfully!")
    print(f"    Models: {', '.join(training_result['metrics']['models_trained'])}")
else:
    print(f"  [FAIL] Training failed: {training_result.get('error')}")
    sys.exit(1)

# Test predictions
print("\n[4/4] Testing predictions...")

# Test with the demo audio features
prediction_result = detection_agent.predict(feature_result["features"])

if prediction_result["success"]:
    pred = prediction_result["prediction"]
    conf = prediction_result["confidence"]
    real_prob = prediction_result["probabilities"]["real"]
    fake_prob = prediction_result["probabilities"]["fake"]
    
    print(f"\n  Prediction: {pred.upper()}")
    print(f"  Confidence: {conf*100:.1f}%")
    print(f"  Probabilities:")
    print(f"    Real: {real_prob*100:.1f}%")
    print(f"    Fake: {fake_prob*100:.1f}%")
    
    # Verify confidence matches prediction
    print(f"\n  Verification:")
    if pred == "real":
        expected_conf = real_prob
        print(f"    Prediction is REAL, so confidence should equal real probability ({real_prob*100:.1f}%)")
    else:
        expected_conf = fake_prob
        print(f"    Prediction is FAKE, so confidence should equal fake probability ({fake_prob*100:.1f}%)")
    
    if abs(conf - expected_conf) < 0.001:
        print(f"    [OK] Confidence matches! ({conf*100:.1f}% == {expected_conf*100:.1f}%)")
    else:
        print(f"    [FAIL] Confidence mismatch! ({conf*100:.1f}% != {expected_conf*100:.1f}%)")
    
    # Check for the original bug
    print(f"\n  Bug Check:")
    if pred == "real" and fake_prob > real_prob:
        print(f"    [FAIL] BUG DETECTED: Prediction is REAL but fake probability ({fake_prob*100:.1f}%) > real probability ({real_prob*100:.1f}%)")
    elif pred == "fake" and real_prob > fake_prob:
        print(f"    [FAIL] BUG DETECTED: Prediction is FAKE but real probability ({real_prob*100:.1f}%) > fake probability ({fake_prob*100:.1f}%)")
    else:
        print(f"    [OK] No bug - prediction matches highest probability")
    
    # Show individual model predictions
    print(f"\n  Individual Model Predictions:")
    for model, pred_val in prediction_result["model_predictions"].items():
        pred_str = "FAKE" if pred_val == 1 else "REAL"
        print(f"    {model}: {pred_str}")
    
    print(f"\n{'='*80}")
    print("Test Complete!")
    print(f"{'='*80}\n")
    
else:
    print(f"\n  [FAIL] Prediction failed: {prediction_result.get('error')}")