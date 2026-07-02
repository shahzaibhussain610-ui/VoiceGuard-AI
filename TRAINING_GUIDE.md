# VoiceGuard AI - Model Training Guide

## Problem Identified

Your VoiceGuard AI system is incorrectly identifying audio files because **the detection model has not been trained with real data**.

### Root Cause Analysis

After analyzing your code and output reports, I found:

1. **No Trained Model**: The `models/` directory is empty - no trained model exists
2. **Untrained Detection Agent**: When the web app loads, it trains on **synthetic random data** (see `app.py` lines 158-185), not real audio features
3. **Mock Predictions**: Without proper training, the system uses random/mock predictions that are essentially guesses
4. **Generic Thresholds**: The explanation agent uses hardcoded thresholds that may not match your audio characteristics

### Evidence from Your Output

Looking at `output/report_VG-20260702172957-6320.json`:
- Model predictions: `{"rf": 0, "svm": 0, "gb": 1}` - models are disagreeing (2 say REAL, 1 says FAKE)
- Low confidence: 57.8% - indicates uncertainty
- Features flagged as suspicious (jitter, shimmer, energy_entropy) may be normal for your audio type

## Solution: Train with Your Audio Data

### Step 1: Prepare Your Dataset

Organize your audio files into two directories:

```
VoiceGuard AI/
├── app/
│   └── data/
│       ├── real/          # Put REAL audio files here
│       │   ├── real_1.wav
│       │   ├── real_2.mp3
│       │   └── ...
│       └── fake/          # Put FAKE/AI-generated audio files here
│           ├── fake_1.wav
│           ├── fake_2.ogg
│           └── ...
```

**Requirements:**
- At least 10-20 audio files per category (real/fake) for decent accuracy
- More data = better accuracy (50+ per category is ideal)
- Supported formats: WAV, MP3, FLAC, M4A, OGG
- Duration: 1-30 seconds per file

### Step 2: Train the Model

Run the training script:

```bash
python train_model.py
```

The script will:
1. Load all audio files from `app/data/real/` and `app/data/fake/`
2. Extract 150+ features from each audio file
3. Train multiple ML models (Random Forest, SVM, Gradient Boosting)
4. Evaluate model performance
5. Save the trained model to `models/voiceguard_model.joblib`

**Expected Output:**
```
================================================================================
  VoiceGuard AI - Model Training
================================================================================

[1/3] Loading Dataset
================================================================================

Loading from directories:
  Real audio: app/data/real
  Fake audio: app/data/fake

Processing 15 real audio files...
  [1/15] real_1.wav... OK
  [2/15] real_2.wav... OK
  ...

Processing 15 fake audio files...
  [1/15] fake_1.wav... OK
  ...

================================================================================
Dataset Summary:
  Total samples: 30
  Real samples: 15
  Fake samples: 15
  Feature dimensions: 208
  Failed files: 0
================================================================================

[2/3] Training Model
================================================================================

Training rf...
Training svm...
Training gb...

✓ Training completed successfully!

Model Performance:

  RF:
    Accuracy:  0.8500
    Precision: 0.8333
    Recall:    0.8750
    F1 Score:  0.8537

  SVM:
    Accuracy:  0.8000
    ...

[3/3] Saving Model
================================================================================

✓ Model trained and saved successfully!

Model saved to:
  models/voiceguard_model_20260702_173000.joblib

Default model saved to:
  models/voiceguard_model.joblib
```

### Step 3: Test the Trained Model

After training, run the web app:

```bash
streamlit run app.py
```

The app will now automatically load your trained model and make accurate predictions.

## Alternative: Quick Test with Synthetic Data

If you don't have a dataset ready, you can test with synthetic data:

```bash
python train_model.py
# When prompted: "Would you like to use synthetic demo data? (y/n): y"
```

**Note:** Synthetic data won't give accurate real-world results, but it will demonstrate the training process.

## Understanding the Results

### Model Performance Metrics

- **Accuracy**: Overall correctness (aim for >80%)
- **Precision**: Of files predicted as fake, how many were actually fake (aim for >75%)
- **Recall**: Of all fake files, how many were correctly identified (aim for >75%)
- **F1 Score**: Balance between precision and recall (aim for >75%)

### If Accuracy is Low

1. **Add more training data** - The #1 factor for accuracy
2. **Ensure data quality** - Make sure labels are correct
3. **Balance your dataset** - Similar number of real and fake samples
4. **Check audio quality** - Clear, noise-free audio works best
5. **Diversify your dataset** - Include different voices, accents, environments

## Advanced: Fine-tuning Thresholds

If your model is trained but still misidentifying certain audio types, you can adjust the detection thresholds in `app/agents/explain_agent.py`:

```python
THRESHOLDS = {
    # Prosodic features
    "jitter": {"low": 0.005, "high": 0.02, "suspicious": True},
    "shimmer": {"low": 0.03, "high": 0.1, "suspicious": True},
    
    # Adjust these based on your audio characteristics
    "energy_entropy": {"low": 0.5, "high": 2.0, "suspicious": True},
    "crest_factor": {"low": 1.5, "high": 5.0, "suspicious": True},
}
```

## How It Works

### Feature Extraction

The system extracts 150+ features from each audio file:

1. **MFCC Features** (13 coefficients + deltas) - Spectral shape
2. **Spectral Features** - Frequency domain characteristics
3. **Temporal Features** - Time domain patterns
4. **Prosodic Features** - Pitch, jitter, shimmer (voice quality)
5. **Formant Features** - Vocal tract resonances
6. **Statistical Features** - Signal statistics

### Model Training

1. **Random Forest**: Ensemble of decision trees, good for feature importance
2. **SVM**: Effective in high-dimensional spaces
3. **Gradient Boosting**: High accuracy, handles complex patterns
4. **Ensemble**: Combines all three for best performance

### Prediction Process

1. Extract features from input audio
2. Normalize features using trained scaler
3. Get predictions from all models
4. Combine using ensemble voting
5. Calculate confidence and risk level
6. Generate explanation with suspicious features

## Troubleshooting

### "No audio files found"
- Check that `app/data/real/` and `app/data/fake/` directories exist
- Verify audio files have supported extensions (.wav, .mp3, .flac, .m4a, .ogg)

### "Too few valid samples for training"
- Add more audio files (need at least 4 total, 10+ per class recommended)
- Check audio files are not corrupted
- Ensure audio duration is 1-30 seconds

### "Training failed"
- Check console output for specific errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify audio files are in valid formats

### Model accuracy is low (<70%)
- Add more training data (most important)
- Ensure correct labeling (real vs fake)
- Increase dataset diversity
- Check for audio quality issues

## Next Steps

1. Prepare your dataset - Organize real and fake audio files in `app/data/real/` and `app/data/fake/`
2. Train the model - Run `python train_model.py`
3. Test the model - Run `streamlit run app.py` and upload test files
4. Iterate - Add more data if accuracy is low
5. Deploy - Use the trained model for production

## Summary

**Yes, your model needs training.** The current system is making predictions without proper training data, which is why it's incorrectly identifying your audio files. Follow this guide to train the model with your actual audio data for accurate deepfake detection.