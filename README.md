p# VoiceGuard AI - Deepfake Audio Detection System

A multi-agent AI system for detecting AI-generated or manipulated audio using advanced machine learning techniques.

## Features

- **Multi-Agent Architecture**: Specialized agents for input processing, feature extraction, detection, explanation, and reporting
- **Comprehensive Feature Extraction**: Extracts 150+ audio features including MFCC, spectral, temporal, prosodic, and formant features
- **Ensemble Detection**: Uses multiple ML models (Random Forest, SVM, Gradient Boosting) for robust classification
- **Explainable AI**: Provides detailed explanations and risk assessments for detection results
- **Multiple Export Formats**: Generate reports in JSON, text, and HTML formats
- **Batch Processing**: Analyze multiple audio files efficiently
- **Configurable**: Easy-to-use configuration system with presets for different use cases

## Project Structure

```
VoiceGuard AI/
├── main.py                      # Main pipeline script
├── demo.py                       # Demo and test script
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── app/
│   ├── __init__.py              # Package initialization
│   ├── agents/                  # AI agents
│   │   ├── __init__.py
│   │   ├── input_agent.py      # Audio input processing
│   │   ├── feature_agent.py    # Feature extraction
│   │   ├── detection_agent.py  # Deepfake detection
│   │   ├── explain_agent.py    # Explanation generation
│   │   └── report_agent.py     # Report generation
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── audio_utils.py      # Audio processing utilities
│   │   └── helper.py           # General helper functions
│   │
│   ├── models/                  # Model management
│   │   ├── __init__.py
│   │   └── model_manager.py    # Model training and inference
│   │
│   └── data/                    # Data directory
│       └── real/                # Real audio samples
│       └── fake/                # Fake audio samples
│
├── models/                      # Saved models directory
├── output/                      # Analysis output directory
└── cache/                       # Cache directory
```

## Installation

1. **Clone or download the project**
   ```bash
   cd "d:\Resume Projects\VoiceGuard AI"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python demo.py
   ```

## Quick Start

### Web Interface (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit web app
streamlit run app.py
```

This will open a web browser with an interactive interface for:
- Uploading and analyzing audio files
- Visualizing waveforms and spectrograms
- Viewing detection results with explanations
- Batch processing multiple files
- Demo mode with synthetic audio

### Command Line Usage

```python
from main import VoiceGuardPipeline

# Initialize pipeline
pipeline = VoiceGuardPipeline(output_dir="output")

# Analyze an audio file
result = pipeline.analyze_audio_file("path/to/audio.wav")

if result["success"]:
    print(f"Prediction: {result['final_report']['report']['executive_summary']['verdict']}")
    print(f"Confidence: {result['final_report']['report']['executive_summary']['confidence_percentage']}")
```

### Analyze Audio Array

```python
import numpy as np
from main import VoiceGuardPipeline

# Initialize pipeline
pipeline = VoiceGuardPipeline()

# Load or create audio (numpy array)
audio = np.load("audio.npy")
sample_rate = 16000

# Analyze
result = pipeline.analyze_audio_array(audio, sample_rate)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
```

### Run Demo (Command Line)

```bash
python main.py
```

This will run a demo with synthetic audio to showcase the system's capabilities.

### Run Tests

```bash
python demo.py
```

This will run comprehensive tests on all components.

## Configuration

### Using Default Configuration

```python
from config import get_default_config, ConfigManager

# Get default config
config = get_default_config()

# Or use ConfigManager
manager = ConfigManager()
config = manager.get_config()
```

### Custom Configuration

```python
from config import PipelineConfig, AudioConfig, FeatureConfig, DetectionConfig

config = PipelineConfig(
    audio=AudioConfig(
        target_sr=16000,
        min_duration=1.0,
        max_duration=30.0
    ),
    features=FeatureConfig(
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    ),
    detection=DetectionConfig(
        model_type="ensemble"  # 'random_forest', 'svm', 'gradient_boosting', 'ensemble'
    ),
    reporting=ReportConfig(
        export_formats=["json", "text", "html"]
    )
)
```

### Configuration Presets

```python
from config import get_fast_config, get_accurate_config

# Fast processing (lower accuracy)
fast_config = get_fast_config()

# High accuracy (slower processing)
accurate_config = get_accurate_config()
```

### Save/Load Configuration

```python
from config import ConfigManager, create_default_config_file

# Create default config file
create_default_config_file("my_config.json")

# Load from file
manager = ConfigManager("my_config.json")
config = manager.get_config()

# Save current config
manager.save_config("my_config_backup.json")
```

## Training Custom Models

### Prepare Training Data

```python
import numpy as np
from app.models.model_manager import ModelManager

# Load your features and labels
# X: feature matrix (n_samples, n_features)
# y: labels (0 for real, 1 for fake)

# Example with synthetic data
np.random.seed(42)
X = np.random.randn(200, 150)  # 200 samples, 150 features
y = np.random.randint(0, 2, 200)  # Binary labels
```

### Train Model

```python
from app.models.model_manager import ModelManager

# Initialize manager
manager = ModelManager(model_dir="models")

# Initialize models
manager.initialize_models(['random_forest', 'svm', 'gradient_boosting'])

# Train
results = manager.train(X, y, test_size=0.2, cross_validate=True)

print(f"Training successful: {results['success']}")
print(f"Model scores: {results['model_scores']}")
```

### Save and Load Model

```python
# Save model
manager.save_model("models/my_model.joblib")

# Load model
manager2 = ModelManager()
manager2.load_model("models/my_model.joblib")

# Use for prediction
prediction = manager2.predict(features)
```

## Using the Pipeline

### Complete Analysis Workflow

```python
from main import VoiceGuardPipeline

# Initialize with custom config
pipeline = VoiceGuardPipeline(
    model_path="models/my_model.joblib",  # Optional pre-trained model
    output_dir="output",
    config={
        "audio": {"target_sr": 16000, "min_duration": 1.0, "max_duration": 30.0},
        "features": {"n_mfcc": 13, "n_fft": 2048, "hop_length": 512},
        "detection": {"model_type": "ensemble"},
        "reporting": {"export_formats": ["json", "text", "html"]}
    }
)

# Analyze single file
result = pipeline.analyze_audio_file("audio.wav")

# Access results
if result["success"]:
    # Get prediction
    prediction = result["final_report"]["report"]["executive_summary"]["verdict"]
    confidence = result["final_report"]["report"]["executive_summary"]["confidence_percentage"]
    
    # Get exported files
    json_report = result["final_report"]["exported_files"]["json"]
    text_report = result["final_report"]["exported_files"]["text"]
    html_report = result["final_report"]["exported_files"]["html"]
```

### Batch Analysis

```python
# Analyze multiple files
audio_files = [
    "audio1.wav",
    "audio2.wav",
    "audio3.wav"
]

batch_results = pipeline.batch_analyze(
    audio_files,
    output_file="batch_results.json"
)

print(f"Successful: {batch_results['successful']}/{batch_results['total_files']}")
```

## Understanding the Results

### Detection Result Structure

```python
{
    "success": True,
    "prediction": "fake",  # or "real"
    "confidence": 0.85,  # 0-1
    "probabilities": {
        "real": 0.15,
        "fake": 0.85
    },
    "model_predictions": {
        "rf": 1,  # Individual model predictions
        "svm": 1,
        "gb": 1
    }
}
```

### Explanation Structure

```python
{
    "success": True,
    "explanation": {
        "prediction": "fake",
        "confidence": 0.85,
        "risk_level": "high",  # 'low', 'medium', 'high', 'critical'
        "primary_reasons": [
            "Detected 2 suspicious feature(s): jitter, spectral_flatness_mean",
            "High confidence detection of synthetic speech patterns"
        ],
        "suspicious_features": {
            "count": 2,
            "features": [...],
            "severity": "high"
        },
        "recommendations": [
            "HIGH RISK: This audio is very likely AI-generated...",
            "Consider verifying the audio source..."
        ],
        "explanation_text": "The audio exhibits multiple characteristics..."
    }
}
```

## API Reference

### VoiceGuardPipeline

Main pipeline class for audio analysis.

**Methods:**
- `analyze_audio_file(audio_path, train_model=False)` - Analyze audio file
- `analyze_audio_array(audio, sr, save_report=True)` - Analyze numpy array
- `batch_analyze(audio_files, output_file=None)` - Analyze multiple files
- `get_pipeline_info()` - Get pipeline configuration

### Agents

#### InputAgent
Handles audio input validation and preprocessing.

**Methods:**
- `process(input_data, sr=None)` - Process audio input
- `validate_audio_file(file_path)` - Validate audio file
- `load_audio(file_path)` - Load audio from file
- `get_audio_info(file_path)` - Get audio file information

#### FeatureAgent
Extracts comprehensive audio features.

**Methods:**
- `process(audio, normalize=True)` - Extract all features
- `extract_mfcc_features(audio)` - Extract MFCC features
- `extract_spectral_features(audio)` - Extract spectral features
- `extract_temporal_features(audio)` - Extract temporal features
- `extract_prosodic_features(audio)` - Extract prosodic features
- `get_feature_vector(features)` - Get flat feature vector

#### DetectionAgent
Performs deepfake detection using ML models.

**Methods:**
- `predict(features)` - Predict if audio is real or fake
- `train(X, y)` - Train the model
- `save_model(filepath)` - Save trained model
- `load_model(filepath)` - Load trained model

#### ExplainAgent
Generates explanations for detection results.

**Methods:**
- `process(detection_result, features)` - Generate explanation
- `get_simple_explanation(detection_result)` - Get one-line explanation

#### ReportAgent
Generates comprehensive analysis reports.

**Methods:**
- `process(input_data, features, detection_result, explanation_result)` - Generate report
- `export_to_json(report, filename)` - Export to JSON
- `export_to_text(report, filename)` - Export to text
- `export_to_html(report, filename)` - Export to HTML

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)

## Feature Categories

The system extracts 150+ features across multiple categories:

1. **MFCC Features** (13 coefficients + deltas)
   - Spectral shape characteristics
   - Time derivatives (delta, delta-delta)

2. **Spectral Features**
   - Spectral centroid, rolloff, bandwidth
   - Spectral contrast, flatness
   - Chroma features, tonnetz

3. **Temporal Features**
   - Zero crossing rate
   - RMS energy
   - Energy entropy
   - Temporal centroid

4. **Prosodic Features**
   - Fundamental frequency (pitch)
   - Jitter, shimmer
   - Harmonics-to-noise ratio (HNR)

5. **Formant Features**
   - F1, F2, F3 formants
   - Formant bandwidths

6. **Statistical Features**
   - Mean, std, variance
   - Skewness, kurtosis
   - Percentiles, crest factor

## Model Types

- **Random Forest**: Good for feature importance, handles non-linear relationships
- **SVM**: Effective in high-dimensional spaces
- **Gradient Boosting**: High accuracy, handles imbalanced data
- **Ensemble**: Combines multiple models for better generalization (recommended)

## Requirements

- Python 3.8+
- librosa >= 0.10.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- soundfile >= 0.12.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- joblib >= 1.2.0

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Audio Loading Errors**: Check if audio file format is supported
3. **Model Not Trained Error**: Train a model before making predictions
4. **Memory Issues**: Reduce audio duration or feature extraction parameters

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure audio files are in supported formats
4. Check that audio duration is within acceptable range (1-30 seconds)

## License

This project is created for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests.

## Version

Current Version: 1.0.0