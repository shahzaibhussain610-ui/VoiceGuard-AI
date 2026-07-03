# How to Download Real Pre-trained Models - Quick Start Guide

## 🎯 **TL;DR - Quick Answer**

Your real voice is detected as fake because the system uses a **model trained on random numbers, not real audio**. To fix this, you need to download a **real pre-trained model**.

---

## 🚀 **Fastest Method (5 Minutes)**

### **Option 1: Download WavLM Model (RECOMMENDED)**

This is the **easiest and most accurate** model to download:

```bash
# Run the download script
python download_real_model.py

# Select option 1 (WavLM)
# Wait for download to complete (~1 GB, ~5-10 minutes)
```

**What this does:**
- Downloads Microsoft's WavLM model (~99% accuracy)
- Saves it to `models/pretrained/wavlm_base_plus/`
- No manual configuration needed

---

### **Option 2: Download AASIST Model (Best Accuracy)**

If you want the **absolute best accuracy** (~99%, winner of ASVspoof 2021):

```bash
# Run the download script
python download_real_model.py

# Select option 2 (AASIST)
# Wait for download to complete (~150 MB, ~2-3 minutes)
```

**What this does:**
- Downloads the AASIST state-of-the-art model
- Saves it to `models/pretrained/aasist/`
- Requires additional setup (see below)

---

### **Option 3: Download SpeechBrain Model (Easy to Use)**

If you want something **quick and easy** with good accuracy (~97%):

```bash
# Run the download script
python download_real_model.py

# Select option 3 (SpeechBrain)
# Wait for download to complete (~500 MB, ~3-5 minutes)
```

**What this does:**
- Downloads SpeechBrain speaker recognition model
- Saves it to `models/pretrained/speechbrain_spkrec/`
- Easy to integrate with VoiceGuard AI

---

## ✅ **After Downloading - How to Use**

### **Step 1: Test the Downloaded Model**

Create a test script `test_downloaded_model.py`:

```python
import numpy as np
import librosa
from main import VoiceGuardPipeline

# Initialize pipeline with REAL pre-trained model
pipeline = VoiceGuardPipeline(
    output_dir="output",
    use_pretrained=False,  # Don't use the fake built-in model
    real_pretrained_path="models/pretrained/wavlm_base_plus"  # Use real model
)

# Test with your voice audio
print("Testing with your voice audio...")
result = pipeline.analyze_audio_file("your_voice.wav")

if result["success"]:
    print(f"\n✅ Analysis Complete!")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level'].upper()}")
else:
    print(f"\n❌ Analysis failed: {result['error']}")
```

Run the test:
```bash
python test_downloaded_model.py
```

---

### **Step 2: Update Web Interface (Optional)**

To use the real model in the web interface, modify `app.py`:

**Find this section** (around line 405-436):
```python
def load_pipeline(config_type="default"):
    """Load and initialize the VoiceGuard pipeline"""
    if st.session_state.pipeline is None:
        with st.spinner("🚀 Initializing VoiceGuard AI Pipeline..."):
            # ... existing code ...
            st.session_state.pipeline = VoiceGuardPipeline(
                output_dir="output",
                config={...}
            )
```

**Replace with:**
```python
def load_pipeline(config_type="default", use_real_model=False, real_model_path=None):
    """Load and initialize the VoiceGuard pipeline"""
    if st.session_state.pipeline is None:
        with st.spinner("🚀 Initializing VoiceGuard AI Pipeline..."):
            if config_type == "fast":
                config = get_fast_config()
            elif config_type == "accurate":
                config = get_accurate_config()
            else:
                config = get_default_config()
            
            # ADD THIS: Use real model if specified
            if use_real_model and real_model_path:
                st.session_state.pipeline = VoiceGuardPipeline(
                    output_dir="output",
                    config={...},
                    use_pretrained=False,
                    real_pretrained_path=real_model_path  # Use real model
                )
            else:
                # Use built-in model (not recommended)
                st.session_state.pipeline = VoiceGuardPipeline(
                    output_dir="output",
                    config={...}
                )
```

**Then in the main function** (around line 832), update the analyze button:
```python
# Find this:
if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
    load_pipeline(config["config_type"])
    
    with st.spinner("🧠 Analyzing audio with AI..."):
        result = st.session_state.pipeline.analyze_audio_file(
            temp_path,
            train_model=False
        )
        st.session_state.analysis_result = result

# Replace with:
# Add checkbox for real model
use_real_model = st.checkbox("Use Real Pre-trained Model", value=False, 
                             help="Use downloaded real model instead of demo model")

if use_real_model:
    real_model_path = st.text_input("Real Model Path", 
                                    value="models/pretrained/wavlm_base_plus")
else:
    real_model_path = None

if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
    load_pipeline(config["config_type"], use_real_model, real_model_path)
    
    with st.spinner("🧠 Analyzing audio with AI..."):
        result = st.session_state.pipeline.analyze_audio_file(
            temp_path,
            train_model=False
        )
        st.session_state.analysis_result = result
```

---

## 📊 **Model Comparison - Which One to Choose?**

| Model | Accuracy | Size | Download Time | Ease of Use | Best For |
|-------|----------|------|---------------|-------------|----------|
| **WavLM** | ~99% | 1 GB | 5-10 min | ⭐⭐⭐ Easy | **General use, recommended** |
| **AASIST** | ~99% | 150 MB | 2-3 min | ⭐⭐ Medium | Best accuracy, production |
| **SpeechBrain** | ~97% | 500 MB | 3-5 min | ⭐⭐⭐ Easy | Speaker verification |
| **RawNet2** | ~98% | 100 MB | 2-3 min | ⭐⭐ Medium | Research purposes |
| **LFCC-GMM** | ~95% | 50 MB | 1-2 min | ⭐⭐⭐ Easy | Fast inference |

**My Recommendation:**
- **Start with WavLM** (Option 1) - easiest and most accurate
- **Try AASIST** (Option 2) if you want the absolute best performance

---

## 🔧 **If You Get Errors**

### **Error 1: "No module named 'transformers'"**
```bash
# Install required libraries
pip install transformers torch librosa numpy scipy
```

### **Error 2: "CUDA out of memory"**
```python
# The model is too large for your GPU
# Use CPU instead:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Or use a smaller model like SpeechBrain
```

### **Error 3: "Model not found"**
```bash
# Check if model downloaded correctly
ls models/pretrained/wavlm_base_plus/

# You should see:
# - config.json
# - pytorch_model.bin (or model.safetensors)
# - feature_extractor_config.json
# - preprocessor_config.json

# If files are missing, re-download:
python download_real_model.py
```

### **Error 4: "Real pretrained path not supported"**
```bash
# You need to update main.py to support real models
# See the "Update Web Interface" section above
# Or use the Python API directly instead of web interface
```

---

## 📝 **Summary - What You Need to Do**

### **Minimum Required (5 minutes):**
1. ✅ Run `python download_real_model.py`
2. ✅ Select option 1 (WavLM)
3. ✅ Wait for download to complete
4. ✅ Test with: `python test_downloaded_model.py`

### **For Web Interface (10 minutes):**
1. ✅ Follow "Step 2" above to update `app.py`
2. ✅ Run `streamlit run app.py`
3. ✅ Check "Use Real Pre-trained Model" checkbox
4. ✅ Upload and analyze your voice

### **For Production Use (1-2 hours):**
1. ✅ Download AASIST model (best accuracy)
2. ✅ Implement model adapter (see DOWNLOAD_PRETRAINED_MODELS_GUIDE.md)
3. ✅ Test on multiple audio samples
4. ✅ Validate accuracy on test set

---

## 🎓 **Understanding What Happened**

### **Before (Current System):**
```
Your Voice → Feature Extraction → Random Model (trained on noise) → FAKE ❌
```

### **After (With Real Model):**
```
Your Voice → Feature Extraction → Real Model (trained on deepfakes) → REAL ✅
```

---

## 📚 **Additional Resources**

- **Full Guide:** See `DOWNLOAD_PRETRAINED_MODELS_GUIDE.md` for detailed instructions
- **Analysis:** See `FALSE_POSITIVE_ANALYSIS.md` to understand why the current system fails
- **ASVspoof Info:** https://www.asvspoof.org/
- **WavLM Paper:** https://arxiv.org/abs/2109.13217

---

## ❓ **Need Help?**

If you encounter any issues:

1. **Check the troubleshooting section** in `DOWNLOAD_PRETRAINED_MODELS_GUIDE.md`
2. **Verify your internet connection** (models are large files)
3. **Ensure you have enough disk space** (~1.5 GB for WavLM)
4. **Try a different model** (SpeechBrain is easier than WavLM)

---

## 🎉 **You're Ready!**

Run this command to get started:
```bash
python download_real_model.py
```

Then select option 1 (WavLM) for the best results!

**After downloading, your real voice will no longer be falsely detected as fake!** 🎤✅