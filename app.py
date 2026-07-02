"""
VoiceGuard AI - Streamlit Web Interface
Provides an interactive web UI for deepfake audio detection
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Import VoiceGuard AI components
from main import VoiceGuardPipeline, create_demo_audio
from config import get_default_config, get_fast_config, get_accurate_config
from app.utils.helper import Helper

# Page configuration
st.set_page_config(
    page_title="VoiceGuard AI - Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #2d3142;
    }
    .success-box {
        background-color: #1e3a2f;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        color: #d4edda;
    }
    .warning-box {
        background-color: #3a3a1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        color: #fff3cd;
    }
    .error-box {
        background-color: #3a1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        color: #f8d7da;
    }
    .analyze-button {
        background-color: #e74c3c;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
    }
    .analyze-button:hover {
        background-color: #c0392b;
    }
    h3 {
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2d3142;
    }
    .stMetric label {
        color: #b0b0b0;
    }
    .stMetric value {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'sample_rate' not in st.session_state:
        st.session_state.sample_rate = None


def load_pipeline(config_type="default"):
    """Load and initialize the VoiceGuard pipeline"""
    if st.session_state.pipeline is None:
        with st.spinner("Initializing VoiceGuard AI Pipeline..."):
            if config_type == "fast":
                config = get_fast_config()
            elif config_type == "accurate":
                config = get_accurate_config()
            else:
                config = get_default_config()
            
            st.session_state.pipeline = VoiceGuardPipeline(
                output_dir="output",
                config={
                    "audio": {
                        "target_sr": config.audio.target_sr,
                        "min_duration": config.audio.min_duration,
                        "max_duration": config.audio.max_duration
                    },
                    "features": {
                        "n_mfcc": config.features.n_mfcc,
                        "n_fft": config.features.n_fft,
                        "hop_length": config.features.hop_length
                    },
                    "detection": {
                        "model_type": config.detection.model_type
                    },
                    "reporting": {
                        "export_formats": ["json", "text"]
                    }
                }
            )
            
            # Train model with synthetic data if not already trained
            if not st.session_state.pipeline.detection_agent.is_trained:
                with st.spinner("Training detection model with synthetic data (first time only)..."):
                    import numpy as np
                    from app.models.model_manager import ModelManager
                    
                    # Generate synthetic training data with correct feature vector size
                    np.random.seed(42)
                    n_samples = 200
                    
                    # Create a sample audio to determine the actual feature vector size
                    from main import create_demo_audio
                    sample_audio, _ = create_demo_audio(duration=3.0, frequency=440.0, sr=16000)
                    sample_features = st.session_state.pipeline.feature_agent.process(sample_audio, normalize=False)
                    n_features = sample_features["metadata"]["feature_vector_size"]
                    
                    print(f"Training model with {n_features} features")
                    
                    X = np.random.randn(n_samples, n_features)
                    y = np.random.randint(0, 2, n_samples)
                    
                    # Train the model
                    training_result = st.session_state.pipeline.detection_agent.train(X, y, test_size=0.2)
                    
                    if training_result["success"]:
                        st.sidebar.success("✓ Model trained successfully!")
                    else:
                        st.sidebar.warning("Model training failed. Using mock predictions.")


def plot_waveform(audio, sr):
    """Plot audio waveform"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=audio,
        mode='lines',
        name='Waveform',
        line=dict(color='#3498db', width=1)
    ))
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#1e2130',
        plot_bgcolor='#1e2130',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='#2d3142', color='#b0b0b0'),
        yaxis=dict(gridcolor='#2d3142', color='#b0b0b0')
    )
    return fig


def plot_spectrogram(audio, sr):
    """Plot audio spectrogram"""
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        colorscale='Viridis',
        colorbar=dict(title="dB", title_font_color='#ffffff', tickfont_color='#b0b0b0')
    ))
    fig.update_layout(
        title="Spectrogram",
        xaxis_title="Time Frame",
        yaxis_title="Frequency Bin",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#1e2130',
        plot_bgcolor='#1e2130',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='#2d3142', color='#b0b0b0'),
        yaxis=dict(gridcolor='#2d3142', color='#b0b0b0')
    )
    return fig


def plot_feature_importance(features):
    """Plot feature importance chart"""
    if 'mfcc' in features and 'mfcc_mean' in features['mfcc']:
        mfcc_means = features['mfcc']['mfcc_mean'][:13]
        fig = go.Figure(data=go.Bar(
            x=[f'MFCC {i+1}' for i in range(len(mfcc_means))],
            y=mfcc_means,
            marker_color='#1f77b4'
        ))
        fig.update_layout(
            title="MFCC Features",
            xaxis_title="Coefficient",
            yaxis_title="Value",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        return fig
    return None


def display_results(result):
    """Display analysis results in a nice format"""
    if not result or not result.get("success"):
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        return
    
    # Get results
    prediction = result.get("prediction", "unknown")
    confidence = result.get("confidence", 0.0)
    risk_level = result.get("risk_level", "unknown")
    
    st.divider()
    
    # Display main metrics in a row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Prediction**")
        prediction_color = "#28a745" if prediction == "real" else "#dc3545" if prediction == "fake" else "#ffc107"
        st.markdown(f"<h1 style='color: {prediction_color}; text-align: center;'>{prediction.upper()}</h1>", unsafe_allow_html=True)
        delta_text = "REAL" if prediction == "real" else "FAKE" if prediction == "fake" else "UNKNOWN"
        st.markdown(f"<p style='text-align: center; color: #b0b0b0;'>↑ {delta_text}</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Confidence**")
        st.markdown(f"<h1 style='text-align: center;'>{confidence*100:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #b0b0b0;'>Model certainty</p>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Risk Level**")
        risk_emoji = {
            "low": "⚪",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }.get(risk_level, "⚪")
        st.markdown(f"<h1 style='text-align: center;'>{risk_emoji} {risk_level.upper()}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #b0b0b0;'>Threat assessment</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Display explanation if available
    if "stages" in result and "explanation" in result["stages"]:
        explanation = result["stages"]["explanation"].get("explanation", {})
        
        st.subheader("📊 Analysis Details")
        
        # Primary reasons
        if "primary_reasons" in explanation:
            st.write("**Primary Reasons:**")
            for reason in explanation["primary_reasons"]:
                st.write(f"• {reason}")
        
        # Suspicious features
        suspicious = explanation.get("suspicious_features", {})
        if suspicious.get("count", 0) > 0:
            st.write(f"\n**Suspicious Features ({suspicious['count']}):**")
            for feature in suspicious.get("features", [])[:5]:
                severity = feature.get("severity", "unknown")
                st.write(f"  • {feature['name']}: {feature['value']:.4f} (expected: {feature['expected_range']}) - {severity}")
        
        # Recommendations
        recommendations = explanation.get("recommendations", [])
        if recommendations:
            st.write("\n**Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")
    
    # Display report files
    if "report" in result and result["report"].get("success"):
        st.subheader("📄 Generated Reports")
        for fmt, filepath in result["report"]["exported_files"].items():
            st.write(f"**{fmt.upper()}**: `{filepath}`")


def sidebar_configuration():
    """Sidebar for configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    config_type = st.sidebar.selectbox(
        "Configuration Preset",
        ["default", "fast", "accurate"],
        help="Choose processing speed vs accuracy trade-off"
    )
    
    st.sidebar.subheader("Audio Settings")
    target_sr = st.sidebar.slider("Sample Rate (Hz)", 8000, 22050, 16000, 1000)
    min_duration = st.sidebar.slider("Min Duration (s)", 0.5, 5.0, 1.0, 0.5)
    max_duration = st.sidebar.slider("Max Duration (s)", 10.0, 60.0, 30.0, 5.0)
    
    st.sidebar.subheader("Model Settings")
    model_type = st.sidebar.selectbox(
        "Detection Model",
        ["ensemble", "random_forest", "svm", "gradient_boosting"],
        help="Ensemble combines multiple models for better accuracy"
    )
    
    return {
        "config_type": config_type,
        "audio": {
            "target_sr": target_sr,
            "min_duration": min_duration,
            "max_duration": max_duration
        },
        "detection": {
            "model_type": model_type
        }
    }


def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🎙️ VoiceGuard AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deepfake Audio Detection System</p>', unsafe_allow_html=True)
    st.divider()
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Analyze Audio", "🎵 Demo Mode", "📊 Batch Analysis"])
    
    # Tab 1: File upload and analysis
    with tab1:
        st.header("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display file info
            file_size = len(uploaded_file.getbuffer())
            st.info(f"📄 **{uploaded_file.name}** ({Helper.format_file_size(file_size)})")
            
            # Load and display audio
            try:
                audio, sr = librosa.load(temp_path, sr=config["audio"]["target_sr"], mono=True)
                duration = len(audio) / sr
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duration", Helper.format_duration(duration))
                    st.metric("Sample Rate", f"{sr} Hz")
                with col2:
                    st.metric("Samples", f"{len(audio):,}")
                    st.metric("Channels", "1 (Mono)")
                
                # Visualize
                st.subheader("Audio Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_waveform(audio, sr), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_spectrogram(audio, sr), use_container_width=True)
                
                st.divider()
                
                # Analyze button
                if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                    load_pipeline(config["config_type"])
                    
                    with st.spinner("Analyzing audio..."):
                        result = st.session_state.pipeline.analyze_audio_file(
                            temp_path,
                            train_model=False
                        )
                        st.session_state.analysis_result = result
                
                # Display results
                if st.session_state.analysis_result:
                    st.divider()
                    display_results(st.session_state.analysis_result)
                
            except Exception as e:
                st.error(f"Error loading audio: {str(e)}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Tab 2: Demo mode
    with tab2:
        st.header("Demo Mode")
        st.write("Generate synthetic audio and test the detection system.")
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Duration (seconds)", 1.0, 10.0, 3.0, 0.5)
            frequency = st.slider("Frequency (Hz)", 100, 1000, 440, 50)
        
        if st.button("🎵 Generate and Analyze Demo Audio", type="primary"):
            load_pipeline(config["config_type"])
            
            with st.spinner("Generating demo audio..."):
                demo_audio, demo_sr = create_demo_audio(
                    duration=duration,
                    frequency=frequency,
                    sr=config["audio"]["target_sr"]
                )
                st.session_state.audio_data = demo_audio
                st.session_state.sample_rate = demo_sr
                
                st.success(f"Generated {duration}s audio at {demo_sr} Hz")
                
                # Visualize
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_waveform(demo_audio, demo_sr), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_spectrogram(demo_audio, demo_sr), use_container_width=True)
            
            with st.spinner("Analyzing..."):
                result = st.session_state.pipeline.analyze_audio_array(
                    demo_audio,
                    demo_sr,
                    save_report=True
                )
                st.session_state.analysis_result = result
            
            st.divider()
            display_results(result)
    
    # Tab 3: Batch analysis
    with tab3:
        st.header("Batch Analysis")
        st.write("Analyze multiple audio files at once.")
        
        uploaded_files = st.file_uploader(
            "Choose multiple audio files",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            if st.button("🔍 Analyze All Files", type="primary"):
                load_pipeline(config["config_type"])
                
                audio_files = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    audio_files.append(temp_path)
                
                with st.spinner(f"Analyzing {len(audio_files)} files..."):
                    batch_results = st.session_state.pipeline.batch_analyze(
                        audio_files,
                        output_file="output/batch_results.json"
                    )
                
                # Display summary
                st.subheader("Batch Analysis Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", batch_results["total_files"])
                with col2:
                    st.metric("Successful", batch_results["successful"])
                with col3:
                    st.metric("Failed", batch_results["failed"])
                
                # Display individual results
                st.subheader("Individual Results")
                for i, result in enumerate(batch_results["results"], 1):
                    with st.expander(f"File {i}: {result.get('audio_file', 'Unknown')}"):
                        if result["success"]:
                            st.write(f"**Prediction:** {result['prediction'].upper()}")
                            st.write(f"**Confidence:** {result['confidence']*100:.1f}%")
                            st.write(f"**Risk Level:** {result['risk_level'].upper()}")
                        else:
                            st.error(f"Error: {result.get('error')}")
                
                # Cleanup
                for temp_path in audio_files:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #b0b0b0;'>
        <p>VoiceGuard AI v1.0.0 | Deepfake Audio Detection System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()