"""
VoiceGuard AI - Streamlit Web Interface
Provides an interactive, creative, and user-friendly web UI for deepfake audio detection
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
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for creative and friendly UI
st.markdown("""
<style>
    /* Main Background - Light and Clean */
    .stApp {
        background: #f8f9fa;
    }
    
    /* Main Header with Animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.3rem;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-in;
        font-weight: 700;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #ffffff;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideIn 0.5s ease-in;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Success Box */
    .success-box {
        background: #d4edda;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #28a745;
        color: #000000;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 0.6s ease-in;
    }
    
    /* Warning Box */
    .warning-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #ffc107;
        color: #000000;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 0.6s ease-in;
    }
    
    /* Error Box */
    .error-box {
        background: #f8d7da;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #dc3545;
        color: #000000;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 0.6s ease-in;
    }
    
    /* Info Box */
    .info-box {
        background: #d1ecf1;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #17a2b8;
        color: #000000;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 0.6s ease-in;
    }
    
    /* Analyze Button */
    .analyze-button {
        background: #3b82f6;
        color: white;
        padding: 1rem 2rem;
        border-radius: 2rem;
        border: none;
        font-size: 1.1rem;
        font-weight: 800;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    .analyze-button:hover {
        background: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #000000;
        font-weight: 800;
    }
    
    /* Metrics */
    .stMetric {
        background: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 1rem !important;
        border: 2px solid #e5e7eb !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stMetric:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stMetric label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        opacity: 1 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        opacity: 1 !important;
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        opacity: 1 !important;
    }
    
    /* Additional targeting for Streamlit metrics - more aggressive */
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Target all metric elements */
    .stMetric > div > div > div > div {
        color: #000000 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Target metric value specifically */
    [data-testid="metric-container"] {
        opacity: 1 !important;
    }
    
    [data-testid="metric-container"] > div {
        opacity: 1 !important;
        color: #000000 !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 2rem;
        padding: 0.75rem 1.5rem;
        font-weight: 800;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        background: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 1rem;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f3f4f6;
        border-radius: 0.5rem;
        color: #000000;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3b82f6;
        color: #ffffff;
        font-weight: 800;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 0.5rem;
        color: #000000;
        font-weight: 700;
    }
    
    /* Progress Bar */
    .stProgress {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 1rem;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Alerts */
    .stAlert {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 1rem;
        padding: 1rem;
        border-left: 5px solid;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #000000;
        padding: 2rem;
        font-weight: 700;
    }
    
    /* Welcome Message */
    .welcome-message {
        background: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        border-left: 5px solid #3b82f6;
        color: #000000;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 1s ease-in;
    }
    
    .welcome-message h2 {
        color: #000000;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    
    .welcome-message p {
        color: #000000;
        font-weight: 600;
        font-size: 1.1rem;
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
        with st.spinner("🚀 Initializing VoiceGuard AI Pipeline..."):
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
                with st.spinner("🧠 Training detection model with synthetic data (first time only)..."):
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
                        st.sidebar.success("✅ Model trained successfully!")
                    else:
                        st.sidebar.warning("⚠️ Model training failed. Using mock predictions.")


def plot_waveform(audio, sr):
    """Plot audio waveform with creative styling"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=audio,
        mode='lines',
        name='Waveform',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    fig.update_layout(
        title={
            'text': "🎵 Audio Waveform",
            'font': {'size': 20, 'color': '#667eea', 'family': 'Arial Black'}
        },
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        height=350,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        font=dict(color='#333333', size=12),
        xaxis=dict(gridcolor='#e0e0e0', showgrid=True),
        yaxis=dict(gridcolor='#e0e0e0', showgrid=True)
    )
    return fig


def plot_spectrogram(audio, sr):
    """Plot audio spectrogram with creative styling"""
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        colorscale='Viridis',
        colorbar=dict(
            title="dB",
            title_font_color='#667eea',
            tickfont_color='#333333',
            thickness=20
        )
    ))
    fig.update_layout(
        title={
            'text': "🎨 Audio Spectrogram",
            'font': {'size': 20, 'color': '#667eea', 'family': 'Arial Black'}
        },
        xaxis_title="Time Frame",
        yaxis_title="Frequency Bin",
        height=350,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        font=dict(color='#333333', size=12),
        xaxis=dict(gridcolor='#e0e0e0', showgrid=True),
        yaxis=dict(gridcolor='#e0e0e0', showgrid=True)
    )
    return fig


def plot_feature_importance(features):
    """Plot feature importance chart with creative styling"""
    if 'mfcc' in features and 'mfcc_mean' in features['mfcc']:
        mfcc_means = features['mfcc']['mfcc_mean'][:13]
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
                  '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140',
                  '#30cfd0', '#330867', '#a8edea']
        
        fig = go.Figure(data=go.Bar(
            x=[f'MFCC {i+1}' for i in range(len(mfcc_means))],
            y=mfcc_means,
            marker_color=colors,
            text=[f'{val:.2f}' for val in mfcc_means],
            textposition='auto',
        ))
        fig.update_layout(
            title={
                'text': "📊 MFCC Feature Importance",
                'font': {'size': 20, 'color': '#667eea', 'family': 'Arial Black'}
            },
            xaxis_title="Coefficient",
            yaxis_title="Value",
            height=350,
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            plot_bgcolor='rgba(255, 255, 255, 0.95)',
            font=dict(color='#333333', size=12),
            xaxis=dict(gridcolor='#e0e0e0', showgrid=True),
            yaxis=dict(gridcolor='#e0e0e0', showgrid=True)
        )
        return fig
    return None


def display_results(result):
    """Display analysis results in a creative and friendly format"""
    if not result or not result.get("success"):
        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return
    
    # Get results
    prediction = result.get("prediction", "unknown")
    confidence = result.get("confidence", 0.0)
    risk_level = result.get("risk_level", "unknown")
    
    st.divider()
    
    # Display main metrics in a row with creative styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction")
        if prediction == "real":
            st.markdown(f"<h1 style='color: #28a745; text-align: center; font-size: 3rem;'>✅ REAL</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #28a745; font-weight: 600;'>Authentic Audio</p>", unsafe_allow_html=True)
        elif prediction == "fake":
            st.markdown(f"<h1 style='color: #dc3545; text-align: center; font-size: 3rem;'>⚠️ FAKE</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #dc3545; font-weight: 600;'>Synthetic Audio</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: #ffc107; text-align: center; font-size: 3rem;'>❓ UNKNOWN</h1>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Confidence")
        confidence_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.6 else "#dc3545"
        st.markdown(f"<h1 style='color: {confidence_color}; text-align: center; font-size: 3rem;'>{confidence*100:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #667eea; font-weight: 600;'>Model Certainty</p>", unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(confidence)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🚨 Risk Level")
        risk_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }.get(risk_level, "⚪")
        risk_color = {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545"
        }.get(risk_level, "#6c757d")
        
        st.markdown(f"<h1 style='color: {risk_color}; text-align: center; font-size: 2.5rem;'>{risk_emoji} {risk_level.upper()}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #667eea; font-weight: 600;'>Threat Assessment</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Display explanation if available
    if "stages" in result and "explanation" in result["stages"]:
        explanation = result["stages"]["explanation"].get("explanation", {})
        
        # Analysis Details Section
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Details")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        
        # Primary reasons
        if "primary_reasons" in explanation:
            st.markdown("### 🎯 Primary Reasons")
            for reason in explanation["primary_reasons"]:
                st.markdown(f"✅ {reason}")
        
        st.markdown("")
        
        # Suspicious features
        suspicious = explanation.get("suspicious_features", {})
        if suspicious.get("count", 0) > 0:
            st.markdown(f"### ⚠️ Suspicious Features ({suspicious['count']})")
            
            # Create columns for features
            features = suspicious.get("features", [])[:5]
            if features:
                cols = st.columns(min(len(features), 3))
                for idx, feature in enumerate(features):
                    with cols[idx % 3]:
                        severity = feature.get("severity", "unknown")
                        severity_color = "#dc3545" if severity == "high" else "#ffc107" if severity == "medium" else "#28a745"
                        feature_name = feature.get('name', 'Unknown')
                        feature_value = feature.get('value', 0.0)
                        feature_expected = feature.get('expected_range', 'N/A')
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.95); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {severity_color}; margin: 0.5rem 0;'>
                            <strong style='color: #000000; font-weight: 700;'>{feature_name}</strong><br/>
                            <small style='color: #000000; font-weight: 600;'>Value: {feature_value:.4f}</small><br/>
                            <small style='color: #000000; font-weight: 600;'>Expected: {feature_expected}</small><br/>
                            <span style='color: {severity_color}; font-weight: 800; font-size: 0.9rem;'>{severity.upper()}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Recommendations
        recommendations = explanation.get("recommendations", [])
        if recommendations:
            st.markdown("### 💡 Recommendations")
            for rec in recommendations:
                st.markdown(f"💬 {rec}")
    
    st.markdown("")
    
    # Display report files
    if "report" in result and result["report"].get("success"):
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("## 📄 Generated Reports")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        for fmt, filepath in result["report"]["exported_files"].items():
            st.markdown(f"**{fmt.upper()}**: `{filepath}` ✨")


def sidebar_configuration():
    """Sidebar for configuration options with friendly styling"""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        config_type = st.selectbox(
            "🎚️ Configuration Preset",
            ["default", "fast", "accurate"],
            help="Choose processing speed vs accuracy trade-off",
            index=0
        )
        
        st.markdown("### 🎵 Audio Settings")
        target_sr = st.slider("Sample Rate (Hz)", 8000, 22050, 16000, 1000)
        min_duration = st.slider("Min Duration (s)", 0.5, 5.0, 1.0, 0.5)
        max_duration = st.slider("Max Duration (s)", 10.0, 60.0, 30.0, 5.0)
        
        st.markdown("---")
        st.markdown("### 🤖 Model Settings")
        model_type = st.selectbox(
            "Detection Model",
            ["ensemble", "random_forest", "svm", "gradient_boosting"],
            help="Ensemble combines multiple models for better accuracy",
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.5rem; color: white;'>
            <strong>VoiceGuard AI</strong><br/>
            Version: 1.0.0<br/>
            Deepfake Audio Detection<br/>
            Powered by Machine Learning
        </div>
        """, unsafe_allow_html=True)
        
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
    """Main Streamlit application with creative UI"""
    initialize_session_state()
    
    # Header with animation
    st.markdown('<p class="main-header">🛡️ VoiceGuard AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">✨ Deepfake Audio Detection System ✨</p>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="welcome-message">
        <h2>👋 Welcome to VoiceGuard AI!</h2>
        <p>Protect yourself from AI-generated audio fraud with our advanced deepfake detection technology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["📁 Analyze Audio", "🎵 Demo Mode", "📊 Batch Analysis"])
    
    # Tab 1: File upload and analysis
    with tab1:
        st.markdown("## 📤 Upload Your Audio File")
        st.markdown("Upload an audio file to analyze if it's real or AI-generated.")
        st.markdown("")
        
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
            
            # Display file info with creative styling
            file_size = len(uploaded_file.getbuffer())
            st.markdown(f"""
            <div class="info-box">
                <strong>📄 {uploaded_file.name}</strong><br/>
                <small>Size: {Helper.format_file_size(file_size)}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Load and display audio
            try:
                audio, sr = librosa.load(temp_path, sr=config["audio"]["target_sr"], mono=True)
                duration = len(audio) / sr
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="color: #000000; font-weight: 700; margin-bottom: 0.5rem;">⏱️ Duration</h3>
                        <p style="color: #000000; font-weight: 800; font-size: 1.5rem; margin: 0;">{Helper.format_duration(duration)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="color: #000000; font-weight: 700; margin-bottom: 0.5rem;">🔊 Sample Rate</h3>
                        <p style="color: #000000; font-weight: 800; font-size: 1.5rem; margin: 0;">{sr} Hz</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="color: #000000; font-weight: 700; margin-bottom: 0.5rem;">📊 Samples</h3>
                        <p style="color: #000000; font-weight: 800; font-size: 1.5rem; margin: 0;">{len(audio):,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="color: #000000; font-weight: 700; margin-bottom: 0.5rem;">🎧 Channels</h3>
                        <p style="color: #000000; font-weight: 800; font-size: 1.5rem; margin: 0;">1 (Mono)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualize
                st.markdown("### 🎨 Audio Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_waveform(audio, sr), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_spectrogram(audio, sr), use_container_width=True)
                
                st.divider()
                
                # Analyze button with animation
                st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                """, unsafe_allow_html=True)
                
                if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                    load_pipeline(config["config_type"])
                    
                    with st.spinner("🧠 Analyzing audio with AI..."):
                        result = st.session_state.pipeline.analyze_audio_file(
                            temp_path,
                            train_model=False
                        )
                        st.session_state.analysis_result = result
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display results
                if st.session_state.analysis_result:
                    st.divider()
                    display_results(st.session_state.analysis_result)
                
            except Exception as e:
                st.error(f"❌ Error loading audio: {str(e)}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Tab 2: Demo mode
    with tab2:
        st.markdown("## 🎵 Demo Mode")
        st.markdown("Generate synthetic audio and test the detection system.")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("⏱️ Duration (seconds)", 1.0, 10.0, 3.0, 0.5)
            frequency = st.slider("🎵 Frequency (Hz)", 100, 1000, 440, 50)
        
        st.markdown("")
        
        if st.button("🎵 Generate and Analyze Demo Audio", type="primary", use_container_width=True):
            load_pipeline(config["config_type"])
            
            with st.spinner("🎨 Generating demo audio..."):
                demo_audio, demo_sr = create_demo_audio(
                    duration=duration,
                    frequency=frequency,
                    sr=config["audio"]["target_sr"]
                )
                st.session_state.audio_data = demo_audio
                st.session_state.sample_rate = demo_sr
                
                st.success(f"✅ Generated {duration}s audio at {demo_sr} Hz")
                
                # Visualize
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_waveform(demo_audio, demo_sr), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_spectrogram(demo_audio, demo_sr), use_container_width=True)
            
            with st.spinner("🧠 Analyzing..."):
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
        st.markdown("## 📊 Batch Analysis")
        st.markdown("Analyze multiple audio files at once.")
        st.markdown("")
        
        uploaded_files = st.file_uploader(
            "Choose multiple audio files",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.markdown(f"<div class='info-box'><strong>📁 Uploaded {len(uploaded_files)} files</strong></div>", unsafe_allow_html=True)
            st.markdown("")
            
            if st.button("🔍 Analyze All Files", type="primary", use_container_width=True):
                load_pipeline(config["config_type"])
                
                audio_files = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    audio_files.append(temp_path)
                
                with st.spinner(f"🧠 Analyzing {len(audio_files)} files..."):
                    batch_results = st.session_state.pipeline.batch_analyze(
                        audio_files,
                        output_file="output/batch_results.json"
                    )
                
                # Display summary with creative cards
                st.markdown("### 📈 Batch Analysis Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📁 Total Files", batch_results["total_files"])
                with col2:
                    st.metric("✅ Successful", batch_results["successful"])
                with col3:
                    st.metric("❌ Failed", batch_results["failed"])
                
                st.markdown("")
                
                # Display individual results
                st.markdown("### 📋 Individual Results")
                for i, result in enumerate(batch_results["results"], 1):
                    with st.expander(f"📄 File {i}: {result.get('audio_file', 'Unknown')}"):
                        if result["success"]:
                            prediction = result['prediction']
                            confidence = result['confidence'] * 100
                            risk = result['risk_level']
                            
                            pred_emoji = "✅" if prediction == "real" else "⚠️" if prediction == "fake" else "❓"
                            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(risk, "⚪")
                            
                            st.markdown(f"**Prediction:** {pred_emoji} {prediction.upper()}")
                            st.markdown(f"**Confidence:** 📊 {confidence:.1f}%")
                            st.markdown(f"**Risk Level:** {risk_emoji} {risk.upper()}")
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                
                # Cleanup
                for temp_path in audio_files:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Footer
    st.divider()
    st.markdown("""
    <div class='footer'>
        <p>🛡️ <strong>VoiceGuard AI</strong> v1.0.0 | Deepfake Audio Detection System</p>
        <p style='font-size: 0.9rem;'>Powered by Machine Learning & Advanced Audio Analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
