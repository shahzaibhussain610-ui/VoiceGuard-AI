"""
VoiceGuard AI - Deepfake Audio Detection System
A multi-agent system for detecting AI-generated or manipulated audio
"""

__version__ = "1.0.0"
__author__ = "VoiceGuard AI Team"

from app.agents.input_agent import InputAgent
from app.agents.feature_agent import FeatureAgent
from app.agents.detection_agent import DetectionAgent
from app.agents.explain_agent import ExplainAgent
from app.agents.report_agent import ReportAgent

__all__ = [
    'InputAgent',
    'FeatureAgent',
    'DetectionAgent',
    'ExplainAgent',
    'ReportAgent'
]