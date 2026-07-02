"""
Explain Agent - Provides explanations for deepfake detection results in VoiceGuard AI
Responsible for interpreting model predictions and generating human-readable explanations
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# Import feature agent for feature analysis
from app.agents.feature_agent import FeatureAgent


@dataclass
class ExplanationResult:
    """
    Data class to store explanation results.
    
    Attributes:
        prediction: The model's prediction ('real' or 'fake')
        confidence: Confidence score of the prediction
        primary_reasons: List of main reasons for the prediction
        suspicious_features: Features that indicate potential deepfake
        feature_analysis: Detailed analysis of key features
        recommendations: Suggestions based on the detection result
        risk_level: Risk assessment (low, medium, high, critical)
        explanation_text: Human-readable explanation
    """
    prediction: str
    confidence: float
    primary_reasons: List[str]
    suspicious_features: Dict[str, Any]
    feature_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_level: str
    explanation_text: str


class ExplainAgent:
    """
    Agent responsible for explaining deepfake detection results in the 
    multi-agent VoiceGuard AI system.
    
    This agent:
    - Analyzes detection results and feature patterns
    - Identifies suspicious audio characteristics
    - Generates human-readable explanations
    - Provides risk assessment and recommendations
    - Highlights key features that contributed to the decision
    """
    
    # Thresholds for suspicious feature detection
    THRESHOLDS = {
        # Prosodic features - synthetic speech often has unnatural patterns
        "jitter": {"low": 0.005, "high": 0.02, "suspicious": True},
        "shimmer": {"low": 0.03, "high": 0.1, "suspicious": True},
        "pitch_std": {"low": 5.0, "high": 50.0, "suspicious": False},
        
        # Spectral features - artifacts in synthetic speech
        "spectral_flatness_mean": {"low": 0.0, "high": 0.3, "suspicious": True},
        "spectral_centroid_std": {"low": 100.0, "high": 2000.0, "suspicious": False},
        
        # Temporal features - unnatural timing patterns
        "zcr_mean": {"low": 0.05, "high": 0.3, "suspicious": True},
        "energy_entropy": {"low": 0.5, "high": 2.0, "suspicious": True},
        
        # Formant features - unnatural vocal tract resonances
        "hnr": {"low": 0.0, "high": 10.0, "suspicious": True},
        
        # Statistical features
        "crest_factor": {"low": 1.5, "high": 5.0, "suspicious": True}
    }
    
    def __init__(self, feature_agent: Optional[FeatureAgent] = None):
        """
        Initialize the Explain Agent.
        
        Args:
            feature_agent: Optional FeatureAgent instance for feature extraction
        """
        self.feature_agent = feature_agent or FeatureAgent()
        self.explanation_templates = self._load_explanation_templates()
        
    def _load_explanation_templates(self) -> Dict[str, List[str]]:
        """
        Load explanation templates for different scenarios.
        
        Returns:
            Dictionary of explanation templates
        """
        return {
            "fake_high_confidence": [
                "The audio exhibits multiple characteristics consistent with synthetic speech.",
                "Strong evidence of voice manipulation detected in the audio signal.",
                "The combination of prosodic and spectral features indicates artificial generation."
            ],
            "fake_medium_confidence": [
                "The audio shows some characteristics that may indicate synthetic origin.",
                "Several features suggest potential voice manipulation, though with moderate certainty.",
                "The audio pattern raises suspicion but requires further analysis."
            ],
            "real_high_confidence": [
                "The audio exhibits natural speech patterns consistent with human voice.",
                "No significant artifacts or anomalies detected in the audio signal.",
                "The feature distribution matches expected patterns for authentic speech."
            ],
            "real_medium_confidence": [
                "The audio appears to be authentic with high probability.",
                "Most features indicate natural speech, though some minor variations exist.",
                "The audio shows characteristics typical of human-generated speech."
            ]
        }
    
    def analyze_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze extracted features to identify suspicious patterns.
        
        This method examines each feature category and identifies values
        that fall outside normal ranges for human speech.
        
        Args:
            features: Dictionary of extracted features from Feature Agent
            
        Returns:
            Dictionary containing feature analysis results
        """
        analysis = {
            "suspicious_features": [],
            "normal_features": [],
            "feature_scores": {},
            "anomaly_count": 0,
            "total_checked": 0
        }
        
        # Check each feature against thresholds
        for feature_name, threshold_config in self.THRESHOLDS.items():
            # Navigate through nested feature dictionary
            feature_value = self._get_nested_feature(features, feature_name)
            
            if feature_value is not None:
                analysis["total_checked"] += 1
                
                # Check if value is in suspicious range
                is_suspicious = (
                    threshold_config.get("suspicious", False) and
                    (feature_value < threshold_config["low"] or 
                     feature_value > threshold_config["high"])
                )
                
                # Calculate how far from normal range (0 = in range, 1 = far from range)
                if feature_value < threshold_config["low"]:
                    distance = (threshold_config["low"] - feature_value) / threshold_config["low"]
                elif feature_value > threshold_config["high"]:
                    distance = (feature_value - threshold_config["high"]) / threshold_config["high"]
                else:
                    distance = 0.0
                
                analysis["feature_scores"][feature_name] = {
                    "value": float(feature_value),
                    "threshold_low": threshold_config["low"],
                    "threshold_high": threshold_config["high"],
                    "distance_from_normal": float(distance),
                    "is_suspicious": is_suspicious
                }
                
                if is_suspicious:
                    analysis["suspicious_features"].append({
                        "name": feature_name,
                        "value": float(feature_value),
                        "expected_range": f"{threshold_config['low']} - {threshold_config['high']}",
                        "severity": "high" if distance > 0.5 else "medium"
                    })
                    analysis["anomaly_count"] += 1
                else:
                    analysis["normal_features"].append(feature_name)
        
        return analysis
    
    def _get_nested_feature(self, features: Dict[str, Any], feature_name: str) -> Optional[float]:
        """
        Get a feature value from nested feature dictionary.
        
        Args:
            features: Nested feature dictionary
            feature_name: Feature name to search for
            
        Returns:
            Feature value if found, None otherwise
        """
        # Search in different feature categories
        categories = ["prosodic", "spectral", "temporal", "formant", "statistical"]
        
        for category in categories:
            if category in features and feature_name in features[category]:
                value = features[category][feature_name]
                if isinstance(value, (int, float)):
                    return float(value)
        
        return None
    
    def generate_explanation(self, 
                           prediction: str, 
                           confidence: float,
                           features: Dict[str, Any],
                           feature_analysis: Dict[str, Any]) -> ExplanationResult:
        """
        Generate a comprehensive explanation for the detection result.
        
        Args:
            prediction: Model prediction ('real' or 'fake')
            confidence: Confidence score (0-1)
            features: Extracted features dictionary
            feature_analysis: Results from feature analysis
            
        Returns:
            ExplanationResult object containing detailed explanation
        """
        # Determine risk level based on prediction and confidence
        risk_level = self._assess_risk_level(prediction, confidence, feature_analysis)
        
        # Generate primary reasons for the prediction
        primary_reasons = self._generate_primary_reasons(
            prediction, confidence, feature_analysis
        )
        
        # Identify suspicious features
        suspicious_features = {
            "count": len(feature_analysis["suspicious_features"]),
            "features": feature_analysis["suspicious_features"],
            "severity": self._calculate_severity(feature_analysis["suspicious_features"])
        }
        
        # Generate detailed feature analysis
        detailed_analysis = self._generate_detailed_analysis(features, feature_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            prediction, confidence, risk_level, feature_analysis
        )
        
        # Generate human-readable explanation text
        explanation_text = self._generate_explanation_text(
            prediction, confidence, primary_reasons, suspicious_features, risk_level
        )
        
        return ExplanationResult(
            prediction=prediction,
            confidence=confidence,
            primary_reasons=primary_reasons,
            suspicious_features=suspicious_features,
            feature_analysis=detailed_analysis,
            recommendations=recommendations,
            risk_level=risk_level,
            explanation_text=explanation_text
        )
    
    def _assess_risk_level(self, 
                          prediction: str, 
                          confidence: float,
                          feature_analysis: Dict[str, Any]) -> str:
        """
        Assess the risk level based on prediction and feature analysis.
        
        Args:
            prediction: Model prediction
            confidence: Confidence score
            feature_analysis: Feature analysis results
            
        Returns:
            Risk level string ('low', 'medium', 'high', 'critical')
        """
        if prediction == "real":
            if confidence > 0.8:
                return "low"
            elif confidence > 0.6:
                return "medium"
            else:
                return "medium"
        else:  # fake
            anomaly_ratio = (
                feature_analysis["anomaly_count"] / 
                max(feature_analysis["total_checked"], 1)
            )
            
            if confidence > 0.9 and anomaly_ratio > 0.5:
                return "critical"
            elif confidence > 0.7 and anomaly_ratio > 0.3:
                return "high"
            elif confidence > 0.5:
                return "medium"
            else:
                return "low"
    
    def _generate_primary_reasons(self, 
                                 prediction: str, 
                                 confidence: float,
                                 feature_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate primary reasons for the prediction.
        
        Args:
            prediction: Model prediction
            confidence: Confidence score
            feature_analysis: Feature analysis results
            
        Returns:
            List of primary reasons
        """
        reasons = []
        
        if prediction == "fake":
            # Add reasons based on suspicious features
            if feature_analysis["anomaly_count"] > 0:
                suspicious_names = [
                    f["name"] for f in feature_analysis["suspicious_features"][:3]
                ]
                reasons.append(
                    f"Detected {feature_analysis['anomaly_count']} suspicious "
                    f"feature(s): {', '.join(suspicious_names)}"
                )
            
            if confidence > 0.8:
                reasons.append(
                    "High confidence detection of synthetic speech patterns"
                )
            elif confidence > 0.6:
                reasons.append(
                    "Moderate confidence in synthetic speech detection"
                )
            
            # Check for specific artifact types
            if any(f["name"] == "jitter" for f in feature_analysis["suspicious_features"]):
                reasons.append(
                    "Unnatural pitch variation (jitter) detected"
                )
            
            if any(f["name"] == "shimmer" for f in feature_analysis["suspicious_features"]):
                reasons.append(
                    "Unnatural amplitude variation (shimmer) detected"
                )
            
            if any(f["name"] == "spectral_flatness_mean" 
                   for f in feature_analysis["suspicious_features"]):
                reasons.append(
                    "Unusual spectral characteristics suggesting synthesis"
                )
        else:  # real
            reasons.append(
                "Audio features consistent with natural human speech"
            )
            
            if confidence > 0.8:
                reasons.append(
                    "High confidence in authentic speech classification"
                )
            
            if feature_analysis["anomaly_count"] == 0:
                reasons.append(
                    "No suspicious features detected"
                )
            else:
                reasons.append(
                    f"Only {feature_analysis['anomaly_count']} minor anomalies detected, "
                    "within acceptable range"
                )
        
        return reasons
    
    def _calculate_severity(self, suspicious_features: List[Dict[str, Any]]) -> str:
        """
        Calculate overall severity of suspicious features.
        
        Args:
            suspicious_features: List of suspicious features
            
        Returns:
            Severity level string
        """
        if not suspicious_features:
            return "none"
        
        high_severity_count = sum(
            1 for f in suspicious_features if f.get("severity") == "high"
        )
        
        if high_severity_count >= 3:
            return "critical"
        elif high_severity_count >= 1:
            return "high"
        elif len(suspicious_features) >= 3:
            return "medium"
        else:
            return "low"
    
    def _generate_detailed_analysis(self, 
                                   features: Dict[str, Any], 
                                   feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed analysis of features.
        
        Args:
            features: Extracted features
            feature_analysis: Feature analysis results
            
        Returns:
            Dictionary containing detailed analysis
        """
        analysis = {
            "summary": {
                "total_features_analyzed": feature_analysis["total_checked"],
                "suspicious_features_found": feature_analysis["anomaly_count"],
                "suspicion_percentage": (
                    feature_analysis["anomaly_count"] / 
                    max(feature_analysis["total_checked"], 1) * 100
                )
            },
            "key_indicators": {},
            "feature_categories": {}
        }
        
        # Analyze prosodic features
        if "prosodic" in features:
            prosodic = features["prosodic"]
            analysis["key_indicators"]["prosodic"] = {
                "pitch_stability": {
                    "pitch_std": prosodic.get("pitch_std", 0),
                    "jitter": prosodic.get("jitter", 0),
                    "status": "suspicious" if prosodic.get("jitter", 0) > 0.02 else "normal"
                },
                "amplitude_variation": {
                    "shimmer": prosodic.get("shimmer", 0),
                    "status": "suspicious" if prosodic.get("shimmer", 0) > 0.1 else "normal"
                },
                "voice_quality": {
                    "hnr": prosodic.get("hnr", 0),
                    "voiced_ratio": prosodic.get("voiced_ratio", 0)
                }
            }
        
        # Analyze spectral features
        if "spectral" in features:
            spectral = features["spectral"]
            analysis["key_indicators"]["spectral"] = {
                "spectral_shape": {
                    "centroid_mean": spectral.get("spectral_centroid_mean", 0),
                    "centroid_std": spectral.get("spectral_centroid_std", 0)
                },
                "spectral_flatness": {
                    "mean": spectral.get("spectral_flatness_mean", 0),
                    "status": "suspicious" if spectral.get("spectral_flatness_mean", 0) > 0.3 else "normal"
                }
            }
        
        # Analyze temporal features
        if "temporal" in features:
            temporal = features["temporal"]
            analysis["key_indicators"]["temporal"] = {
                "energy_dynamics": {
                    "rms_mean": temporal.get("rms_mean", 0),
                    "rms_std": temporal.get("rms_std", 0)
                },
                "zero_crossing": {
                    "mean": temporal.get("zcr_mean", 0),
                    "status": "suspicious" if temporal.get("zcr_mean", 0) > 0.3 else "normal"
                }
            }
        
        return analysis
    
    def _generate_recommendations(self, 
                                 prediction: str, 
                                 confidence: float,
                                 risk_level: str,
                                 feature_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on detection results.
        
        Args:
            prediction: Model prediction
            confidence: Confidence score
            risk_level: Risk level assessment
            feature_analysis: Feature analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if prediction == "fake":
            if risk_level == "critical":
                recommendations.append(
                    "HIGH RISK: This audio is very likely AI-generated or manipulated. "
                    "Do not trust this audio as authentic."
                )
                recommendations.append(
                    "Consider verifying the audio source through alternative channels."
                )
            elif risk_level == "high":
                recommendations.append(
                    "SIGNIFICANT RISK: This audio shows strong indicators of being synthetic. "
                    "Exercise caution and verify authenticity."
                )
            else:
                recommendations.append(
                    "MODERATE RISK: Some features suggest potential manipulation. "
                    "Further verification recommended."
                )
            
            if feature_analysis["anomaly_count"] > 0:
                recommendations.append(
                    f"Review the {feature_analysis['anomaly_count']} suspicious features "
                    "identified in the analysis."
                )
            
            recommendations.append(
                "For critical applications, consult with audio forensic experts."
            )
            
        else:  # real
            if risk_level == "low":
                recommendations.append(
                    "LOW RISK: This audio appears to be authentic with high confidence."
                )
                recommendations.append(
                    "No immediate action required. Standard verification procedures apply."
                )
            else:
                recommendations.append(
                    "MODERATE CONFIDENCE: Audio appears authentic but with some uncertainty."
                )
                recommendations.append(
                    "Consider additional verification if this is for critical applications."
                )
        
        return recommendations
    
    def _generate_explanation_text(self, 
                                  prediction: str, 
                                  confidence: float,
                                  primary_reasons: List[str],
                                  suspicious_features: Dict[str, Any],
                                  risk_level: str) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            prediction: Model prediction
            confidence: Confidence score
            primary_reasons: List of primary reasons
            suspicious_features: Suspicious features information
            risk_level: Risk level
            
        Returns:
            Human-readable explanation text
        """
        # Select appropriate template
        if prediction == "fake":
            if confidence > 0.7:
                template_key = "fake_high_confidence"
            else:
                template_key = "fake_medium_confidence"
        else:
            if confidence > 0.7:
                template_key = "real_high_confidence"
            else:
                template_key = "real_medium_confidence"
        
        # Build explanation
        explanation_parts = []
        
        # Add template-based opening
        import random
        explanation_parts.append(
            random.choice(self.explanation_templates[template_key])
        )
        
        # Add confidence statement
        explanation_parts.append(
            f"\n\nModel confidence: {confidence*100:.1f}%"
        )
        
        # Add primary reasons
        if primary_reasons:
            explanation_parts.append("\n\nPrimary reasons for this assessment:")
            for i, reason in enumerate(primary_reasons, 1):
                explanation_parts.append(f"\n{i}. {reason}")
        
        # Add suspicious features details
        if suspicious_features["count"] > 0:
            explanation_parts.append(
                f"\n\n{suspicious_features['count']} suspicious feature(s) detected:"
            )
            for feature in suspicious_features["features"][:5]:  # Top 5
                explanation_parts.append(
                    f"\n- {feature['name']}: {feature['value']:.4f} "
                    f"(expected: {feature['expected_range']})"
                )
        
        # Add risk assessment
        explanation_parts.append(
            f"\n\nRisk Level: {risk_level.upper()}"
        )
        
        return "".join(explanation_parts)
    
    def process(self, 
                detection_result: Dict[str, Any], 
                features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method - generates explanation for detection result.
        
        Args:
            detection_result: Results from Detection Agent
            features: Extracted features from Feature Agent
            
        Returns:
            Dictionary containing explanation results
        """
        result = {
            "success": False,
            "explanation": None,
            "error": None
        }
        
        try:
            # Extract prediction and confidence
            prediction = detection_result.get("prediction", "unknown")
            confidence = detection_result.get("confidence", 0.0)
            
            # Analyze features
            feature_analysis = self.analyze_features(features)
            
            # Generate explanation
            explanation = self.generate_explanation(
                prediction, confidence, features, feature_analysis
            )
            
            # Convert to dictionary
            result["success"] = True
            result["explanation"] = {
                "prediction": explanation.prediction,
                "confidence": explanation.confidence,
                "risk_level": explanation.risk_level,
                "primary_reasons": explanation.primary_reasons,
                "suspicious_features": explanation.suspicious_features,
                "feature_analysis": explanation.feature_analysis,
                "recommendations": explanation.recommendations,
                "explanation_text": explanation.explanation_text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def get_simple_explanation(self, detection_result: Dict[str, Any]) -> str:
        """
        Get a simple one-line explanation.
        
        Args:
            detection_result: Results from Detection Agent
            
        Returns:
            Simple explanation string
        """
        prediction = detection_result.get("prediction", "unknown")
        confidence = detection_result.get("confidence", 0.0)
        
        if prediction == "fake":
            if confidence > 0.8:
                return f"This audio is likely AI-generated or manipulated ({confidence*100:.1f}% confidence)"
            else:
                return f"This audio may be manipulated with moderate confidence ({confidence*100:.1f}%)"
        else:
            if confidence > 0.8:
                return f"This audio appears to be authentic ({confidence*100:.1f}% confidence)"
            else:
                return f"This audio is likely authentic but with some uncertainty ({confidence*100:.1f}%)"


# Example usage
if __name__ == "__main__":
    print("Explain Agent - Example Usage")
    print("=" * 60)
    
    # Initialize the agent
    explain_agent = ExplainAgent()
    
    # Example: Explain a detection result
    print("\nExample: Explaining detection results")
    print("-" * 60)
    
    # Simulate detection result (from Detection Agent)
    detection_result = {
        "success": True,
        "prediction": "fake",
        "confidence": 0.85,
        "probabilities": {
            "real": 0.15,
            "fake": 0.85
        }
    }
    
    # Simulate features (from Feature Agent)
    features = {
        "prosodic": {
            "pitch_mean": 150.0,
            "pitch_std": 25.0,
            "jitter": 0.025,  # Suspicious: above threshold
            "shimmer": 0.08,  # Normal
            "hnr": 8.0,
            "voiced_ratio": 0.7
        },
        "spectral": {
            "spectral_centroid_mean": 2500.0,
            "spectral_centroid_std": 500.0,
            "spectral_flatness_mean": 0.35,  # Suspicious: above threshold
            "spectral_bandwidth_mean": 1500.0
        },
        "temporal": {
            "zcr_mean": 0.15,
            "rms_mean": 0.4,
            "energy_entropy": 1.2
        },
        "formant": {
            "f1": 500.0,
            "f2": 1500.0,
            "f3": 2500.0
        },
        "statistical": {
            "mean": 0.1,
            "std": 0.5,
            "skewness": 0.2,
            "crest_factor": 2.5
        }
    }
    
    # Generate explanation
    explanation_result = explain_agent.process(detection_result, features)
    
    if explanation_result["success"]:
        explanation = explanation_result["explanation"]
        
        print(f"\n✓ Explanation generated successfully!")
        print(f"\nPrediction: {explanation['prediction'].upper()}")
        print(f"Confidence: {explanation['confidence']*100:.1f}%")
        print(f"Risk Level: {explanation['risk_level'].upper()}")
        
        print(f"\nPrimary Reasons:")
        for reason in explanation["primary_reasons"]:
            print(f"  • {reason}")
        
        print(f"\nSuspicious Features ({explanation['suspicious_features']['count']}):")
        for feature in explanation["suspicious_features"]["features"]:
            print(f"  • {feature['name']}: {feature['value']:.4f} "
                  f"(expected: {feature['expected_range']})")
        
        print(f"\nRecommendations:")
        for rec in explanation["recommendations"]:
            print(f"  • {rec}")
        
        print(f"\nFull Explanation:")
        print(explanation["explanation_text"])
        
        # Get simple explanation
        print("\n" + "-" * 60)
        print("Simple Explanation:")
        print("-" * 60)
        simple = explain_agent.get_simple_explanation(detection_result)
        print(simple)
    
    print("\n" + "=" * 60)
    print("Explain agent example complete!")