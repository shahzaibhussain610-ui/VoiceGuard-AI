"""
Report Agent - Generates comprehensive analysis reports in VoiceGuard AI
Responsible for creating detailed reports of deepfake detection results
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import os

# Import other agents for comprehensive reporting
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ReportMetadata:
    """
    Metadata for the analysis report.
    
    Attributes:
        report_id: Unique identifier for the report
        timestamp: When the report was generated
        audio_file: Name/path of the analyzed audio file
        analysis_duration: Time taken for analysis
        voiceguard_version: Version of VoiceGuard AI used
    """
    report_id: str
    timestamp: str
    audio_file: str
    analysis_duration: float
    voiceguard_version: str = "1.0.0"


@dataclass
class DetectionSummary:
    """
    Summary of detection results.
    
    Attributes:
        prediction: Final prediction ('real' or 'fake')
        confidence: Confidence score (0-1)
        risk_level: Risk assessment level
        probabilities: Probability distribution
    """
    prediction: str
    confidence: float
    risk_level: str
    probabilities: Dict[str, float]


class ReportAgent:
    """
    Agent responsible for generating comprehensive analysis reports in the 
    multi-agent VoiceGuard AI system.
    
    This agent:
    - Compiles results from all agents into structured reports
    - Generates multiple report formats (JSON, text, HTML)
    - Provides executive summaries and detailed analysis
    - Includes visualizations and statistical summaries
    - Supports export to various formats
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the Report Agent.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        self.report_templates = self._load_report_templates()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_report_templates(self) -> Dict[str, str]:
        """
        Load report templates for different formats.
        
        Returns:
            Dictionary of report templates
        """
        return {
            "header": "=" * 80,
            "subheader": "-" * 80,
            "section": "\n{title}\n{separator}\n{content}\n",
            "bullet": "  • {item}\n",
            "key_value": "  {key}: {value}\n"
        }
    
    def generate_report(self, 
                       input_data: Dict[str, Any],
                       features: Dict[str, Any],
                       detection_result: Dict[str, Any],
                       explanation_result: Dict[str, Any],
                       audio_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        This is the main method that combines results from all agents
        into a complete report.
        
        Args:
            input_data: Results from Input Agent
            features: Extracted features from Feature Agent
            detection_result: Detection results from Detection Agent
            explanation_result: Explanation from Explain Agent
            audio_info: Optional audio file information
            
        Returns:
            Dictionary containing the complete report
        """
        report = {
            "success": False,
            "report": None,
            "error": None
        }
        
        try:
            # Generate report metadata
            metadata = self._generate_metadata(input_data)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                detection_result, explanation_result
            )
            
            # Generate detailed sections
            input_section = self._generate_input_section(input_data, audio_info)
            features_section = self._generate_features_section(features)
            detection_section = self._generate_detection_section(detection_result)
            explanation_section = self._generate_explanation_section(explanation_result)
            
            # Generate recommendations and conclusions
            recommendations = self._generate_recommendations_section(explanation_result)
            conclusion = self._generate_conclusion(detection_result, explanation_result)
            
            # Compile complete report
            complete_report = {
                "metadata": asdict(metadata),
                "executive_summary": executive_summary,
                "sections": {
                    "input_analysis": input_section,
                    "feature_extraction": features_section,
                    "detection_results": detection_section,
                    "explanation": explanation_section,
                    "recommendations": recommendations,
                    "conclusion": conclusion
                },
                "raw_data": {
                    "input_data": input_data,
                    "features": features,
                    "detection_result": detection_result,
                    "explanation_result": explanation_result
                }
            }
            
            report["success"] = True
            report["report"] = complete_report
            
        except Exception as e:
            report["error"] = str(e)
            report["error_type"] = type(e).__name__
        
        return report
    
    def _generate_metadata(self, input_data: Dict[str, Any]) -> ReportMetadata:
        """
        Generate report metadata.
        
        Args:
            input_data: Input processing results
            
        Returns:
            ReportMetadata object
        """
        # Generate unique report ID
        report_id = f"VG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(str(input_data)) % 10000:04d}"
        
        # Get audio file name
        if isinstance(input_data.get("audio"), str):
            audio_file = input_data["audio"]
        else:
            audio_file = input_data.get("metadata", {}).get("source", "unknown")
        
        return ReportMetadata(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            audio_file=audio_file,
            analysis_duration=0.0  # Would be calculated in real implementation
        )
    
    def _generate_executive_summary(self, 
                                   detection_result: Dict[str, Any],
                                   explanation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary of the analysis.
        
        Args:
            detection_result: Detection results
            explanation_result: Explanation results
            
        Returns:
            Executive summary dictionary
        """
        prediction = detection_result.get("prediction", "unknown")
        confidence = detection_result.get("confidence", 0.0)
        
        explanation = explanation_result.get("explanation", {})
        risk_level = explanation.get("risk_level", "unknown")
        
        return {
            "verdict": prediction.upper(),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "risk_level": risk_level.upper(),
            "key_findings": explanation.get("primary_reasons", [])[:3],
            "suspicious_features_count": explanation.get("suspicious_features", {}).get("count", 0),
            "summary": self._generate_summary_text(prediction, confidence, risk_level)
        }
    
    def _generate_summary_text(self, prediction: str, confidence: float, risk_level: str) -> str:
        """
        Generate summary text for executive summary.
        
        Args:
            prediction: Model prediction
            confidence: Confidence score
            risk_level: Risk level
            
        Returns:
            Summary text string
        """
        if prediction == "fake":
            if confidence > 0.8:
                return (
                    "The analyzed audio exhibits strong indicators of being AI-generated "
                    "or manipulated. Multiple suspicious features were detected with high "
                    "confidence. Immediate verification of audio source is recommended."
                )
            elif confidence > 0.6:
                return (
                    "The analyzed audio shows several characteristics that may indicate "
                    "synthetic origin. While confidence is moderate, caution is advised "
                    "and further verification is recommended."
                )
            else:
                return (
                    "The analysis suggests potential manipulation, though with lower "
                    "confidence. Additional verification through alternative methods is "
                    "recommended before making any critical decisions."
                )
        else:  # real
            if confidence > 0.8:
                return (
                    "The analyzed audio appears to be authentic with high confidence. "
                    "No significant artifacts or anomalies were detected. The audio "
                    "exhibits characteristics consistent with natural human speech."
                )
            else:
                return (
                    "The audio appears to be authentic, though with moderate confidence. "
                    "Minor variations were observed but within acceptable ranges for "
                    "natural speech. Standard verification procedures apply."
                )
    
    def _generate_input_section(self, 
                               input_data: Dict[str, Any], 
                               audio_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate input analysis section of the report.
        
        Args:
            input_data: Input processing results
            audio_info: Optional audio file information
            
        Returns:
            Input section dictionary
        """
        metadata = input_data.get("metadata", {})
        validation = input_data.get("validation", {})
        
        section = {
            "title": "Input Analysis",
            "validation_status": "PASSED" if input_data.get("success") else "FAILED",
            "audio_properties": {
                "duration_seconds": metadata.get("duration_seconds", 0),
                "sample_rate": metadata.get("sample_rate", 0),
                "num_samples": metadata.get("num_samples", 0),
                "rms_energy": metadata.get("rms_energy", 0),
                "max_amplitude": metadata.get("max_amplitude", 0)
            },
            "validation_details": {
                "valid": validation.get("valid", False),
                "format": validation.get("format", "unknown"),
                "file_size_bytes": validation.get("file_size_bytes", 0),
                "message": validation.get("message", "")
            }
        }
        
        if audio_info:
            section["file_info"] = audio_info
        
        return section
    
    def _generate_features_section(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature extraction section of the report.
        
        Args:
            features: Extracted features
            
        Returns:
            Features section dictionary
        """
        metadata = features.get("metadata", {})
        
        section = {
            "title": "Feature Extraction",
            "feature_vector_size": metadata.get("feature_vector_size", 0),
            "audio_duration": metadata.get("audio_duration", 0),
            "num_frames": metadata.get("num_frames", 0),
            "feature_categories": {}
        }
        
        # Summarize each feature category
        if "mfcc" in features:
            section["feature_categories"]["MFCC"] = {
                "description": "Mel-Frequency Cepstral Coefficients - spectral shape features",
                "num_features": len(features["mfcc"].get("mfcc_mean", [])),
                "key_values": {
                    "mean": features["mfcc"].get("mfcc_mean", [])[:3],
                    "std": features["mfcc"].get("mfcc_std", [])[:3]
                }
            }
        
        if "spectral" in features:
            section["feature_categories"]["Spectral"] = {
                "description": "Frequency domain features",
                "key_features": {
                    "spectral_centroid_mean": features["spectral"].get("spectral_centroid_mean", 0),
                    "spectral_flatness_mean": features["spectral"].get("spectral_flatness_mean", 0),
                    "spectral_bandwidth_mean": features["spectral"].get("spectral_bandwidth_mean", 0)
                }
            }
        
        if "temporal" in features:
            section["feature_categories"]["Temporal"] = {
                "description": "Time domain features",
                "key_features": {
                    "zcr_mean": features["temporal"].get("zcr_mean", 0),
                    "rms_mean": features["temporal"].get("rms_mean", 0),
                    "energy_entropy": features["temporal"].get("energy_entropy", 0)
                }
            }
        
        if "prosodic" in features:
            section["feature_categories"]["Prosodic"] = {
                "description": "Speech prosody features (pitch, jitter, shimmer)",
                "key_features": {
                    "pitch_mean": features["prosodic"].get("pitch_mean", 0),
                    "pitch_std": features["prosodic"].get("pitch_std", 0),
                    "jitter": features["prosodic"].get("jitter", 0),
                    "shimmer": features["prosodic"].get("shimmer", 0),
                    "hnr": features["prosodic"].get("hnr", 0)
                }
            }
        
        if "formant" in features:
            section["feature_categories"]["Formant"] = {
                "description": "Vocal tract resonance features",
                "key_features": {
                    "f1": features["formant"].get("f1", 0),
                    "f2": features["formant"].get("f2", 0),
                    "f3": features["formant"].get("f3", 0)
                }
            }
        
        if "statistical" in features:
            section["feature_categories"]["Statistical"] = {
                "description": "Statistical properties of the audio signal",
                "key_features": {
                    "mean": features["statistical"].get("mean", 0),
                    "std": features["statistical"].get("std", 0),
                    "skewness": features["statistical"].get("skewness", 0),
                    "kurtosis": features["statistical"].get("kurtosis", 0)
                }
            }
        
        return section
    
    def _generate_detection_section(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detection results section of the report.
        
        Args:
            detection_result: Detection results
            
        Returns:
            Detection section dictionary
        """
        section = {
            "title": "Detection Results",
            "prediction": detection_result.get("prediction", "unknown"),
            "confidence": detection_result.get("confidence", 0.0),
            "probabilities": detection_result.get("probabilities", {}),
            "model_predictions": detection_result.get("model_predictions", {}),
            "success": detection_result.get("success", False)
        }
        
        # Add model performance if available
        if "model_scores" in detection_result:
            section["model_performance"] = detection_result["model_scores"]
        
        return section
    
    def _generate_explanation_section(self, explanation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation section of the report.
        
        Args:
            explanation_result: Explanation results
            
        Returns:
            Explanation section dictionary
        """
        explanation = explanation_result.get("explanation", {})
        
        section = {
            "title": "Detailed Analysis & Explanation",
            "risk_level": explanation.get("risk_level", "unknown"),
            "primary_reasons": explanation.get("primary_reasons", []),
            "suspicious_features": explanation.get("suspicious_features", {}),
            "feature_analysis": explanation.get("feature_analysis", {}),
            "detailed_explanation": explanation.get("explanation_text", ""),
            "timestamp": explanation.get("timestamp", "")
        }
        
        return section
    
    def _generate_recommendations_section(self, explanation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations section of the report.
        
        Args:
            explanation_result: Explanation results
            
        Returns:
            Recommendations section dictionary
        """
        explanation = explanation_result.get("explanation", {})
        
        return {
            "title": "Recommendations",
            "recommendations": explanation.get("recommendations", []),
            "priority": "HIGH" if explanation.get("risk_level") in ["high", "critical"] else "NORMAL"
        }
    
    def _generate_conclusion(self, 
                            detection_result: Dict[str, Any], 
                            explanation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate conclusion section of the report.
        
        Args:
            detection_result: Detection results
            explanation_result: Explanation results
            
        Returns:
            Conclusion dictionary
        """
        prediction = detection_result.get("prediction", "unknown")
        confidence = detection_result.get("confidence", 0.0)
        explanation = explanation_result.get("explanation", {})
        risk_level = explanation.get("risk_level", "unknown")
        
        # Determine final verdict
        if prediction == "fake" and confidence > 0.7:
            verdict = "LIKELY MANIPULATED"
            action_required = "YES"
        elif prediction == "fake":
            verdict = "POTENTIALLY MANIPULATED"
            action_required = "RECOMMENDED"
        elif confidence > 0.8:
            verdict = "LIKELY AUTHENTIC"
            action_required = "NO"
        else:
            verdict = "PROBABLY AUTHENTIC"
            action_required = "OPTIONAL"
        
        return {
            "title": "Conclusion",
            "final_verdict": verdict,
            "action_required": action_required,
            "risk_level": risk_level.upper(),
            "confidence": f"{confidence * 100:.1f}%",
            "next_steps": self._generate_next_steps(prediction, confidence, risk_level)
        }
    
    def _generate_next_steps(self, prediction: str, confidence: float, risk_level: str) -> List[str]:
        """
        Generate recommended next steps.
        
        Args:
            prediction: Model prediction
            confidence: Confidence score
            risk_level: Risk level
            
        Returns:
            List of next steps
        """
        next_steps = []
        
        if prediction == "fake":
            if risk_level in ["high", "critical"]:
                next_steps.append("Verify audio source through independent channels")
                next_steps.append("Consult with audio forensic experts")
                next_steps.append("Document findings for potential legal proceedings")
            else:
                next_steps.append("Consider additional verification methods")
                next_steps.append("Review suspicious features in detail")
        else:
            if confidence > 0.8:
                next_steps.append("No immediate action required")
                next_steps.append("Proceed with standard verification protocols")
            else:
                next_steps.append("Consider secondary verification if critical")
                next_steps.append("Monitor for any additional suspicious indicators")
        
        return next_steps
    
    def export_to_json(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export report to JSON format.
        
        Args:
            report: Report dictionary
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            metadata = report.get("metadata", {})
            report_id = metadata.get("report_id", "unknown")
            filename = f"{self.output_dir}/report_{report_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename
    
    def export_to_text(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export report to human-readable text format.
        
        Args:
            report: Report dictionary
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            metadata = report.get("metadata", {})
            report_id = metadata.get("report_id", "unknown")
            filename = f"{self.output_dir}/report_{report_id}.txt"
        
        # Generate text report
        text_lines = []
        templates = self.report_templates
        
        # Header
        text_lines.append(templates["header"])
        text_lines.append("VOICEGUARD AI - DEEPFAKE DETECTION REPORT")
        text_lines.append(templates["header"])
        
        # Metadata
        metadata = report.get("metadata", {})
        text_lines.append(templates["section"].format(
            title="REPORT INFORMATION",
            separator=templates["subheader"],
            content=self._format_dict(metadata, templates)
        ))
        
        # Executive Summary
        exec_summary = report.get("executive_summary", {})
        text_lines.append(templates["section"].format(
            title="EXECUTIVE SUMMARY",
            separator=templates["subheader"],
            content=self._format_dict(exec_summary, templates)
        ))
        
        # Sections
        sections = report.get("sections", {})
        
        for section_name, section_data in sections.items():
            if isinstance(section_data, dict) and "title" in section_data:
                text_lines.append(templates["section"].format(
                    title=section_data["title"].upper(),
                    separator=templates["subheader"],
                    content=self._format_dict(section_data, templates, exclude=["title"])
                ))
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("\n".join(text_lines))
        
        return filename
    
    def _format_dict(self, d: Dict[str, Any], templates: Dict[str, str], 
                    exclude: List[str] = None) -> str:
        """
        Format dictionary for text output.
        
        Args:
            d: Dictionary to format
            templates: Report templates
            exclude: Keys to exclude from output
            
        Returns:
            Formatted string
        """
        if exclude is None:
            exclude = []
        
        lines = []
        
        for key, value in d.items():
            if key in exclude:
                continue
            
            if isinstance(value, dict):
                lines.append(f"\n{key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"  {k}: {json.dumps(v, default=str)}")
                    else:
                        lines.append(templates["key_value"].format(key=k, value=v))
            elif isinstance(value, list):
                lines.append(f"\n{key.replace('_', ' ').title()}:")
                for item in value:
                    lines.append(templates["bullet"].format(item=item))
            else:
                lines.append(templates["key_value"].format(
                    key=key.replace('_', ' ').title(),
                    value=value
                ))
        
        return "\n".join(lines)
    
    def export_to_html(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export report to HTML format with basic styling.
        
        Args:
            report: Report dictionary
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            metadata = report.get("metadata", {})
            report_id = metadata.get("report_id", "unknown")
            filename = f"{self.output_dir}/report_{report_id}.html"
        
        # Generate HTML report
        html = self._generate_html_content(report)
        
        with open(filename, 'w') as f:
            f.write(html)
        
        return filename
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """
        Generate HTML content for the report.
        
        Args:
            report: Report dictionary
            
        Returns:
            HTML string
        """
        metadata = report.get("metadata", {})
        exec_summary = report.get("executive_summary", {})
        sections = report.get("sections", {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoiceGuard AI Report - {metadata.get('report_id', 'Unknown')}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #000000;
        }}
        .header {{
            background-color: #2c3e50;
            color: #ffffff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: #1a1a1a;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
            color: #ffffff;
        }}
        .verdict {{
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            margin: 20px 0;
        }}
        .verdict.fake {{
            background-color: #e74c3c;
            color: #ffffff;
        }}
        .verdict.real {{
            background-color: #27ae60;
            color: #ffffff;
        }}
        .risk-critical {{
            background-color: #c0392b;
        }}
        .risk-high {{
            background-color: #e74c3c;
        }}
        .risk-medium {{
            background-color: #f39c12;
        }}
        .risk-low {{
            background-color: #27ae60;
        }}
        .feature-list {{
            list-style: none;
            padding: 0;
        }}
        .feature-item {{
            padding: 10px;
            margin: 5px 0;
            background-color: #2a2a2a;
            border-left: 4px solid #3498db;
            color: #ffffff;
        }}
        .suspicious {{
            border-left-color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            color: #ffffff;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #444444;
            color: #ffffff;
        }}
        th {{
            background-color: #3498db;
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VoiceGuard AI - Deepfake Detection Report</h1>
        <p>Report ID: {metadata.get('report_id', 'N/A')}</p>
        <p>Generated: {metadata.get('timestamp', 'N/A')}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="verdict {exec_summary.get('verdict', 'unknown').lower()}">
            {exec_summary.get('verdict', 'UNKNOWN')}
        </div>
        <p><strong>Confidence:</strong> {exec_summary.get('confidence_percentage', 'N/A')}</p>
        <p><strong>Risk Level:</strong> {exec_summary.get('risk_level', 'N/A')}</p>
        <p><strong>Summary:</strong> {exec_summary.get('summary', 'N/A')}</p>
    </div>
"""
        
        # Add sections
        for section_name, section_data in sections.items():
            if isinstance(section_data, dict):
                html += f"""
    <div class="section">
        <h2>{section_data.get('title', section_name.replace('_', ' ').title())}</h2>
"""
                html += self._dict_to_html(section_data, exclude=["title"])
                html += """
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def _dict_to_html(self, d: Dict[str, Any], exclude: List[str] = None, 
                     indent: int = 2) -> str:
        """
        Convert dictionary to HTML.
        
        Args:
            d: Dictionary to convert
            exclude: Keys to exclude
            indent: Indentation level
            
        Returns:
            HTML string
        """
        if exclude is None:
            exclude = []
        
        html = ""
        spaces = " " * indent
        
        for key, value in d.items():
            if key in exclude:
                continue
            
            if isinstance(value, dict):
                html += f"\n{spaces}<h3>{key.replace('_', ' ').title()}</h3>\n"
                html += f"{spaces}<div class='subsection'>\n"
                html += self._dict_to_html(value, indent=indent+2)
                html += f"{spaces}</div>\n"
            elif isinstance(value, list):
                html += f"\n{spaces}<h3>{key.replace('_', ' ').title()}</h3>\n"
                html += f"{spaces}<ul class='feature-list'>\n"
                for item in value:
                    if isinstance(item, dict):
                        css_class = "suspicious" if item.get("severity") == "high" else ""
                        html += f"{spaces}  <li class='feature-item {css_class}'>\n"
                        html += f"{spaces}    <strong>{item.get('name', 'N/A')}</strong>: "
                        html += f"{item.get('value', 'N/A'):.4f} "
                        html += f"(expected: {item.get('expected_range', 'N/A')})\n"
                        html += f"{spaces}  </li>\n"
                    else:
                        html += f"{spaces}  <li class='feature-item'>{item}</li>\n"
                html += f"{spaces}</ul>\n"
            elif isinstance(value, (int, float)):
                html += f"{spaces}<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>\n"
            else:
                html += f"{spaces}<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>\n"
        
        return html
    
    def process(self, 
                input_data: Dict[str, Any],
                features: Dict[str, Any],
                detection_result: Dict[str, Any],
                explanation_result: Dict[str, Any],
                audio_info: Optional[Dict[str, Any]] = None,
                export_formats: List[str] = None) -> Dict[str, Any]:
        """
        Main processing method - generates and exports reports.
        
        Args:
            input_data: Results from Input Agent
            features: Extracted features from Feature Agent
            detection_result: Detection results from Detection Agent
            explanation_result: Explanation from Explain Agent
            audio_info: Optional audio file information
            export_formats: List of formats to export ('json', 'text', 'html')
            
        Returns:
            Dictionary containing report results and file paths
        """
        result = {
            "success": False,
            "report": None,
            "exported_files": {},
            "error": None
        }
        
        try:
            # Generate report
            report_result = self.generate_report(
                input_data, features, detection_result, explanation_result, audio_info
            )
            
            if not report_result["success"]:
                raise RuntimeError(f"Report generation failed: {report_result.get('error')}")
            
            report = report_result["report"]
            result["report"] = report
            
            # Export in requested formats
            if export_formats is None:
                export_formats = ["json", "text", "html"]
            
            for fmt in export_formats:
                if fmt == "json":
                    filepath = self.export_to_json(report)
                    result["exported_files"]["json"] = filepath
                elif fmt == "text":
                    filepath = self.export_to_text(report)
                    result["exported_files"]["text"] = filepath
                elif fmt == "html":
                    filepath = self.export_to_html(report)
                    result["exported_files"]["html"] = filepath
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def get_report_summary(self, report: Dict[str, Any]) -> str:
        """
        Get a brief text summary of the report.
        
        Args:
            report: Report dictionary
            
        Returns:
            Summary string
        """
        exec_summary = report.get("executive_summary", {})
        
        summary = f"""
VoiceGuard AI Analysis Report
{'=' * 80}
Report ID: {report.get('metadata', {}).get('report_id', 'N/A')}
Generated: {report.get('metadata', {}).get('timestamp', 'N/A')}

VERDICT: {exec_summary.get('verdict', 'UNKNOWN')}
Confidence: {exec_summary.get('confidence_percentage', 'N/A')}
Risk Level: {exec_summary.get('risk_level', 'UNKNOWN')}

Summary: {exec_summary.get('summary', 'N/A')}

Key Findings:
"""
        for finding in exec_summary.get('key_findings', []):
            summary += f"  • {finding}\n"
        
        conclusion = report.get("sections", {}).get("conclusion", {})
        summary += f"\nAction Required: {conclusion.get('action_required', 'N/A')}\n"
        
        return summary


# Example usage
if __name__ == "__main__":
    print("Report Agent - Example Usage")
    print("=" * 80)
    
    # Initialize the agent
    report_agent = ReportAgent(output_dir="reports")
    
    # Example: Generate a comprehensive report
    print("\nExample: Generating comprehensive analysis report")
    print("-" * 80)
    
    # Simulate results from all agents
    input_data = {
        "success": True,
        "audio": "sample_audio.wav",
        "metadata": {
            "duration_seconds": 3.5,
            "sample_rate": 16000,
            "num_samples": 56000,
            "rms_energy": 0.45,
            "max_amplitude": 0.95
        },
        "validation": {
            "valid": True,
            "format": ".wav",
            "file_size_bytes": 112000,
            "message": "Audio file validated successfully"
        }
    }
    
    features = {
        "mfcc": {
            "mfcc_mean": [0.1, 0.2, 0.3, 0.4, 0.5],
            "mfcc_std": [0.1, 0.15, 0.2, 0.25, 0.3]
        },
        "spectral": {
            "spectral_centroid_mean": 2500.0,
            "spectral_flatness_mean": 0.35,
            "spectral_bandwidth_mean": 1500.0
        },
        "temporal": {
            "zcr_mean": 0.15,
            "rms_mean": 0.4,
            "energy_entropy": 1.2
        },
        "prosodic": {
            "pitch_mean": 150.0,
            "pitch_std": 25.0,
            "jitter": 0.025,
            "shimmer": 0.08,
            "hnr": 8.0
        },
        "formant": {
            "f1": 500.0,
            "f2": 1500.0,
            "f3": 2500.0
        },
        "statistical": {
            "mean": 0.1,
            "std": 0.5,
            "skewness": 0.2
        },
        "metadata": {
            "feature_vector_size": 150,
            "audio_duration": 3.5,
            "num_frames": 109
        }
    }
    
    detection_result = {
        "success": True,
        "prediction": "fake",
        "confidence": 0.85,
        "probabilities": {
            "real": 0.15,
            "fake": 0.85
        },
        "model_predictions": {
            "rf": 1,
            "svm": 1,
            "gb": 1
        }
    }
    
    explanation_result = {
        "success": True,
        "explanation": {
            "prediction": "fake",
            "confidence": 0.85,
            "risk_level": "high",
            "primary_reasons": [
                "Detected 2 suspicious feature(s): jitter, spectral_flatness_mean",
                "High confidence detection of synthetic speech patterns",
                "Unnatural pitch variation (jitter) detected"
            ],
            "suspicious_features": {
                "count": 2,
                "features": [
                    {
                        "name": "jitter",
                        "value": 0.025,
                        "expected_range": "0.005 - 0.02",
                        "severity": "high"
                    },
                    {
                        "name": "spectral_flatness_mean",
                        "value": 0.35,
                        "expected_range": "0.0 - 0.3",
                        "severity": "medium"
                    }
                ],
                "severity": "high"
            },
            "feature_analysis": {
                "summary": {
                    "total_features_analyzed": 7,
                    "suspicious_features_found": 2,
                    "suspicion_percentage": 28.57
                }
            },
            "recommendations": [
                "SIGNIFICANT RISK: This audio shows strong indicators of being synthetic. Exercise caution and verify authenticity.",
                "Review the 2 suspicious features identified in the analysis.",
                "For critical applications, consult with audio forensic experts."
            ],
            "explanation_text": "The audio exhibits multiple characteristics consistent with synthetic speech...",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Generate report
    print("\nGenerating comprehensive report...")
    report_result = report_agent.process(
        input_data=input_data,
        features=features,
        detection_result=detection_result,
        explanation_result=explanation_result,
        export_formats=["json", "text", "html"]
    )
    
    if report_result["success"]:
        print("\n✓ Report generated successfully!")
        
        # Display summary
        print("\n" + report_agent.get_report_summary(report_result["report"]))
        
        # Show exported files
        print("\nExported Files:")
        for fmt, filepath in report_result["exported_files"].items():
            print(f"  {fmt.upper()}: {filepath}")
    else:
        print(f"\n✗ Error: {report_result['error']}")
    
    print("\n" + "=" * 80)
    print("Report agent example complete!")
