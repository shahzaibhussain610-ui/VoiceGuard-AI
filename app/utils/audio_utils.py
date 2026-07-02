"""
Audio Utilities - Helper functions for audio processing in VoiceGuard AI
Provides common audio processing operations used across the system
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, Optional, Tuple, List
import os


class AudioUtils:
    """
    Utility class for common audio processing operations.
    
    This class provides:
    - Audio format conversion
    - Audio quality checks
    - Common preprocessing operations
    - Audio visualization helpers
    - Batch processing utilities
    """
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize AudioUtils.
        
        Args:
            target_sr: Target sample rate for audio processing
        """
        self.target_sr = target_sr
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate audio file format and basic properties.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": False,
            "file_path": file_path,
            "error": None,
            "info": {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result["error"] = f"File not found: {file_path}"
                return result
            
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                result["error"] = f"Unsupported format: {file_ext}"
                return result
            
            # Get file info
            file_size = os.path.getsize(file_path)
            duration = librosa.get_duration(path=file_path)
            sr = librosa.get_samplerate(path=file_path)
            
            result["valid"] = True
            result["info"] = {
                "format": file_ext,
                "file_size_bytes": file_size,
                "duration_seconds": duration,
                "sample_rate": sr,
                "num_samples": int(duration * sr)
            }
            
        except Exception as e:
            result["error"] = f"Validation error: {str(e)}"
        
        return result
    
    def convert_to_wav(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Convert audio file to WAV format.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save WAV file
            
        Returns:
            Dictionary with conversion results
        """
        result = {
            "success": False,
            "output_path": output_path,
            "error": None
        }
        
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)
            
            # Save as WAV
            sf.write(output_path, audio, self.target_sr)
            
            result["success"] = True
            result["output_path"] = output_path
            
        except Exception as e:
            result["error"] = f"Conversion error: {str(e)}"
        
        return result
    
    def normalize_audio(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """
        Normalize audio to target peak amplitude.
        
        Args:
            audio: Audio signal
            target_peak: Target peak amplitude (0-1)
            
        Returns:
            Normalized audio signal
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (target_peak / peak)
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio: Audio signal
            top_db: Threshold in dB for silence detection
            
        Returns:
            Trimmed audio signal
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio signal
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio signal
        """
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def calculate_audio_quality_metrics(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate audio quality metrics.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with quality metrics
        """
        # Signal-to-noise ratio (SNR) estimation
        signal_power = np.mean(audio ** 2)
        noise_floor = np.percentile(np.abs(audio), 10) ** 2
        snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
        
        # Clipping detection
        clipping_threshold = 0.99
        clipping_samples = np.sum(np.abs(audio) > clipping_threshold)
        clipping_ratio = clipping_samples / len(audio)
        
        return {
            "snr_db": float(snr),
            "dynamic_range_db": float(dynamic_range),
            "clipping_ratio": float(clipping_ratio),
            "peak_amplitude": float(np.max(np.abs(audio))),
            "rms_amplitude": float(np.sqrt(np.mean(audio ** 2))),
            "duration_seconds": len(audio) / sr
        }
    
    def detect_audio_issues(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect common audio issues.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with detected issues
        """
        issues = []
        quality_metrics = self.calculate_audio_quality_metrics(audio, sr)
        
        # Check for clipping
        if quality_metrics["clipping_ratio"] > 0.001:
            issues.append({
                "type": "clipping",
                "severity": "high" if quality_metrics["clipping_ratio"] > 0.01 else "medium",
                "description": f"Clipping detected in {quality_metrics['clipping_ratio']*100:.2f}% of samples"
            })
        
        # Check for low SNR
        if quality_metrics["snr_db"] < 20:
            issues.append({
                "type": "low_snr",
                "severity": "high" if quality_metrics["snr_db"] < 10 else "medium",
                "description": f"Low signal-to-noise ratio: {quality_metrics['snr_db']:.2f} dB"
            })
        
        # Check for silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            issues.append({
                "type": "low_amplitude",
                "severity": "high",
                "description": "Audio signal is very quiet or silent"
            })
        
        # Check for DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.01:
            issues.append({
                "type": "dc_offset",
                "severity": "low",
                "description": f"DC offset detected: {dc_offset:.4f}"
            })
        
        return {
            "issues_detected": len(issues),
            "issues": issues,
            "quality_metrics": quality_metrics,
            "is_clean": len(issues) == 0
        }
    
    def batch_process_files(self, 
                           file_paths: List[str], 
                           process_func: callable,
                           **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple audio files with a given function.
        
        Args:
            file_paths: List of audio file paths
            process_func: Function to apply to each file
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results for each file
        """
        results = []
        
        for file_path in file_paths:
            try:
                # Validate file first
                validation = self.validate_audio_file(file_path)
                if not validation["valid"]:
                    results.append({
                        "file_path": file_path,
                        "success": False,
                        "error": validation["error"]
                    })
                    continue
                
                # Load audio
                audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
                
                # Apply processing function
                result = process_func(audio, sr, **kwargs)
                result["file_path"] = file_path
                result["success"] = True
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "file_path": file_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def get_audio_statistics(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive statistics about audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with audio statistics
        """
        return {
            "length": len(audio),
            "mean": float(np.mean(audio)),
            "std": float(np.std(audio)),
            "min": float(np.min(audio)),
            "max": float(np.max(audio)),
            "rms": float(np.sqrt(np.mean(audio ** 2))),
            "peak": float(np.max(np.abs(audio))),
            "zero_crossings": int(np.sum(np.diff(np.sign(audio)) != 0)),
            "silence_ratio": float(np.sum(np.abs(audio) < 0.01) / len(audio))
        }


def load_and_preprocess_audio(file_path: str, 
                              target_sr: int = 16000,
                              min_duration: float = 1.0,
                              max_duration: float = 30.0) -> Dict[str, Any]:
    """
    Convenience function to load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        
    Returns:
        Dictionary with preprocessed audio and metadata
    """
    utils = AudioUtils(target_sr=target_sr)
    
    # Validate file
    validation = utils.validate_audio_file(file_path)
    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"]
        }
    
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Check duration
        duration = len(audio) / sr
        if duration < min_duration:
            return {
                "success": False,
                "error": f"Audio too short: {duration:.2f}s (min: {min_duration}s)"
            }
        
        if duration > max_duration:
            return {
                "success": False,
                "error": f"Audio too long: {duration:.2f}s (max: {max_duration}s)"
            }
        
        # Normalize and trim
        audio = utils.normalize_audio(audio)
        audio = utils.trim_silence(audio)
        
        # Get statistics
        stats = utils.get_audio_statistics(audio)
        quality = utils.detect_audio_issues(audio, sr)
        
        return {
            "success": True,
            "audio": audio,
            "sample_rate": target_sr,
            "duration": len(audio) / target_sr,
            "statistics": stats,
            "quality": quality
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load audio: {str(e)}"
        }


def create_demo_audio(duration: float = 3.0, 
                      frequency: float = 440.0,
                      sr: int = 16000) -> tuple:
    """
    Create a demo audio signal (sine wave) for testing.
    
    Args:
        duration: Duration in seconds
        frequency: Frequency in Hz
        sr: Sample rate
        
    Returns:
        Tuple of (audio_signal, sample_rate)
    """
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it more interesting
    audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32), sr
