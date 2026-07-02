"""
Input Agent - Handles voice/audio input processing for VoiceGuard AI
Responsible for receiving, validating, and preprocessing audio input
"""

import os
import numpy as np
from typing import Union, Optional, Dict, Any
import librosa
import soundfile as sf


class InputAgent:
    """
    Agent responsible for handling audio input in the multi-agent VoiceGuard AI system.
    
    This agent:
    - Accepts audio input from various sources (file paths, numpy arrays, etc.)
    - Validates audio format and quality
    - Preprocesses audio for downstream agents
    - Ensures consistent audio format across the pipeline
    """
    
    def __init__(self, target_sr: int = 16000, min_duration: float = 1.0, max_duration: float = 30.0):
        """
        Initialize the Input Agent.
        
        Args:
            target_sr: Target sample rate for audio processing (default: 16000 Hz)
            min_duration: Minimum audio duration in seconds (default: 1.0s)
            max_duration: Maximum audio duration in seconds (default: 30.0s)
        """
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate if the provided file is a supported audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing validation results
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        file_size = os.path.getsize(file_path)
        
        return {
            "valid": True,
            "file_path": file_path,
            "format": file_ext,
            "file_size_bytes": file_size,
            "message": "Audio file validated successfully"
        }
    
    def load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio file and return audio data with sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            ValueError: If audio duration is outside acceptable range
            RuntimeError: If audio loading fails
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            
            # Validate duration
            duration = len(audio) / sr
            if duration < self.min_duration:
                raise ValueError(
                    f"Audio too short: {duration:.2f}s. "
                    f"Minimum duration: {self.min_duration}s"
                )
            
            if duration > self.max_duration:
                raise ValueError(
                    f"Audio too long: {duration:.2f}s. "
                    f"Maximum duration: {self.max_duration}s"
                )
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Preprocess audio data for feature extraction.
        
        Args:
            audio: Raw audio data as numpy array
            sr: Sample rate of the audio
            
        Returns:
            Dictionary containing preprocessed audio and metadata
        """
        # Normalize audio
        audio_normalized = librosa.util.normalize(audio)
        
        # Remove silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=20)
        
        # Calculate audio statistics
        duration = len(audio_trimmed) / sr
        rms_energy = np.sqrt(np.mean(audio_trimmed**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_trimmed)[0])
        
        metadata = {
            "duration_seconds": duration,
            "sample_rate": sr,
            "num_samples": len(audio_trimmed),
            "rms_energy": float(rms_energy),
            "zero_crossing_rate": float(zero_crossing_rate),
            "max_amplitude": float(np.max(np.abs(audio_trimmed))),
            "mean_amplitude": float(np.mean(np.abs(audio_trimmed)))
        }
        
        return {
            "audio": audio_trimmed,
            "metadata": metadata
        }
    
    def process(self, input_data: Union[str, np.ndarray], sr: Optional[int] = None) -> Dict[str, Any]:
        """
        Main processing method - handles the complete input processing pipeline.
        
        Args:
            input_data: Either a file path (str) or numpy array of audio data
            sr: Sample rate (required if input_data is numpy array)
            
        Returns:
            Dictionary containing processed audio and all metadata
        """
        result = {
            "success": False,
            "audio": None,
            "metadata": {},
            "validation": {},
            "error": None
        }
        
        try:
            # Case 1: Input is a file path
            if isinstance(input_data, str):
                # Validate file
                validation = self.validate_audio_file(input_data)
                result["validation"] = validation
                
                # Load audio
                audio, sr = self.load_audio(input_data)
                
            # Case 2: Input is numpy array
            elif isinstance(input_data, np.ndarray):
                if sr is None:
                    raise ValueError("Sample rate (sr) must be provided when input is numpy array")
                
                # Validate duration
                duration = len(input_data) / sr
                if duration < self.min_duration or duration > self.max_duration:
                    raise ValueError(
                        f"Audio duration {duration:.2f}s is outside acceptable range "
                        f"[{self.min_duration}s, {self.max_duration}s]"
                    )
                
                audio = input_data
                validation = {
                    "valid": True,
                    "source": "numpy_array",
                    "message": "Audio array validated successfully"
                }
                result["validation"] = validation
                
            else:
                raise TypeError(
                    f"Unsupported input type: {type(input_data)}. "
                    "Expected str (file path) or np.ndarray"
                )
            
            # Preprocess audio
            processed = self.preprocess_audio(audio, sr)
            
            # Update result
            result["success"] = True
            result["audio"] = processed["audio"]
            result["metadata"] = processed["metadata"]
            result["metadata"]["source"] = input_data if isinstance(input_data, str) else "numpy_array"
            
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed information about an audio file without loading it completely.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing audio file information
        """
        try:
            # Get duration and sample rate
            duration = librosa.get_duration(path=file_path)
            sr = librosa.get_samplerate(path=file_path)
            
            return {
                "file_path": file_path,
                "duration_seconds": duration,
                "sample_rate": sr,
                "num_samples": int(duration * sr),
                "format": os.path.splitext(file_path)[1].lower(),
                "file_size_bytes": os.path.getsize(file_path)
            }
        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    input_agent = InputAgent(target_sr=16000, min_duration=1.0, max_duration=30.0)
    
    # Example 1: Process from file
    print("Example 1: Processing audio from file")
    print("-" * 50)
    result = input_agent.process("sample_audio.wav")
    
    if result["success"]:
        print(f"✓ Audio processed successfully")
        print(f"  Duration: {result['metadata']['duration_seconds']:.2f}s")
        print(f"  Sample Rate: {result['metadata']['sample_rate']} Hz")
        print(f"  RMS Energy: {result['metadata']['rms_energy']:.4f}")
    else:
        print(f"✗ Error: {result['error']}")
    
    # Example 2: Get audio info
    print("\nExample 2: Getting audio file info")
    print("-" * 50)
    info = input_agent.get_audio_info("sample_audio.wav")
    print(f"File: {info['file_path']}")
    print(f"Duration: {info['duration_seconds']:.2f}s")
    print(f"Sample Rate: {info['sample_rate']} Hz")