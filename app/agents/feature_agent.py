"""
Feature Agent - Extracts audio features for deepfake detection in VoiceGuard AI
Responsible for extracting comprehensive audio features for machine learning analysis
"""

import numpy as np
import librosa
import librosa.display
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class FeatureAgent:
    """
    Agent responsible for extracting audio features in the multi-agent VoiceGuard AI system.
    
    This agent:
    - Extracts spectral features (MFCC, chroma, spectral contrast, etc.)
    - Extracts temporal features (zero-crossing rate, RMS energy, etc.)
    - Extracts prosodic features (pitch, jitter, shimmer, etc.)
    - Extracts voice quality features (HNR, formants, etc.)
    - Normalizes and structures features for downstream detection agents
    """
    
    def __init__(self, 
                 sr: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_chroma: int = 12):
        """
        Initialize the Feature Agent.
        
        Args:
            sr: Sample rate for audio processing (default: 16000 Hz)
            n_mfcc: Number of MFCC coefficients to extract (default: 13)
            n_fft: FFT window size (default: 2048)
            hop_length: Hop length for feature extraction (default: 512)
            n_chroma: Number of chroma features (default: 12)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        
    def extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCC) features.
        
        MFCCs are widely used in audio processing and speech recognition.
        They capture the spectral shape of audio signals and are effective
        for distinguishing between real and synthetic speech.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary containing MFCC features and statistics
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Calculate delta (first derivative) and delta-delta (second derivative)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Compute statistics across time frames for each coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        
        # Statistics for delta features
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta_std = np.std(mfcc_delta, axis=1)
        
        # Statistics for delta-delta features
        delta2_mean = np.mean(mfcc_delta2, axis=1)
        delta2_std = np.std(mfcc_delta2, axis=1)
        
        return {
            "mfcc_mean": mfcc_mean.tolist(),
            "mfcc_std": mfcc_std.tolist(),
            "mfcc_max": mfcc_max.tolist(),
            "mfcc_min": mfcc_min.tolist(),
            "delta_mean": delta_mean.tolist(),
            "delta_std": delta_std.tolist(),
            "delta2_mean": delta2_mean.tolist(),
            "delta2_std": delta2_std.tolist(),
            "mfcc_features": mfccs.tolist()  # Raw MFCC matrix
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract spectral domain features.
        
        Spectral features analyze the frequency content of audio signals
        and are crucial for identifying artifacts in synthetic speech.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary containing spectral features
        """
        # Spectral Centroid - center of mass of the spectrum
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral Rolloff - frequency below which a percentage of spectral energy is contained
        rolloff_85 = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85
        )[0]
        rolloff_95 = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.95
        )[0]
        
        # Spectral Bandwidth - width of the spectrum
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral Contrast - difference between peaks and valleys in spectrum
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spectral Flatness - measures how flat a spectrum is (noise-like vs. tone-like)
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Chroma Features - represent the 12 pitch classes
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Tonnetz - tonal centroid features
        tonnetz = librosa.feature.tonnetz(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        
        return {
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "spectral_centroid_std": float(np.std(spectral_centroids)),
            "rolloff_85_mean": float(np.mean(rolloff_85)),
            "rolloff_95_mean": float(np.mean(rolloff_95)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "spectral_flatness_mean": float(np.mean(spectral_flatness)),
            "spectral_flatness_std": float(np.std(spectral_flatness)),
            "spectral_contrast_mean": np.mean(spectral_contrast, axis=1).tolist(),
            "spectral_contrast_std": np.std(spectral_contrast, axis=1).tolist(),
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
            "chroma_std": np.std(chroma, axis=1).tolist(),
            "tonnetz_mean": np.mean(tonnetz, axis=1).tolist(),
            "tonnetz_std": np.std(tonnetz, axis=1).tolist()
        }
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract temporal/time-domain features.
        
        Temporal features capture the time-varying characteristics of audio
        and help identify unnatural patterns in synthetic speech.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary containing temporal features
        """
        # Zero Crossing Rate - rate at which signal changes sign
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # RMS Energy - root mean square energy per frame
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Temporal Centroid - center of temporal energy distribution
        temporal_centroid = np.sum(np.arange(len(rms)) * rms) / np.sum(rms)
        
        # Energy entropy - measure of energy distribution irregularity
        energy_frames = rms ** 2
        energy_frames_norm = energy_frames / (np.sum(energy_frames) + 1e-10)
        energy_entropy = -np.sum(energy_frames_norm * np.log2(energy_frames_norm + 1e-10))
        
        return {
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "zcr_max": float(np.max(zcr)),
            "zcr_min": float(np.min(zcr)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_max": float(np.max(rms)),
            "rms_min": float(np.min(rms)),
            "temporal_centroid": float(temporal_centroid),
            "energy_entropy": float(energy_entropy)
        }
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract prosodic features (pitch, jitter, shimmer).
        
        Prosodic features relate to the rhythm, stress, and intonation of speech.
        Synthetic speech often exhibits unnatural prosodic patterns.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary containing prosodic features
        """
        # Fundamental frequency (F0) using pYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sr,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Filter out unvoiced frames (where F0 is NaN)
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) > 0:
            # Pitch statistics
            pitch_mean = float(np.mean(f0_voiced))
            pitch_std = float(np.std(f0_voiced))
            pitch_min = float(np.min(f0_voiced))
            pitch_max = float(np.max(f0_voiced))
            pitch_range = pitch_max - pitch_min
            
            # Pitch dynamics (changes between consecutive frames)
            if len(f0_voiced) > 1:
                pitch_changes = np.diff(f0_voiced)
                pitch_change_rate = float(np.mean(np.abs(pitch_changes)))
                pitch_change_std = float(np.std(pitch_changes))
            else:
                pitch_change_rate = 0.0
                pitch_change_std = 0.0
            
            # Jitter - variation in fundamental frequency
            # Simplified jitter calculation
            if len(f0_voiced) > 1:
                jitter = float(np.mean(np.abs(np.diff(f0_voiced))) / np.mean(f0_voiced))
            else:
                jitter = 0.0
            
            # Shimmer - variation in amplitude
            rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
            if len(rms) > 1:
                shimmer = float(np.mean(np.abs(np.diff(rms))) / np.mean(rms))
            else:
                shimmer = 0.0
                
        else:
            # No voiced segments detected
            pitch_mean = pitch_std = pitch_min = pitch_max = pitch_range = 0.0
            pitch_change_rate = pitch_change_std = jitter = shimmer = 0.0
        
        # Harmonics-to-Noise Ratio (HNR)
        # Using spectral approach
        harmonic, percussive = librosa.effects.hpss(audio)
        hnr = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-10))
        
        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "pitch_range": pitch_range,
            "pitch_change_rate": pitch_change_rate,
            "pitch_change_std": pitch_change_std,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
            "voiced_ratio": float(len(f0_voiced) / len(f0))
        }
    
    def extract_formant_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract formant-related features using LPC analysis.
        
        Formants are resonant frequencies of the vocal tract and are important
        for speaker identification and detecting synthetic speech artifacts.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary containing formant features
        """
        # Use LPC to estimate formants
        # Order of LPC (number of formants to estimate)
        lpc_order = 12
        
        # Compute LPC coefficients
        try:
            # Use librosa's LPC function
            lpc_coeffs = librosa.lpc(audio, order=lpc_order)
            
            # Find roots of LPC polynomial (these correspond to formant frequencies)
            roots = np.roots(lpc_coeffs)
            
            # Keep only roots with positive imaginary part (complex conjugate pairs)
            roots = roots[np.imag(roots) >= 0]
            
            # Convert to angles (formant frequencies)
            angles = np.arctan2(np.imag(roots), np.real(roots))
            formants_hz = angles * self.sr / (2 * np.pi)
            
            # Filter to speech formant range (typically 0-5000 Hz)
            formants_hz = formants_hz[(formants_hz > 0) & (formants_hz < 5000)]
            formants_hz = np.sort(formants_hz)
            
            if len(formants_hz) >= 3:
                # Extract first 3 formants (F1, F2, F3)
                f1, f2, f3 = formants_hz[0], formants_hz[1], formants_hz[2]
                
                # Formant bandwidths
                formant_bandwidths = []
                for root in roots[:3]:
                    bandwidth = -0.5 * (np.log(np.abs(root))) * self.sr / np.pi
                    formant_bandwidths.append(float(bandwidth))
                
                return {
                    "f1": float(f1),
                    "f2": float(f2),
                    "f3": float(f3),
                    "f1_f2_ratio": float(f1 / f2) if f2 > 0 else 0.0,
                    "f2_f3_ratio": float(f2 / f3) if f3 > 0 else 0.0,
                    "formant_bandwidths": formant_bandwidths,
                    "num_formants_detected": len(formants_hz)
                }
            else:
                return {
                    "f1": 0.0,
                    "f2": 0.0,
                    "f3": 0.0,
                    "f1_f2_ratio": 0.0,
                    "f2_f3_ratio": 0.0,
                    "formant_bandwidths": [0.0],
                    "num_formants_detected": len(formants_hz)
                }
        except Exception as e:
            # Return default values if formant extraction fails
            return {
                "f1": 0.0,
                "f2": 0.0,
                "f3": 0.0,
                "f1_f2_ratio": 0.0,
                "f2_f3_ratio": 0.0,
                "formant_bandwidths": [0.0],
                "num_formants_detected": 0,
                "error": str(e)
            }
    
    def extract_statistical_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract statistical features from the audio signal.
        
        Statistical features provide a summary of the audio signal's
        distribution and characteristics.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary containing statistical features
        """
        # Basic statistics
        mean_val = float(np.mean(audio))
        std_val = float(np.std(audio))
        var_val = float(np.var(audio))
        max_val = float(np.max(audio))
        min_val = float(np.min(audio))
        range_val = max_val - min_val
        
        # Higher-order statistics
        skewness = float(stats.skew(audio))
        kurtosis = float(stats.kurtosis(audio))
        
        # Percentiles
        percentile_25 = float(np.percentile(audio, 25))
        percentile_75 = float(np.percentile(audio, 75))
        percentile_95 = float(np.percentile(audio, 95))
        percentile_5 = float(np.percentile(audio, 5))
        
        # Signal characteristics
        peak_amplitude = float(np.max(np.abs(audio)))
        crest_factor = float(peak_amplitude / (np.sqrt(np.mean(audio**2)) + 1e-10))
        
        # Zero crossings
        zero_crossings = int(np.sum(np.diff(np.sign(audio)) != 0))
        zcr_total = float(zero_crossings / len(audio))
        
        return {
            "mean": mean_val,
            "std": std_val,
            "variance": var_val,
            "max": max_val,
            "min": min_val,
            "range": range_val,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "percentile_25": percentile_25,
            "percentile_75": percentile_75,
            "percentile_95": percentile_95,
            "percentile_5": percentile_5,
            "peak_amplitude": peak_amplitude,
            "crest_factor": crest_factor,
            "zero_crossings": zero_crossings,
            "zcr_total": zcr_total
        }
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract all features from audio signal.
        
        This is the main feature extraction method that combines all
        feature extraction methods into a comprehensive feature vector.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary containing all extracted features
        """
        # Extract all feature types
        mfcc_features = self.extract_mfcc_features(audio)
        spectral_features = self.extract_spectral_features(audio)
        temporal_features = self.extract_temporal_features(audio)
        prosodic_features = self.extract_prosodic_features(audio)
        formant_features = self.extract_formant_features(audio)
        statistical_features = self.extract_statistical_features(audio)
        
        # Combine all features
        all_features = {
            "mfcc": mfcc_features,
            "spectral": spectral_features,
            "temporal": temporal_features,
            "prosodic": prosodic_features,
            "formant": formant_features,
            "statistical": statistical_features
        }
        
        # Calculate feature vector size for validation
        feature_vector_size = self._calculate_feature_vector_size(all_features)
        all_features["metadata"] = {
            "feature_vector_size": feature_vector_size,
            "audio_duration": len(audio) / self.sr,
            "num_frames": len(audio) // self.hop_length
        }
        
        return all_features
    
    def _calculate_feature_vector_size(self, features: Dict[str, Any]) -> int:
        """
        Calculate the total size of the feature vector.
        
        Args:
            features: Dictionary of all features
            
        Returns:
            Total number of features
        """
        count = 0
        
        # Count scalar features (matching the filtering in prepare_feature_vector)
        for key, value in features.items():
            if key == "metadata":
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    # Skip raw feature matrices (keep only statistics)
                    if k.endswith("_features") or k.endswith("_matrix"):
                        continue
                    if isinstance(v, (int, float)):
                        count += 1
                    elif isinstance(v, list):
                        count += len(v)
        
        return count
    
    def normalize_features(self, features: Dict[str, Any], 
                          method: str = 'zscore') -> Dict[str, Any]:
        """
        Normalize extracted features for machine learning.
        
        Args:
            features: Dictionary of extracted features
            method: Normalization method ('zscore', 'minmax', or 'robust')
            
        Returns:
            Dictionary of normalized features
        """
        normalized = {}
        
        for feature_type, feature_dict in features.items():
            if feature_type == "metadata":
                normalized[feature_type] = feature_dict
                continue
                
            normalized[feature_type] = {}
            
            for feature_name, feature_value in feature_dict.items():
                if isinstance(feature_value, list):
                    # Convert to numpy array for normalization
                    arr = np.array(feature_value)
                    
                    if method == 'zscore':
                        # Z-score normalization
                        mean = np.mean(arr)
                        std = np.std(arr)
                        normalized_arr = (arr - mean) / (std + 1e-10)
                    elif method == 'minmax':
                        # Min-max normalization
                        min_val = np.min(arr)
                        max_val = np.max(arr)
                        normalized_arr = (arr - min_val) / (max_val - min_val + 1e-10)
                    elif method == 'robust':
                        # Robust scaling using median and IQR
                        median = np.median(arr)
                        q75, q25 = np.percentile(arr, [75, 25])
                        iqr = q75 - q25
                        normalized_arr = (arr - median) / (iqr + 1e-10)
                    else:
                        raise ValueError(f"Unknown normalization method: {method}")
                    
                    normalized[feature_type][feature_name] = normalized_arr.tolist()
                else:
                    # Keep scalar values as is (they're already aggregated)
                    normalized[feature_type][feature_name] = feature_value
        
        return normalized
    
    def process(self, audio: np.ndarray, normalize: bool = True, 
                normalization_method: str = 'zscore') -> Dict[str, Any]:
        """
        Main processing method - extracts all features from audio.
        
        Args:
            audio: Audio signal as numpy array
            normalize: Whether to normalize features (default: True)
            normalization_method: Method for normalization ('zscore', 'minmax', 'robust')
            
        Returns:
            Dictionary containing all extracted features and metadata
        """
        result = {
            "success": False,
            "features": {},
            "normalized_features": {},
            "metadata": {},
            "error": None
        }
        
        try:
            # Extract all features
            features = self.extract_all_features(audio)
            
            # Normalize if requested
            if normalize:
                normalized_features = self.normalize_features(
                    features, 
                    method=normalization_method
                )
                result["normalized_features"] = normalized_features
            
            # Update result
            result["success"] = True
            result["features"] = features
            result["metadata"] = {
                "sample_rate": self.sr,
                "audio_duration": len(audio) / self.sr,
                "num_samples": len(audio),
                "feature_vector_size": features["metadata"]["feature_vector_size"],
                "normalization_applied": normalize,
                "normalization_method": normalization_method if normalize else None
            }
            
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
        
        return result
    
    def get_feature_vector(self, features: Dict[str, Any], 
                          flatten: bool = True) -> np.ndarray:
        """
        Convert feature dictionary to a flat feature vector for ML models.
        
        Args:
            features: Dictionary of extracted features
            flatten: Whether to flatten nested structures
            
        Returns:
            Numpy array containing feature vector
        """
        feature_list = []
        
        def extract_values(d):
            """Recursively extract numeric values from nested dict."""
            if isinstance(d, dict):
                for key in sorted(d.keys()):  # Sort for consistency
                    if key == "metadata":
                        continue
                    extract_values(d[key])
            elif isinstance(d, list):
                feature_list.extend(d)
            elif isinstance(d, (int, float)):
                feature_list.append(d)
        
        extract_values(features)
        
        return np.array(feature_list)
    
    def get_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of extracted features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary containing feature summary
        """
        summary = {
            "total_features": features["metadata"]["feature_vector_size"],
            "feature_categories": list(features.keys()),
            "audio_duration": features["metadata"]["audio_duration"],
            "key_indicators": {}
        }
        
        # Extract key indicators for deepfake detection
        if "prosodic" in features:
            prosodic = features["prosodic"]
            summary["key_indicators"]["pitch_stability"] = {
                "pitch_std": prosodic.get("pitch_std", 0),
                "jitter": prosodic.get("jitter", 0)
            }
            summary["key_indicators"]["amplitude_variation"] = {
                "shimmer": prosodic.get("shimmer", 0)
            }
        
        if "spectral" in features:
            spectral = features["spectral"]
            summary["key_indicators"]["spectral_characteristics"] = {
                "spectral_centroid_mean": spectral.get("spectral_centroid_mean", 0),
                "spectral_flatness_mean": spectral.get("spectral_flatness_mean", 0)
            }
        
        if "temporal" in features:
            temporal = features["temporal"]
            summary["key_indicators"]["temporal_dynamics"] = {
                "zcr_mean": temporal.get("zcr_mean", 0),
                "energy_entropy": temporal.get("energy_entropy", 0)
            }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    feature_agent = FeatureAgent(sr=16000)
    
    # Example: Extract features from audio
    print("Feature Agent - Example Usage")
    print("=" * 60)
    
    # Create a sample audio signal (sine wave for demonstration)
    duration = 3  # seconds
    t = np.linspace(0, duration, int(16000 * duration))
    sample_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    print(f"\nProcessing audio: {duration}s sine wave at 440 Hz")
    print("-" * 60)
    
    # Extract features
    result = feature_agent.process(sample_audio, normalize=True)
    
    if result["success"]:
        print(f"\n✓ Feature extraction successful!")
        print(f"  Feature vector size: {result['metadata']['feature_vector_size']}")
        print(f"  Audio duration: {result['metadata']['audio_duration']:.2f}s")
        
        # Display some key features
        print(f"\nKey Features Extracted:")
        print(f"  Spectral Centroid: {result['features']['spectral']['spectral_centroid_mean']:.2f} Hz")
        print(f"  Spectral Flatness: {result['features']['spectral']['spectral_flatness_mean']:.4f}")
        print(f"  RMS Energy: {result['features']['temporal']['rms_mean']:.4f}")
        print(f"  Zero Crossing Rate: {result['features']['temporal']['zcr_mean']:.4f}")
        
        # Get feature summary
        summary = feature_agent.get_feature_summary(result['features'])
        print(f"\nFeature Summary:")
        print(f"  Total features: {summary['total_features']}")
        print(f"  Categories: {', '.join(summary['feature_categories'])}")
        
        # Get flat feature vector
        feature_vector = feature_agent.get_feature_vector(result['features'])
        print(f"\nFeature Vector Shape: {feature_vector.shape}")
        print(f"First 10 values: {feature_vector[:10]}")
        
    else:
        print(f"\n✗ Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("Feature extraction complete!")