#!/usr/bin/env python3
"""
PercePiano Audio Preprocessing Pipeline
Standardized audio feature extraction for ML training
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import librosa
    import librosa.display
    print("✅ librosa available for preprocessing")
except ImportError:
    print("❌ librosa required: pip install librosa")
    exit(1)

class PercePianoPreprocessor:
    """Standardized audio preprocessing for PercePiano dataset"""
    
    def __init__(self, 
                 target_sr: int = 22050,  # Standard sampling rate for ML
                 n_fft: int = 2048,       # FFT window size
                 hop_length: int = 512,   # Hop length for spectrograms
                 n_mels: int = 128,       # Number of mel bands
                 n_mfcc: int = 13):       # Number of MFCCs
        
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        print(f"PercePiano Preprocessor initialized:")
        print(f"  Target SR: {target_sr}Hz")
        print(f"  FFT size: {n_fft}, Hop: {hop_length}")
        print(f"  Mel bands: {n_mels}, MFCCs: {n_mfcc}")
    
    def load_and_normalize_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file with consistent preprocessing
        Returns: (normalized_audio, sample_rate)
        """
        
        # Load audio with target sampling rate
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # Normalize audio to [-1, 1] range
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        print(f"  Loaded: {len(y)} samples ({len(y)/sr:.2f}s) at {sr}Hz")
        
        return y, sr
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract spectral features for piano analysis"""
        
        features = {}
        
        # 1. Short-time Fourier Transform (STFT) 
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        features['stft_magnitude'] = magnitude
        features['stft_phase'] = phase
        
        # 2. Mel-spectrogram (for timbre analysis)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = mel_spec_db
        
        # 3. MFCCs (compact spectral representation)
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features['mfccs'] = mfccs
        
        # 4. Chromagram (harmonic content)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features['chromagram'] = chroma
        
        print(f"  Spectral features extracted:")
        print(f"    STFT: {magnitude.shape}")
        print(f"    Mel-spec: {mel_spec_db.shape}") 
        print(f"    MFCCs: {mfccs.shape}")
        print(f"    Chroma: {chroma.shape}")
        
        return features
    
    def extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract timing and rhythm features"""
        
        features = {}
        
        # 1. Tempo and beat tracking
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            features['tempo'] = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            features['num_beats'] = len(beats)
            features['beat_consistency'] = float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0
        except:
            features['tempo'] = 0.0
            features['num_beats'] = 0
            features['beat_consistency'] = 0.0
        
        # 2. Zero crossing rate (articulation indicator)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 3. Onset detection (note attack timing)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, 
            hop_length=self.hop_length,
            units='frames'
        )
        features['num_onsets'] = len(onset_frames)
        features['onset_density'] = len(onset_frames) / (len(y) / sr)  # onsets per second
        
        return features
    
    def extract_dynamic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract dynamic and energy features"""
        
        features = {}
        
        # 1. RMS energy (loudness)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_max'] = float(np.max(rms))
        features['rms_min'] = float(np.min(rms))
        features['dynamic_range'] = features['rms_max'] - features['rms_min']
        
        # 2. Spectral rolloff (brightness indicator)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        features['spectral_rolloff_std'] = float(np.std(rolloff))
        
        # 3. Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_centroid_mean'] = float(np.mean(centroid))
        features['spectral_centroid_std'] = float(np.std(centroid))
        
        # 4. Spectral bandwidth (timbre richness)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(bandwidth))
        
        return features
    
    def process_audio_file(self, audio_path: Path, performance_name: str) -> Dict:
        """
        Complete preprocessing pipeline for one audio file
        Returns all features needed for ML training
        """
        
        print(f"\\n=== PREPROCESSING: {performance_name} ===")
        
        # 1. Load and normalize audio
        y, sr = self.load_and_normalize_audio(audio_path)
        
        # 2. Extract all feature types
        spectral_features = self.extract_spectral_features(y, sr)
        temporal_features = self.extract_temporal_features(y, sr)
        dynamic_features = self.extract_dynamic_features(y, sr)
        
        # 3. Package results
        processed_data = {
            'performance_name': performance_name,
            'audio_path': str(audio_path),
            'duration': len(y) / sr,
            'sample_rate': sr,
            
            # Raw spectral data (for neural networks)
            'spectral_data': {
                'mel_spectrogram': spectral_features['mel_spectrogram'].tolist(),
                'mfccs': spectral_features['mfccs'].tolist(),
                'chromagram': spectral_features['chromagram'].tolist()
            },
            
            # Scalar features (for traditional ML)
            'scalar_features': {
                **temporal_features,
                **dynamic_features
            }
        }
        
        print(f"  ✅ Processing complete - {len(processed_data['scalar_features'])} scalar features")
        
        return processed_data

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline on available audio files"""
    
    print("=== PercePiano Preprocessing Pipeline Test ===\\n")
    
    # Initialize preprocessor
    preprocessor = PercePianoPreprocessor()
    
    # Test on example audio file
    example_audio = Path('examples/Beethoven_WoO80_var27_8bars_3_15.wav')
    
    if not example_audio.exists():
        print(f"❌ No audio file found at {example_audio}")
        return
    
    performance_name = example_audio.stem
    
    # Process the file
    try:
        processed_data = preprocessor.process_audio_file(example_audio, performance_name)
        
        # Show results summary
        print(f"\\n📊 FEATURE SUMMARY:")
        print(f"  Performance: {processed_data['performance_name']}")
        print(f"  Duration: {processed_data['duration']:.2f}s")
        
        print(f"\\n  Spectral data shapes:")
        for key, data in processed_data['spectral_data'].items():
            shape = np.array(data).shape
            print(f"    {key}: {shape}")
        
        print(f"\\n  Scalar features ({len(processed_data['scalar_features'])} total):")
        for key, value in processed_data['scalar_features'].items():
            print(f"    {key}: {value:.4f}")
        
        # Save processed data
        output_file = f'{performance_name}_preprocessed.json'
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"\\n✅ PREPROCESSING TEST COMPLETE")
        print(f"Results saved to: {output_file}")
        print(f"\\n🎯 Ready for ML training pipeline!")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None

if __name__ == "__main__":
    test_preprocessing_pipeline()