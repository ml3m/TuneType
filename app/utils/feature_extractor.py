import librosa
import numpy as np
import logging
import os
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)

def extract_features(file_path, duration=30, offset=0):
    """
    Extract features from an audio file for genre classification.
    This function must output exactly 135 features to match the model's expectations.
    
    Args:
        file_path (str): Path to the audio file
        duration (int): Duration in seconds to analyze from the file
        offset (int): Starting point in seconds for analysis
        
    Returns:
        numpy.ndarray: Processed features ready for model prediction (135 features)
    """
    logger.info(f"Extracting features from {file_path}")
    
    try:
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load the audio file with appropriate parameters
        y, sr = librosa.load(file_path, duration=duration, offset=offset, sr=22050)
        logger.info(f"Audio loaded: {len(y)/sr:.2f}s at {sr}Hz")
        
        # If audio is too short, pad it
        if len(y) < sr * duration:
            y = np.pad(y, (0, int(sr * duration) - len(y)), 'constant')
        
        # Separate harmonic and percussive components for better genre detection
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Initialize the feature vector with exactly 135 features
        features = []
        
        # 1. Mel-frequency cepstral coefficients (MFCCs) - 26 features
        # Increased from 13 to capture more timbre characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)  # 13 features
        mfccs_var = np.var(mfccs, axis=1)    # 13 features
        for val in mfccs_mean:
            features.append(float(val))
        for val in mfccs_var:
            features.append(float(val))
        
        # 2. Spectral Centroid - 4 features with skewness and kurtosis
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(float(np.mean(spectral_centroid)))
        features.append(float(np.var(spectral_centroid)))
        features.append(float(skew(spectral_centroid)))
        features.append(float(kurtosis(spectral_centroid)))
        
        # 3. Spectral Rolloff - 3 features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(float(np.mean(spectral_rolloff)))
        features.append(float(np.var(spectral_rolloff)))
        features.append(float(skew(spectral_rolloff)))
        
        # 4. Spectral Contrast - 12 features (reduced from 14)
        # Using 6 bands for spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)  # 7 features (6 bands + 1)
        spectral_contrast_var = np.var(spectral_contrast, axis=1)    # 5 features (taking only 5 of the variances)
        for val in spectral_contrast_mean:
            features.append(float(val))
        for val in spectral_contrast_var[:5]:  # Only use 5 variance features
            features.append(float(val))
        
        # 5. Spectral Bandwidth - 3 features
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.append(float(np.mean(spectral_bandwidth)))
        features.append(float(np.var(spectral_bandwidth)))
        features.append(float(skew(spectral_bandwidth)))
        
        # 6. Zero Crossing Rate - 3 features
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(float(np.mean(zcr)))
        features.append(float(np.var(zcr)))
        features.append(float(skew(zcr)))
        
        # 7. Root Mean Square Energy - 3 features
        rms = librosa.feature.rms(y=y)[0]
        features.append(float(np.mean(rms)))
        features.append(float(np.var(rms)))
        features.append(float(skew(rms)))
        
        # 8. Rhythm features - 6 features
        # Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))
        
        # Beat statistics (important for genre detection)
        if len(beats) > 1:
            beat_diffs = np.diff(beats)
            features.append(float(np.mean(beat_diffs)))
            features.append(float(np.std(beat_diffs)))
            features.append(float(skew(beat_diffs) if len(beat_diffs) > 2 else 0))
        else:
            # Padding for insufficient beats
            features.extend([0.0, 0.0, 0.0])
        
        # Add low-energy ratio (useful for distinguishing certain genres)
        # Percentage of frames with RMS less than average RMS
        rms_mean = np.mean(rms)
        low_energy = np.mean(rms < rms_mean)
        features.append(float(low_energy))
        
        # Ratio of harmonic to percussive energy (good for distinguishing electronic vs acoustic)
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        if percussive_energy > 0:
            ratio = harmonic_energy / percussive_energy
        else:
            ratio = 0.0
        features.append(float(ratio))
        
        # 9. Chroma Features - 24 features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # 12 features
        chroma_var = np.var(chroma, axis=1)    # 12 features
        for val in chroma_mean:
            features.append(float(val))
        for val in chroma_var:
            features.append(float(val))
        
        # 10. Mel Spectrogram - 48 features
        # We aim for 135 total features, so we need 135 - 87 = 48 more from mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=24)
        
        # Ensure mel_spec has 24 bands
        if mel_spec.shape[0] < 24:
            mel_pad = np.zeros((24, mel_spec.shape[1]))
            mel_pad[:mel_spec.shape[0], :] = mel_spec
            mel_spec = mel_pad
        
        mel_spec_mean = np.mean(mel_spec, axis=1)  # 24 features
        mel_spec_var = np.var(mel_spec, axis=1)    # 24 features
        
        for val in mel_spec_mean:
            features.append(float(val))
        for val in mel_spec_var:
            features.append(float(val))
        
        # Count features for debugging
        logger.debug(f"Feature count before check: {len(features)}")
        
        # Ensuring we have exactly 135 features
        features = np.array(features, dtype=np.float32)
        
        if len(features) != 135:
            logger.warning(f"Feature count mismatch: got {len(features)}, expected 135")
            if len(features) > 135:
                # Truncate to 135 features
                features = features[:135]
            else:
                # Pad with zeros to 135 features
                features = np.pad(features, (0, 135 - len(features)), 'constant')
        
        # Debug check for NaN or infinity values that could break the model
        if np.isnan(features).any() or np.isinf(features).any():
            problematic_indices = np.where(np.isnan(features) | np.isinf(features))[0]
            logger.warning(f"Features contain NaN or Inf values at indices: {problematic_indices}")
            # Replace problematic values with zeros
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Features extracted successfully: shape={features.shape}, count={len(features)}")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None 