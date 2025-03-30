import librosa
import numpy as np
import logging
import os
from scipy.stats import skew, kurtosis
import librosa.display
import scipy.signal
import random
import soundfile as sf
from functools import lru_cache

logger = logging.getLogger(__name__)

# Audio file loading with caching to avoid reloading the same file
@lru_cache(maxsize=32)
def cached_load(file_path, sr=22050, duration=None, offset=0):
    return librosa.load(file_path, sr=sr, duration=duration, offset=offset, res_type='kaiser_fast')

def extract_features(file_path, duration=20, offset=0, advanced_mode=True):
    """
    Extract features from an audio file for genre classification.
    This function must output exactly 135 features to match the model's expectations.
    
    Args:
        file_path (str): Path to the audio file
        duration (int): Duration in seconds to analyze from the file
        offset (int): Starting point in seconds for analysis
        advanced_mode (bool): Whether to use advanced feature extraction techniques
        
    Returns:
        numpy.ndarray: Processed features ready for model prediction (135 features)
    """
    logger.info(f"Extracting features from {file_path}")
    
    try:
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load the audio file with appropriate parameters - use res_type='kaiser_fast' for speed
        y, sr = cached_load(file_path, sr=22050, duration=duration, offset=offset)
        logger.info(f"Audio loaded: {len(y)/sr:.2f}s at {sr}Hz")
        
        # If audio is too short, pad it
        if len(y) < sr * duration:
            y = np.pad(y, (0, int(sr * duration) - len(y)), 'constant')
        
        # Use shorter hop_length for faster computation
        hop_length = 1024  # Default is 512, increasing to 1024 makes computation ~2x faster
        
        # Separate harmonic and percussive components for better genre detection
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Initialize the feature vector with exactly 135 features
        features = []
        
        # 1. Mel-frequency cepstral coefficients (MFCCs) - 26 features
        # Increased from 13 to capture more timbre characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs, axis=1)  # 13 features
        mfccs_var = np.var(mfccs, axis=1)    # 13 features
        for val in mfccs_mean:
            features.append(float(val))
        for val in mfccs_var:
            features.append(float(val))
        
        # 2. Spectral Centroid - 4 features with skewness and kurtosis
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        features.append(float(np.mean(spectral_centroid)))
        features.append(float(np.var(spectral_centroid)))
        features.append(float(skew(spectral_centroid)))
        
        # 3. Spectral Rolloff - 3 features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        features.append(float(np.mean(spectral_rolloff)))
        features.append(float(np.var(spectral_rolloff)))
        features.append(float(skew(spectral_rolloff)))
        
        # 4. Spectral Contrast - 12 features (across more bands for better genre discrimination)
        # Using 6 bands for spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, hop_length=hop_length)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)  # 7 features (6 bands + 1)
        spectral_contrast_var = np.var(spectral_contrast, axis=1)    # 5 features (taking only 5 of the variances)
        for val in spectral_contrast_mean:
            features.append(float(val))
        for val in spectral_contrast_var[:5]:  # Only use 5 variance features
            features.append(float(val))
        
        # 5. Spectral Bandwidth - 3 features
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        features.append(float(np.mean(spectral_bandwidth)))
        features.append(float(np.var(spectral_bandwidth)))
        features.append(float(skew(spectral_bandwidth)))
        
        # 6. Zero Crossing Rate - 3 features (especially important for distinguishing metal/rock)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        features.append(float(np.mean(zcr)))
        features.append(float(np.var(zcr)))
        features.append(float(skew(zcr)))
        
        # 7. Root Mean Square Energy - 3 features (important for detecting high energy genres)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        features.append(float(np.mean(rms)))
        features.append(float(np.var(rms)))
        features.append(float(skew(rms)))
        
        if advanced_mode:
            # Advanced rhythm analysis
            y_percussive_hpss, _ = librosa.effects.hpss(y_percussive)  # Double HPSS for cleaner rhythm
            onset_env = librosa.onset.onset_strength(y=y_percussive_hpss, sr=sr)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            
            # Tempo estimation with dynamic range
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            tempo_mean = np.mean(tempogram, axis=1)
            tempo_var = np.var(tempogram, axis=1)
            tempo_features = np.concatenate([tempo_mean[:5], tempo_var[:4]])
        else:
            # 8. Regular rhythm features - 10 features (expanded from 6)
            # Calculate tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features.append(float(tempo))
            
            # Beat statistics (important for genre detection)
            if len(beats) > 2:
                beat_diffs = np.diff(beats)
                features.append(float(np.mean(beat_diffs)))
                features.append(float(np.std(beat_diffs)))
                features.append(float(skew(beat_diffs) if len(beat_diffs) > 2 else 0))
                features.append(float(kurtosis(beat_diffs) if len(beat_diffs) > 2 else 0))
                # Rhythm regularity - coefficient of variation of beat intervals
                features.append(float(np.std(beat_diffs) / np.mean(beat_diffs) if np.mean(beat_diffs) > 0 else 0))
            else:
                # Padding for insufficient beats
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Add low-energy ratio (useful for distinguishing certain genres)
            # Percentage of frames with RMS less than average RMS
            rms_mean = np.mean(rms)
            low_energy = np.mean(rms < rms_mean)
            features.append(float(low_energy))
            
            # Ratio of harmonic to percussive energy (crucial for electronic/acoustic distinction)
            harmonic_energy = np.sum(y_harmonic**2)
            percussive_energy = np.sum(y_percussive**2)
            
            if percussive_energy > 0:
                h_to_p_ratio = harmonic_energy / percussive_energy
            else:
                h_to_p_ratio = 1.0
            features.append(float(h_to_p_ratio))
            
            # Add percussive energy ratio (excellent for detecting hip-hop, electronic, metal)
            total_energy = harmonic_energy + percussive_energy
            if total_energy > 0:
                percussive_ratio = percussive_energy / total_energy
            else:
                percussive_ratio = 0.5
            features.append(float(percussive_ratio))
        
        # 9. Chroma Features - 24 features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)  # 12 features
        chroma_var = np.var(chroma, axis=1)    # 12 features
        for val in chroma_mean:
            features.append(float(val))
        for val in chroma_var:
            features.append(float(val))
        
        # 10. Improved Mel Spectrogram Features - 24 features instead of 48
        # Using fewer but more informative mel features
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12, hop_length=hop_length)
        
        # Ensure mel_spec has 12 bands
        if mel_spec.shape[0] < 12:
            mel_pad = np.zeros((12, mel_spec.shape[1]))
            mel_pad[:mel_spec.shape[0], :] = mel_spec
            mel_spec = mel_pad
        
        mel_spec_mean = np.mean(mel_spec, axis=1)  # 12 features
        mel_spec_var = np.var(mel_spec, axis=1)    # 12 features
        
        for val in mel_spec_mean:
            features.append(float(val))
        for val in mel_spec_var:
            features.append(float(val))
        
        # 11. Spectral Flatness - 3 features (helps detect electronic/synthetic music)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        features.append(float(np.mean(flatness)))
        features.append(float(np.var(flatness)))
        features.append(float(skew(flatness)))
        
        if advanced_mode:
            # NEW: Specific genre discriminators
            
            # Metal/Rock detection: High frequency energy ratio
            spec = np.abs(librosa.stft(y))
            high_freq_energy = np.sum(spec[int(spec.shape[0]*0.6):, :])
            total_spec_energy = np.sum(spec)
            high_freq_ratio = high_freq_energy / total_spec_energy if total_spec_energy > 0 else 0.5
            features.append(float(high_freq_ratio))
            
            # Classical detection: Dynamics range
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            frame_rms = np.sqrt(np.mean(frames**2, axis=0))
            dynamics_range = np.max(frame_rms) / (np.mean(frame_rms) if np.mean(frame_rms) > 0 else 1.0)
            features.append(float(dynamics_range))
            
            # Hip-hop detection: Bass energy ratio
            bass_energy = np.sum(spec[:int(spec.shape[0]*0.1), :])
            bass_ratio = bass_energy / total_spec_energy if total_spec_energy > 0 else 0.2
            features.append(float(bass_ratio))
            
            # Jazz detection: Harmonic complexity
            # Count significant peaks in chromagram
            chroma_peaks = np.mean([np.count_nonzero(frame > 0.7*np.max(frame)) for frame in chroma.T])
            features.append(float(chroma_peaks))
            
            # Electronic music: Repetitiveness
            # Compute novelty curve and its autocorrelation
            novelty = librosa.onset.onset_strength(y=y, sr=sr)
            ac = librosa.autocorrelate(novelty)
            # Normalized autocorrelation peak after zero
            if len(ac) > 1:
                ac_peak = np.max(ac[1:min(len(ac)//3, 100)]) / ac[0] if ac[0] > 0 else 0.5
            else:
                ac_peak = 0.5
            features.append(float(ac_peak))
            
            # For advanced rhythm features
            for val in tempo_features:
                features.append(float(val))
        else:
            # 12. Onset Strength and Statistics - 5 features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # Onset rate (onsets per second) - helps distinguish fast vs slow music
            onset_rate = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y) / sr)
            features.append(float(onset_rate))
            # Onset statistics
            features.append(float(np.mean(onset_env)))
            features.append(float(np.std(onset_env)))
            features.append(float(skew(onset_env)))
            features.append(float(np.max(onset_env) / (np.mean(onset_env) if np.mean(onset_env) > 0 else 1.0)))
            
            # 13. Spectral Flux - 2 features (change in spectrum over time)
            # Particularly good for distinguishing genres with rapid timbral changes
            spectral = librosa.stft(y)
            spectral_magnitude = np.abs(spectral)
            spectral_flux = np.diff(spectral_magnitude, axis=1)
            features.append(float(np.mean(np.mean(spectral_flux, axis=1))))
            features.append(float(np.std(np.mean(spectral_flux, axis=1))))
        
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

def extract_multi_segment_features(file_path, num_segments=3, segment_duration=10):
    """
    Extract features from multiple segments of an audio file for more robust prediction
    
    Args:
        file_path (str): Path to the audio file
        num_segments (int): Number of segments to extract
        segment_duration (int): Duration of each segment in seconds
        
    Returns:
        list: List of feature vectors, one for each segment
    """
    logger.info(f"Analyzing file with total duration: {segment_duration*num_segments}s")
    
    try:
        # Load the full audio first to determine its length
        y, sr = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
        total_duration = len(y) / sr
        
        logger.info(f"Audio loaded: {total_duration:.2f}s at {sr}Hz")
        
        # If the file is shorter than our target analysis duration, adjust segments
        if total_duration < segment_duration:
            logger.info(f"Audio too short ({total_duration:.2f}s), analyzing entire file")
            features = extract_features(file_path, duration=None, advanced_mode=True)
            if features is not None:
                return [features]  # Return as a list with single segment
            return []
            
        # For very long files, pick segments strategically
        if total_duration > segment_duration * num_segments * 2:
            logger.info(f"Long audio file detected ({total_duration:.2f}s), selecting diverse segments")
            
            # Strategy: Take segments from beginning, middle, and end to capture the song structure
            offsets = []
            
            # Beginning segment (always include)
            offsets.append(0)
            
            # Middle segments
            middle_start = max(30, (total_duration - segment_duration) / 2)  # Skip first 30s if possible
            offsets.append(middle_start)
            
            # Add segments at 1/4 and 3/4 points if we want more segments
            if num_segments > 3:
                offsets.append(total_duration * 0.25)
                offsets.append(total_duration * 0.75)
            
            # End segment (always include if enough duration)
            if total_duration > segment_duration + 30:  # Only if we have enough room
                offsets.append(total_duration - segment_duration - 5)  # 5s before the end
                
            # If we still need more segments, add some with random offsets
            while len(offsets) < num_segments:
                # Generate random offset, avoiding existing ones
                random_offset = random.uniform(15, total_duration - segment_duration - 5)
                
                # Check if it's far enough from existing offsets (at least 10s apart)
                if all(abs(random_offset - offset) > 10 for offset in offsets):
                    offsets.append(random_offset)
            
            # Sort offsets and take only what we need
            offsets = sorted(offsets)[:num_segments]
        else:
            # For shorter files, divide evenly
            logger.info(f"Dividing audio file into {num_segments} segments")
            usable_duration = min(total_duration, segment_duration * num_segments)
            offsets = np.linspace(0, total_duration - segment_duration, num=min(num_segments, 
                                 max(1, int(total_duration // segment_duration))))
        
        # Extract features for each segment
        logger.info(f"Extracting features from {len(offsets)} segments at offsets: {[f'{o:.1f}s' for o in offsets]}")
        features_list = []
        
        for offset in offsets:
            features = extract_features(file_path, duration=segment_duration, offset=offset, advanced_mode=True)
            if features is not None:
                features_list.append(features)
        
        return features_list
    
    except Exception as e:
        logger.error(f"Error extracting multi-segment features: {str(e)}")
        return []

def augment_audio(file_path):
    """
    Augment the audio file by applying a random filter
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        str: Path to the augmented audio file
    """
    try:
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the audio file with appropriate parameters
        y, sr = librosa.load(file_path, sr=22050)
        logger.info(f"Audio loaded: {len(y)/sr:.2f}s at {sr}Hz")
        
        # If audio is too short, pad it
        if len(y) < sr * 30:
            y = np.pad(y, (0, int(sr * 30) - len(y)), 'constant')
        
        filter_type = random.choice(['lowpass', 'highpass'])
        if filter_type == 'lowpass':
            cutoff = random.uniform(0.5, 0.9)
            b = librosa.filters.get_window('hann', 15)
            a = [1]
            # Replace filter_audio with scipy.signal.lfilter
            y_aug = scipy.signal.lfilter(b, a, y)
        else:
            cutoff = random.uniform(0.05, 0.2)
            b = librosa.filters.get_window('hann', 15)
            a = [1]
            y_aug = y - scipy.signal.lfilter(b, a, y)
        
        # Save the augmented audio file
        augmented_file_path = f"{file_path.rsplit('.', 1)[0]}_augmented.{file_path.rsplit('.', 1)[1]}"
        sf.write(augmented_file_path, y_aug, sr)
        
        logger.info(f"Augmented audio saved to: {augmented_file_path}")
        return augmented_file_path
        
    except Exception as e:
        logger.error(f"Error augmenting audio: {str(e)}")
        return None 