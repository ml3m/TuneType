#!/usr/bin/env python3
import os
import argparse
import numpy as np
import logging
import librosa
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import pandas as pd
import tensorflow as tf
from app.utils.model import GenreClassifier
from app.utils.feature_extractor import extract_features, extract_multi_segment_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_and_augment_features(sample_dir, augment_factor=3):
    """
    Extract features from audio samples with extensive data augmentation
    
    Args:
        sample_dir (str): Directory containing genre samples
        augment_factor (int): How many augmented samples to create per real sample
        
    Returns:
        tuple: (features, labels, genres)
    """
    # Get all genre directories
    genre_dirs = [d for d in os.listdir(sample_dir) 
                 if os.path.isdir(os.path.join(sample_dir, d))
                 and not d.startswith('.')]
    
    if not genre_dirs:
        logger.error(f"No genre directories found in {sample_dir}")
        return None, None, None
    
    # Map genres to integer labels
    genres = sorted(genre_dirs)  # Sort to ensure consistent ordering
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    
    # Count samples per genre to determine augmentation needs
    genre_counts = {}
    for genre in genres:
        genre_dir = os.path.join(sample_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))
                      and not f.startswith('.')]
        genre_counts[genre] = len(audio_files)
    
    # Find the genre with the most samples to determine target count
    max_count = max(genre_counts.values()) if genre_counts else 0
    
    # Extract features from each genre, with appropriate augmentation
    all_features = []
    all_labels = []
    
    for genre in genres:
        genre_dir = os.path.join(sample_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))
                      and not f.startswith('.')]
        
        if not audio_files:
            logger.warning(f"No audio files found in {genre_dir}, skipping genre")
            continue
        
        # Determine augmentation factor for this genre to balance dataset
        genre_specific_augment_factor = max(
            augment_factor,
            min(10, int(max_count / len(audio_files)))  # Cap at 10x
        )
        
        logger.info(f"Processing {len(audio_files)} audio files for genre '{genre}' with {genre_specific_augment_factor}x augmentation")
        
        # Process each audio file
        genre_features = []
        for audio_file in tqdm(audio_files, desc=f"Extracting {genre}"):
            file_path = os.path.join(genre_dir, audio_file)
            
            try:
                # Extract standard features
                features = extract_features(file_path, advanced_mode=True)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(genre_to_idx[genre])
                    genre_features.append(features)
                
                # Basic augmentation: process with time shifts
                for offset in [0.5, 1.0, 1.5]:
                    features = extract_features(file_path, offset=offset, advanced_mode=True)
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(genre_to_idx[genre])
                        genre_features.append(features)
                
                # For genres with fewer samples, apply more advanced augmentation
                if genre_specific_augment_factor > augment_factor:
                    # Load the audio file for advanced augmentation
                    y, sr = librosa.load(file_path, sr=22050)
                    
                    # Apply more advanced augmentations
                    for i in range(genre_specific_augment_factor - augment_factor):
                        # Choose random augmentation type
                        aug_type = random.choice(['pitch', 'speed', 'noise', 'filter', 'combined'])
                        
                        if aug_type == 'pitch':
                            # Pitch shift
                            shift = random.uniform(-2, 2)  # Shift by up to 2 semitones
                            y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
                        
                        elif aug_type == 'speed':
                            # Time stretch
                            rate = random.uniform(0.8, 1.2)  # Speed up or slow down
                            y_aug = librosa.effects.time_stretch(y, rate=rate)
                            # Ensure consistent length
                            if len(y_aug) > len(y):
                                y_aug = y_aug[:len(y)]
                            elif len(y_aug) < len(y):
                                y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)))
                        
                        elif aug_type == 'noise':
                            # Add noise
                            noise_level = random.uniform(0.001, 0.005)
                            noise = np.random.randn(len(y)) * noise_level
                            y_aug = y + noise
                        
                        elif aug_type == 'filter':
                            # Apply random filter
                            filter_type = random.choice(['lowpass', 'highpass'])
                            if filter_type == 'lowpass':
                                cutoff = random.uniform(0.5, 0.9)  # Fraction of Nyquist frequency
                                b = librosa.filters.get_window('hann', 15)
                                a = [1]
                                y_aug = librosa.util.filter_audio(y, b, a)
                            else:
                                cutoff = random.uniform(0.05, 0.2)
                                y_aug = y - librosa.util.filter_audio(y, librosa.filters.get_window('hann', 15), [1])
                        
                        elif aug_type == 'combined':
                            # Combine multiple augmentations
                            # Pitch shift
                            shift = random.uniform(-1, 1)
                            y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
                            # Speed change
                            rate = random.uniform(0.9, 1.1)
                            y_aug = librosa.effects.time_stretch(y_aug, rate=rate)
                            # Ensure consistent length
                            if len(y_aug) > len(y):
                                y_aug = y_aug[:len(y)]
                            elif len(y_aug) < len(y):
                                y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)))
                            # Add slight noise
                            noise_level = random.uniform(0.001, 0.003)
                            noise = np.random.randn(len(y_aug)) * noise_level
                            y_aug = y_aug + noise
                        
                        # Extract features from augmented audio
                        # We need to save a temporary file since our extractor works with files
                        tmp_path = f"/tmp/aug_sample_{random.randint(1000, 9999)}.wav"
                        librosa.output.write_wav(tmp_path, y_aug, sr)
                        
                        # Extract features
                        try:
                            features = extract_features(tmp_path, advanced_mode=True)
                            # Clean up temporary file
                            os.remove(tmp_path)
                            
                            if features is not None:
                                all_features.append(features)
                                all_labels.append(genre_to_idx[genre])
                                genre_features.append(features)
                        except Exception as e:
                            logger.error(f"Error extracting features from augmented audio: {e}")
                            # Clean up in case of error
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Extracted {len(genre_features)} features for genre '{genre}' (original + augmented)")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    logger.info(f"Total extracted features: {len(X)} from {len(genres)} genres")
    
    return X, y, genres

def plot_confusion_matrix(y_true, y_pred, genres, output_file='confusion_matrix.png'):
    """Plot and save confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    
    # Create normalized confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Handle divisions by zero
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=genres, yticklabels=genres)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Confusion matrix saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Music Genre Classification Training')
    parser.add_argument('--sample-dir', type=str, default='samples', 
                       help='Directory containing genre samples')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Basic augmentation factor (will be increased for underrepresented genres)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Portion of data to use for testing')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs for deep learning model training')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for deep learning model training')
    parser.add_argument('--use-deep-learning', action='store_true',
                       help='Use advanced deep learning model with transfer learning')
    parser.add_argument('--calibrate', action='store_true',
                       help='Apply automatic calibration after training')
    
    args = parser.parse_args()
    
    # Extract features with augmentation
    logger.info("Extracting and augmenting features from samples...")
    X, y, genres = extract_and_augment_features(args.sample_dir, args.augment_factor)
    
    if X is None or len(X) == 0:
        logger.error("No features extracted, aborting")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    # Initialize model
    model = GenreClassifier(genres=genres, use_deep_learning=args.use_deep_learning)
    
    # Train model
    training_results = model.train(
        X_train, y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        cross_validation=True
    )
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    
    # Get predictions
    test_preds = []
    for features in X_test:
        # Reshape to match expected input format
        features_reshaped = features.reshape(1, -1)
        # Get probabilities
        probs = model.predict(features_reshaped)
        # Convert to class index
        pred_class = np.argmax(probs[0])
        test_preds.append(pred_class)
    
    # Calculate accuracy
    accuracy = balanced_accuracy_score(y_test, test_preds)
    logger.info(f"Test set balanced accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, test_preds, genres)
    
    # Generate classification report
    try:
        report = classification_report(y_test, test_preds, target_names=genres)
        logger.info(f"Classification report:\n{report}")
        
        # Save report to CSV
        report_dict = classification_report(y_test, test_preds, target_names=genres, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv('classification_report.csv')
        logger.info("Classification report saved to classification_report.csv")
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
    
    # Apply calibration if requested
    if args.calibrate:
        logger.info("Applying automatic genre calibration...")
        try:
            import tune_genre_bias
            tune_genre_bias.main(['--sample-dir', args.sample_dir])
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
    
if __name__ == "__main__":
    main() 