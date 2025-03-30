#!/usr/bin/env python3
import os
import argparse
import numpy as np
import logging
import librosa
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from app.utils.feature_extractor import extract_features, extract_multi_segment_features
from enhanced_genre_model import EnhancedGenreClassifier
import scipy.signal
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio_file(args):
    """
    Process a single audio file for parallel execution
    
    Args:
        args (tuple): (file_path, genre, genre_to_idx, genre_specific_augment_factor, advanced_augmentation)
        
    Returns:
        tuple: (features_list, labels_list)
    """
    file_path, genre, genre_to_idx, genre_specific_augment_factor, advanced_augmentation = args
    features_list = []
    labels_list = []
    
    try:
        # Extract features from multiple segments
        segments = extract_multi_segment_features(file_path, num_segments=2)
        
        if segments:
            # Add original segments
            for features in segments:
                features_list.append(features)
                labels_list.append(genre_to_idx[genre])
            
            # Only apply augmentation if we need more samples
            if advanced_augmentation and genre_specific_augment_factor > 1:
                # Load audio for augmentation
                y, sr = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
                
                # Apply different augmentations based on the genre
                for i in range(genre_specific_augment_factor - 1):
                    # Choose simple augmentation techniques for speed
                    aug_types = ['pitch', 'speed', 'noise']
                    
                    aug_type = random.choice(aug_types)
                    y_aug = np.copy(y)
                    
                    if aug_type == 'pitch':
                        # Pitch shift
                        shift = random.uniform(-3, 3)  # Shift by up to 3 semitones
                        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
                    elif aug_type == 'speed':
                        # Time stretch
                        rate = random.uniform(0.8, 1.2)
                        y_aug = librosa.effects.time_stretch(y, rate=rate)
                        
                        # Ensure consistent length
                        if len(y_aug) > len(y):
                            y_aug = y_aug[:len(y)]
                        elif len(y_aug) < len(y):
                            y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)))
                    else:  # noise
                        # Add noise
                        noise_level = random.uniform(0.001, 0.005)
                        noise = np.random.randn(len(y)) * noise_level
                        y_aug = y + noise
                    
                    # Extract features from augmented audio
                    tmp_path = f"/tmp/aug_sample_{genre}_{random.randint(1000, 9999)}.wav"
                    try:
                        sf.write(tmp_path, y_aug, sr)
                        
                        # Extract features directly
                        aug_features = extract_features(tmp_path, advanced_mode=True)
                            
                        # Clean up temporary file
                        os.remove(tmp_path)
                        
                        if aug_features is not None:
                            features_list.append(aug_features)
                            labels_list.append(genre_to_idx[genre])
                    except Exception as e:
                        logger.error(f"Error processing augmented audio: {str(e)}")
                        # Clean up in case of error
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
    
    return features_list, labels_list

def load_and_augment_data(sample_dir, augment_factor=3, advanced_augmentation=True):
    """
    Load and augment audio data from sample directory
    
    Args:
        sample_dir (str): Directory containing genre samples
        augment_factor (int): How many augmented samples to create per original
        advanced_augmentation (bool): Whether to use advanced augmentation techniques
        
    Returns:
        tuple: (features, labels, genres)
    """
    logger.info(f"Loading and augmenting data from {sample_dir} with {augment_factor}x augmentation")
    
    # Get all genre directories
    genre_dirs = [d for d in os.listdir(sample_dir) 
                 if os.path.isdir(os.path.join(sample_dir, d))
                 and not d.startswith('.')]
    
    if not genre_dirs:
        logger.error(f"No genre directories found in {sample_dir}")
        return None, None, None
    
    # Map genres to integer labels
    genres = sorted(genre_dirs)
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
    logger.info(f"Genre counts: {genre_counts}")
    logger.info(f"Maximum samples in a genre: {max_count}")
    
    # Extract features from each genre with appropriate augmentation
    all_features = []
    all_labels = []
    
    # Determine number of processes to use (leave one CPU free)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {num_processes} processes for parallel extraction")
    
    for genre in genres:
        genre_dir = os.path.join(sample_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))
                      and not f.startswith('.')]
        
        if not audio_files:
            logger.warning(f"No audio files found in {genre_dir}, skipping genre")
            continue
        
        # Determine genre-specific augmentation factor to balance dataset
        genre_specific_augment_factor = max(
            augment_factor,
            min(8, int(max_count / len(audio_files))) if len(audio_files) > 0 else augment_factor
        )
        
        logger.info(f"Processing {len(audio_files)} files for '{genre}' with {genre_specific_augment_factor}x augmentation")
        
        # Prepare arguments for parallel processing
        process_args = [
            (os.path.join(genre_dir, audio_file), genre, genre_to_idx, genre_specific_augment_factor, advanced_augmentation)
            for audio_file in audio_files
        ]
        
        # Process audio files in parallel
        genre_features = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_audio_file, args) for args in process_args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {genre}"):
                features_list, labels_list = future.result()
                all_features.extend(features_list)
                all_labels.extend(labels_list)
                genre_features.extend(features_list)
        
        logger.info(f"Extracted {len(genre_features)} features for genre '{genre}' (original + augmented)")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    logger.info(f"Total features: {len(X)} from {len(genres)} genres")
    
    return X, y, genres

def plot_learning_curve(history):
    """Plot and save the learning curve from training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.get('accuracy', []))
    plt.plot(history.get('val_accuracy', []))
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.get('loss', []))
    plt.plot(history.get('val_loss', []))
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('enhanced_learning_curve.png')
    logger.info("Learning curve saved to enhanced_learning_curve.png")

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Music Genre Classifier')
    parser.add_argument('--sample-dir', type=str, default='samples',
                       help='Directory containing genre samples')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Base augmentation factor for samples')
    parser.add_argument('--no-advanced-aug', action='store_true',
                       help='Disable advanced augmentation techniques')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--test-file', type=str,
                       help='Test file to classify after training')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Set environment variable to control parallelism in librosa
    if not args.no_parallel:
        os.environ['NUMBA_NUM_THREADS'] = str(max(1, multiprocessing.cpu_count() - 1))
    
    # Load and augment data
    X, y, genres = load_and_augment_data(
        args.sample_dir, 
        args.augment_factor,
        not args.no_advanced_aug
    )
    
    if X is None or len(X) == 0:
        logger.error("No features extracted, aborting")
        return
    
    # Create enhanced classifier
    classifier = EnhancedGenreClassifier(genres=genres)
    
    # Train the model
    logger.info(f"Training model with {len(X)} samples for {args.epochs} epochs")
    history = classifier.train(X, y, sample_dir=args.sample_dir, epochs=args.epochs, batch_size=args.batch_size)
    
    # Plot learning curve
    plot_learning_curve(history)
    
    # Split data for tuning genre weights
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Tune genre weights
    logger.info("Tuning genre weights to improve accuracy")
    classifier.tune_genre_weights(X_val, y_val)
    
    # Test if a test file was provided
    if args.test_file:
        logger.info(f"Testing model on {args.test_file}")
        
        # Extract features
        features = extract_multi_segment_features(args.test_file)
        
        if not features:
            logger.error(f"Failed to extract features from {args.test_file}")
            return
        
        # Make prediction
        result = classifier.predict(features)
        
        # Sort genres by probability
        sorted_genres = sorted(result.items(), key=lambda x: x[1], reverse=True)
        
        # Print results
        print(f"\nPredictions for {os.path.basename(args.test_file)}:")
        for genre, prob in sorted_genres:
            print(f"{genre}: {prob*100:.2f}%")
    
    logger.info("Enhanced model training complete")

if __name__ == "__main__":
    main() 