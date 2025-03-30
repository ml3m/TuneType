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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import tensorflow as tf
from enhanced_genre_model import EnhancedGenreClassifier
from app.utils.feature_extractor import extract_features, extract_multi_segment_features
import soundfile as sf
import glob
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio_file(args):
    """Process a single audio file to extract features and augment if needed"""
    file_path, genre, genre_to_idx, augment_factor = args
    features_list = []
    labels_list = []
    
    try:
        # Extract features from original audio
        segments = extract_multi_segment_features(file_path, num_segments=2)
        
        if segments:
            # Add original segments
            for features in segments:
                features_list.append(features)
                labels_list.append(genre_to_idx[genre])
            
            # Apply augmentation if needed
            if augment_factor > 1:
                y, sr = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
                
                # Apply different augmentations based on the augment factor
                for i in range(augment_factor - 1):
                    y_aug = np.copy(y)
                    
                    # Choose augmentation type randomly
                    aug_type = random.choice(['pitch', 'speed', 'noise', 'time_stretch'])
                    
                    if aug_type == 'pitch':
                        # Pitch shift - different ranges for different genres
                        if genre in ['classical', 'jazz']:
                            shift = random.uniform(-2, 2)  # Smaller shift for pitch-sensitive genres
                        else:
                            shift = random.uniform(-4, 4)  # Larger shift for others
                        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
                    
                    elif aug_type == 'speed':
                        # Speed change
                        rate = random.uniform(0.8, 1.2)
                        y_aug = librosa.effects.time_stretch(y, rate=rate)
                        
                        # Ensure consistent length
                        if len(y_aug) > len(y):
                            y_aug = y_aug[:len(y)]
                        elif len(y_aug) < len(y):
                            y_aug = np.pad(y_aug, (0, len(y) - len(y_aug)))
                    
                    elif aug_type == 'noise':
                        # Add noise - lower for classical/jazz
                        if genre in ['classical', 'jazz']:
                            noise_level = random.uniform(0.0005, 0.002)
                        else:
                            noise_level = random.uniform(0.001, 0.005)
                        noise = np.random.randn(len(y)) * noise_level
                        y_aug = y + noise
                    
                    else:  # time_stretch
                        # Time stretching with pitch preservation
                        segments = []
                        segment_length = len(y) // 3
                        
                        for seg_idx in range(3):
                            start = seg_idx * segment_length
                            end = start + segment_length
                            segment = y[start:end]
                            
                            # Apply different stretching to each segment
                            stretch_rate = random.uniform(0.9, 1.1)
                            stretched = librosa.effects.time_stretch(segment, rate=stretch_rate)
                            
                            # Pad or trim to original length
                            if len(stretched) > segment_length:
                                stretched = stretched[:segment_length]
                            elif len(stretched) < segment_length:
                                stretched = np.pad(stretched, (0, segment_length - len(stretched)))
                            
                            segments.append(stretched)
                        
                        # Combine segments
                        y_aug = np.concatenate(segments)
                        
                        # Handle any remaining samples
                        remainder = y[3 * segment_length:]
                        if len(remainder) > 0:
                            y_aug = np.concatenate([y_aug, remainder])
                    
                    # Save augmented audio temporarily
                    tmp_path = f"/tmp/aug_{genre}_{random.randint(1000, 9999)}.wav"
                    sf.write(tmp_path, y_aug, sr)
                    
                    try:
                        # Extract features from augmented audio
                        aug_features = extract_features(tmp_path, advanced_mode=True)
                        
                        if aug_features is not None:
                            features_list.append(aug_features)
                            labels_list.append(genre_to_idx[genre])
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
    
    return features_list, labels_list

def load_and_process_data(sample_dir, max_files_per_genre=100, augment_factor=3):
    """Load audio data, balance the dataset, and extract features"""
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
    
    # Count files per genre
    genre_files = {}
    for genre in genres:
        genre_dir = os.path.join(sample_dir, genre)
        audio_files = glob.glob(os.path.join(genre_dir, "*.mp3")) + \
                      glob.glob(os.path.join(genre_dir, "*.wav")) + \
                      glob.glob(os.path.join(genre_dir, "*.ogg")) + \
                      glob.glob(os.path.join(genre_dir, "*.flac"))
        genre_files[genre] = audio_files
    
    # Calculate min and max files count
    min_files = min(len(files) for files in genre_files.values())
    max_files = max(len(files) for files in genre_files.values())
    
    logger.info(f"Found {len(genres)} genres: {genres}")
    logger.info(f"Files per genre - Min: {min_files}, Max: {max_files}")
    
    # Determine if we need to augment based on the file count
    augment_factors = {}
    for genre, files in genre_files.items():
        if len(files) < 20:  # Very few files
            augment_factors[genre] = min(8, augment_factor * 2)
        elif len(files) < 50:  # Few files
            augment_factors[genre] = augment_factor
        else:  # Enough files
            augment_factors[genre] = max(1, augment_factor // 2)
    
    logger.info(f"Augmentation factors per genre: {augment_factors}")
    
    # Process audio files in parallel
    all_features = []
    all_labels = []
    
    # Determine number of processes to use
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    for genre in genres:
        files = genre_files[genre]
        
        # Limit number of files if needed
        if max_files_per_genre and len(files) > max_files_per_genre:
            logger.info(f"Limiting {genre} files from {len(files)} to {max_files_per_genre}")
            files = random.sample(files, max_files_per_genre)
        
        genre_augment_factor = augment_factors[genre]
        logger.info(f"Processing {len(files)} files for '{genre}' with {genre_augment_factor}x augmentation")
        
        # Prepare arguments for parallel processing
        process_args = [
            (file_path, genre, genre_to_idx, genre_augment_factor)
            for file_path in files
        ]
        
        # Process in parallel
        genre_features = []
        genre_labels = []
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_audio_file, args) for args in process_args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{genre}"):
                features_list, labels_list = future.result()
                all_features.extend(features_list)
                all_labels.extend(labels_list)
                genre_features.extend(features_list)
                genre_labels.extend(labels_list)
        
        logger.info(f"Extracted {len(genre_features)} features for '{genre}'")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    logger.info(f"Total dataset: {len(X)} samples across {len(genres)} genres")
    
    # Log class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    class_distribution = {genres[label]: count for label, count in zip(unique_labels, counts)}
    logger.info(f"Class distribution: {class_distribution}")
    
    return X, y, genres

def main():
    parser = argparse.ArgumentParser(description='Improved Music Genre Classifier Training')
    parser.add_argument('--sample-dir', type=str, default='samples',
                       help='Directory containing genre samples')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Base augmentation factor for samples')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max-files', type=int, default=100,
                       help='Maximum files to use per genre (for balance)')
    parser.add_argument('--test-file', type=str,
                       help='Test file to classify after training')
    
    args = parser.parse_args()
    
    # Set memory growth for GPUs if available
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info(f"Found {len(physical_devices)} GPU(s), enabled memory growth")
    except:
        logger.info("No GPU available or unable to set memory growth")
    
    # Load and process data
    X, y, genres = load_and_process_data(
        args.sample_dir, 
        max_files_per_genre=args.max_files,
        augment_factor=args.augment_factor
    )
    
    if X is None or len(X) == 0:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Initialize classifier with the genres we have
    classifier = EnhancedGenreClassifier(genres=genres)
    
    # Train model
    history = classifier.train(
        X, y, 
        sample_dir=args.sample_dir,
        epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.get('accuracy', []))
    plt.plot(history.get('val_accuracy', []))
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.get('loss', []))
    plt.plot(history.get('val_loss', []))
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    
    # Test on file if provided
    if args.test_file and os.path.exists(args.test_file):
        logger.info(f"Testing on file: {args.test_file}")
        
        # Extract features
        features = extract_multi_segment_features(args.test_file, num_segments=5)
        
        if features:
            # Make prediction
            result = classifier.predict(features)
            
            # Sort and display results
            sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
            
            print("\nPrediction Results:")
            print("=" * 40)
            for genre, prob in sorted_results:
                print(f"{genre.ljust(15)}: {prob*100:6.2f}%")
            print("=" * 40)

if __name__ == "__main__":
    main() 