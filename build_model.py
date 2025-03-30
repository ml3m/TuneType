#!/usr/bin/env python3
import argparse
import os
import logging
import sys
import subprocess
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the major music genres we want to support
MAJOR_GENRES = [
    'blues', 'classical', 'country', 'electronic', 'folk', 
    'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'world'
]

def run_command(command, desc=None):
    """Run a shell command and log its output"""
    if desc:
        logger.info(desc)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                logger.info(line)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"Output: {e.stdout}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

def check_samples():
    """Check if we have samples for all major genres"""
    samples_dir = "samples"
    
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir, exist_ok=True)
        logger.info(f"Created samples directory: {samples_dir}")
        return False, []
    
    existing_genres = [d for d in os.listdir(samples_dir) 
                      if os.path.isdir(os.path.join(samples_dir, d)) 
                      and d in MAJOR_GENRES]
    
    missing_genres = [g for g in MAJOR_GENRES if g not in existing_genres]
    
    if missing_genres:
        logger.info(f"Missing samples for genres: {', '.join(missing_genres)}")
        return False, missing_genres
    
    # Check if we have enough samples in each genre
    min_samples = 10
    insufficient_genres = []
    
    for genre in existing_genres:
        genre_dir = os.path.join(samples_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg'))
                      and not f.startswith('.')]
        
        if len(audio_files) < min_samples:
            insufficient_genres.append((genre, len(audio_files)))
    
    if insufficient_genres:
        genres_info = ", ".join([f"{g} ({n} samples)" for g, n in insufficient_genres])
        logger.info(f"Insufficient samples for genres: {genres_info}")
        return False, missing_genres
    
    logger.info(f"All {len(existing_genres)} genres have sufficient samples")
    return True, []

def generate_synthetic_data(force=False):
    """Generate synthetic data for training"""
    samples_dir = "samples"
    
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir, exist_ok=True)
    
    # Create synthetic data for each genre
    logger.info("Generating synthetic training data...")
    run_command("python create_dummy_data.py", "Creating synthetic audio samples")
    
    logger.info("Synthetic data generation complete")
    return True

def download_real_samples(force=False):
    """Download real audio samples for training"""
    samples_dir = "samples"
    
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir, exist_ok=True)
    
    # Check if we already have some data
    if not force and os.path.exists(samples_dir):
        genre_dirs = [d for d in os.listdir(samples_dir) 
                     if os.path.isdir(os.path.join(samples_dir, d))]
        
        if genre_dirs and len(genre_dirs) >= 4:
            user_input = input(f"Found existing samples for {len(genre_dirs)} genres. Download more? (y/n): ")
            if user_input.lower() != 'y':
                logger.info("Skipping download of additional samples")
                return True
    
    # Run the download script
    logger.info("Downloading real audio samples...")
    run_command("python download_samples.py", "Downloading audio samples")
    
    logger.info("Sample download complete")
    return True

def train_model(force_extract=False):
    """Train the genre classification model"""
    logger.info("Starting model training...")
    
    # Train with all available samples
    run_command(
        f"python train_with_samples.py --force-extract={str(force_extract).lower()}", 
        "Training model with audio samples"
    )
    
    logger.info("Model training complete")
    return True

def test_model(test_file=None):
    """Test the trained model on a single file or samples from each genre"""
    if test_file:
        # Test on a specific file
        if not os.path.exists(test_file):
            logger.error(f"Test file not found: {test_file}")
            return False
        
        logger.info(f"Testing model on file: {test_file}")
        run_command(f"python test_model.py \"{test_file}\"", "Testing model on file")
        return True
    
    # Test on a sample from each genre
    logger.info("Testing model on samples from each genre...")
    
    samples_dir = "samples"
    if not os.path.exists(samples_dir):
        logger.error(f"Samples directory not found: {samples_dir}")
        return False
    
    genre_dirs = [d for d in os.listdir(samples_dir) 
                 if os.path.isdir(os.path.join(samples_dir, d))]
    
    for genre in genre_dirs:
        genre_dir = os.path.join(samples_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg'))
                      and not f.startswith('.')]
        
        if not audio_files:
            logger.warning(f"No audio files found for genre: {genre}")
            continue
        
        # Test on the first file
        test_file = os.path.join(genre_dir, audio_files[0])
        logger.info(f"Testing model on {genre} sample: {os.path.basename(test_file)}")
        run_command(f"python test_model.py \"{test_file}\"", f"Testing {genre} sample")
    
    logger.info("Model testing complete")
    return True

def main():
    parser = argparse.ArgumentParser(description='Build and train a genre classification model')
    parser.add_argument('--download', action='store_true', help='Download real audio samples')
    parser.add_argument('--force-download', action='store_true', help='Force download even if samples exist')
    parser.add_argument('--create-synthetic', action='store_true', help='Create synthetic training data')
    parser.add_argument('--force-extract', action='store_true', help='Force re-extraction of features')
    parser.add_argument('--test-file', help='Test the model on a specific audio file')
    parser.add_argument('--test-all', action='store_true', help='Test the model on a sample from each genre')
    parser.add_argument('--comprehensive', action='store_true', help='Build a comprehensive model (download, synthetic, train, test)')
    
    args = parser.parse_args()
    
    # Check if we're building a comprehensive model
    if args.comprehensive:
        logger.info("Building comprehensive genre classification model...")
        
        # Check if we have samples for all major genres
        samples_ok, missing_genres = check_samples()
        
        # If we don't have samples, or we're forcing download
        if not samples_ok or args.force_download or args.download:
            download_real_samples(force=args.force_download)
        
        # Create synthetic data to fill in gaps
        if args.create_synthetic or not samples_ok:
            generate_synthetic_data(force=args.force_download)
        
        # Train the model
        train_model(force_extract=args.force_extract)
        
        # Test the model
        if args.test_all:
            test_model()
        elif args.test_file:
            test_model(args.test_file)
        
        logger.info("Comprehensive model build complete!")
        return
    
    # Run individual steps based on arguments
    if args.download or args.force_download:
        download_real_samples(force=args.force_download)
    
    if args.create_synthetic:
        generate_synthetic_data()
    
    # Always train the model unless we're only downloading or creating synthetic data
    if not (args.download and not args.create_synthetic and not args.test_file and not args.test_all):
        train_model(force_extract=args.force_extract)
    
    if args.test_file:
        test_model(args.test_file)
    elif args.test_all:
        test_model()
    
    logger.info("Model build process complete!")

if __name__ == "__main__":
    main() 