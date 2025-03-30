#!/usr/bin/env python3
import os
import json
import numpy as np
import random
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from app.utils.model import GenreClassifier
from app.utils.feature_extractor import extract_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_and_cache_features(sample_dir, cache_file='features_cache.json', force_recompute=False):
    """Extract features from audio samples and cache them for faster reuse"""
    cache_path = os.path.join(sample_dir, cache_file)
    
    # Check if cache exists and we're not forced to recompute
    if os.path.exists(cache_path) and not force_recompute:
        logger.info(f"Loading features from cache: {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            
            # Prepare arrays for X and y
            X = np.array(cache['features'])
            y = np.array(cache['labels'])
            genres = cache['genres']
            
            return X, y, genres
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            logger.info("Extracting features from scratch")
    else:
        logger.info(f"{'Force recomputing' if force_recompute else 'No cache found'}, extracting features from audio files")
    
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
    
    # Extract features from each audio file
    features = []
    labels = []
    file_paths = []
    
    for genre in genres:
        genre_dir = os.path.join(sample_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))
                      and not f.startswith('.')]
        
        if not audio_files:
            logger.warning(f"No audio files found in {genre_dir}, skipping genre")
            continue
        
        logger.info(f"Processing {len(audio_files)} audio files for genre '{genre}'")
        
        for audio_file in tqdm(audio_files, desc=f"Extracting {genre}"):
            file_path = os.path.join(genre_dir, audio_file)
            
            try:
                # Extract features with data augmentation options
                # Standard extraction
                feature_vector = extract_features(file_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(genre_to_idx[genre])
                    file_paths.append(file_path)
                
                # Data augmentation - process small time shifts of the audio
                # This creates 2 additional training samples per file
                for offset in [0.5, 1.0]:
                    feature_vector = extract_features(file_path, offset=offset)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(genre_to_idx[genre])
                        file_paths.append(file_path + f"_offset{offset}")
                
            except Exception as e:
                logger.error(f"Error extracting features from {file_path}: {str(e)}")
    
    if not features:
        logger.error("No features were successfully extracted!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Cache the results
    try:
        cache = {
            'features': X.tolist(),
            'labels': y.tolist(),
            'genres': genres,
            'files': file_paths
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        
        logger.info(f"Cached {len(X)} feature vectors to {cache_path}")
    except Exception as e:
        logger.error(f"Error caching features: {str(e)}")
    
    return X, y, genres

def plot_confusion_matrix(y_true, y_pred, genres, output_file='confusion_matrix.png'):
    """Plot and save confusion matrix visualization"""
    try:
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
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")

def save_classification_report(y_true, y_pred, genres, output_file='classification_report.csv'):
    """Save classification report to CSV for easy analysis"""
    try:
        # Ensure we have consistent labels between predictions and actual values
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        actual_genres = [genres[i] if i < len(genres) else f"Unknown-{i}" for i in unique_classes]
        
        report = classification_report(y_true, y_pred, labels=unique_classes, 
                                      target_names=actual_genres, output_dict=True)
        
        # Convert to DataFrame for easy CSV output
        df = pd.DataFrame(report).transpose()
        df.to_csv(output_file)
        logger.info(f"Classification report saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving classification report: {str(e)}")

def plot_learning_curve(model, X, y, genres, output_file='learning_curve.png'):
    """Plot learning curve to visualize model performance vs. training size"""
    try:
        # Create scoring function that balances accuracy and F1 score
        def combined_score(estimator, X, y):
            y_pred = estimator.predict(X)
            acc = balanced_accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='macro')
            return (acc + f1) / 2
        
        # Define training sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model.model, X, y, train_sizes=train_sizes, cv=5, 
            scoring=combined_score, n_jobs=-1, verbose=1
        )
        
        # Calculate mean and standard deviation for training and test scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.title('Learning Curve (Combined Accuracy & F1 Score)')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.grid()
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                        alpha=0.1, color='g')
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        
        plt.legend(loc='best')
        plt.savefig(output_file)
        
        logger.info(f"Learning curve saved to {output_file}")
    except Exception as e:
        logger.error(f"Error plotting learning curve: {str(e)}")

def evaluate_genre_specific_accuracy(model, X, y, genres):
    """Calculate and report genre-specific accuracy metrics"""
    try:
        # Get model predictions
        probabilities = model.predict(X)
        y_pred = np.argmax(probabilities, axis=1)
        
        # For each genre, calculate accuracy when that's the true genre
        genre_accuracies = {}
        
        for i, genre in enumerate(genres):
            # Find samples where the true genre is this genre
            mask = (y == i)
            
            if np.sum(mask) > 0:
                # Calculate accuracy for this genre
                genre_acc = np.mean(y_pred[mask] == y[mask])
                genre_accuracies[genre] = genre_acc
            else:
                genre_accuracies[genre] = 0.0
        
        logger.info("Genre-specific accuracies:")
        for genre, acc in genre_accuracies.items():
            logger.info(f"{genre}: {acc:.4f}")
        
        return genre_accuracies
    except Exception as e:
        logger.error(f"Error calculating genre-specific accuracy: {str(e)}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Train genre classifier with audio samples')
    parser.add_argument('--sample-dir', type=str, default='samples', 
                       help='Directory containing genre samples')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Percentage of data to use for testing (default: 0.2)')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation of features instead of using cache')
    parser.add_argument('--cross-validation', action='store_true',
                       help='Use cross-validation during training')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Extract features from samples
    X, y, genres = extract_and_cache_features(
        args.sample_dir, 
        force_recompute=args.force_recompute
    )
    
    if X is None or y is None:
        logger.error("Failed to extract features or no features found!")
        return
    
    # Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    
    # Initialize model with the genres from our dataset
    model = GenreClassifier(genres=genres)
    
    # Train the model
    logger.info("Training the model...")
    
    # Train with cross-validation if requested
    validation_score = model.train(
        X_train, y_train, 
        cross_validation=args.cross_validation
    )
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    
    # Get predictions on test set
    test_probabilities = model.predict(X_test)
    test_predictions = np.argmax(test_probabilities, axis=1)
    
    # Calculate balanced accuracy to handle class imbalance
    balanced_acc = balanced_accuracy_score(y_test, test_predictions)
    logger.info(f"Balanced accuracy: {balanced_acc:.4f}")
    
    # Calculate macro F1 score for balanced evaluation
    f1 = f1_score(y_test, test_predictions, average='macro')
    logger.info(f"Macro F1 score: {f1:.4f}")
    
    # Save detailed classification report
    save_classification_report(y_test, test_predictions, genres)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, test_predictions, genres)
    
    # Plot learning curve
    plot_learning_curve(model, X, y, genres)
    
    # Calculate genre-specific accuracies
    evaluate_genre_specific_accuracy(model, X_train, y_train, genres)
    
    logger.info("Model training and evaluation complete")

if __name__ == "__main__":
    main() 