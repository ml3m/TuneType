#!/usr/bin/env python3
import os
import json
import numpy as np
import logging
import random
from tqdm import tqdm
from app.utils.model import GenreClassifier
from app.utils.feature_extractor import extract_features
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_sample_features(sample_dir):
    """Extract features from sample files for each genre"""
    # Get all genre directories
    genre_dirs = [d for d in os.listdir(sample_dir) 
                 if os.path.isdir(os.path.join(sample_dir, d))
                 and not d.startswith('.')]
    
    if not genre_dirs:
        logger.error(f"No genre directories found in {sample_dir}")
        return None, None
    
    logger.info(f"Found genres: {genre_dirs}")
    
    # Extract features from each audio file
    X = []
    y = []
    file_paths = []
    
    for genre_idx, genre in enumerate(genre_dirs):
        genre_dir = os.path.join(sample_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg'))
                      and not f.startswith('.')]
        
        if not audio_files:
            logger.warning(f"No audio files found in {genre_dir}")
            continue
        
        # Sample a reasonable number of files for calibration
        sample_count = min(5, len(audio_files))
        sampled_files = random.sample(audio_files, sample_count)
        
        logger.info(f"Extracting features from {sample_count} files for genre '{genre}'")
        
        for audio_file in sampled_files:
            file_path = os.path.join(genre_dir, audio_file)
            
            try:
                # Extract features
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(genre_idx)
                    file_paths.append(file_path)
            except Exception as e:
                logger.error(f"Error extracting features from {file_path}: {e}")
    
    if not X:
        logger.error("No features were successfully extracted!")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Extracted features from {len(X)} audio samples across {len(genre_dirs)} genres")
    
    return X, y, genre_dirs

def calculate_calibration_factors(model, X, y, genres):
    """Calculate calibration factors to correct genre bias"""
    # Make predictions
    all_probs = []
    for features in X:
        # Reshape if needed
        features_reshaped = features.reshape(1, -1)
        # Predict
        probs = model.predict(features_reshaped)
        all_probs.append(probs[0])
    
    # Make sure all_probs has consistent dimensions
    model_genres = model.get_genres()
    model_genre_count = len(model_genres)
    
    # Convert to array
    all_probs = np.array(all_probs)
    
    # Get predicted classes
    y_pred = np.argmax(all_probs, axis=1)
    
    # Calculate confusion matrix
    plot_confusion_matrix(confusion_matrix(y, y_pred), genres, title='Before Calibration: Confusion Matrix', output_file='confusion_matrix_before_calibration.png')
    
    # Get unique classes in predictions and test set
    unique_classes = np.unique(np.concatenate([y, y_pred]))
    actual_genres = [genres[i] if i < len(genres) else f"Unknown-{i}" for i in unique_classes]
        
    # Generate classification report with actual labels
    try:
        report = classification_report(y, y_pred, labels=unique_classes, 
                                      target_names=actual_genres, output_dict=True)
        logger.info("Classification report before calibration:")
        for genre_name in actual_genres:
            if genre_name in report:
                logger.info(f"  {genre_name}: Precision={report[genre_name]['precision']:.2f}, "
                          f"Recall={report[genre_name]['recall']:.2f}, "
                          f"F1={report[genre_name]['f1-score']:.2f}")
    except Exception as e:
        logger.error(f"Error generating classification report: {str(e)}")
        report = {}
    
    # Calculate calibration factors using a more sophisticated method
    calibration_factors = {}
    
    # Collect per-genre statistics
    genre_stats = {}
    for i, genre in enumerate(model_genres):
        genre_stats[genre] = {
            'predictions': [],
            'true_positives': 0,
            'false_positives': 0,
            'confusion_with': {},
            'confidence': []
        }
    
    # Analyze predictions to understand model behavior
    for idx, (true_class, pred_probs) in enumerate(zip(y, all_probs)):
        pred_class = np.argmax(pred_probs)
        true_genre = genres[true_class] if true_class < len(genres) else f"Unknown-{true_class}"
        pred_genre = model_genres[pred_class] if pred_class < len(model_genres) else f"Unknown-{pred_class}"
        
        # Record confidence
        confidence = pred_probs[pred_class]
        
        # Record prediction data
        for i, genre in enumerate(model_genres):
            if genre in genres:
                genre_stats[genre]['predictions'].append(pred_probs[i])
                
                # If this is the true genre
                if genres.index(genre) == true_class:
                    genre_stats[genre]['confidence'].append(pred_probs[i])
                    
                    # Record accuracy
                    if pred_class == i:
                        genre_stats[genre]['true_positives'] += 1
                    else:
                        # Record what it was confused with
                        confused_with = model_genres[pred_class] if pred_class < len(model_genres) else "unknown"
                        if confused_with in genre_stats[genre]['confusion_with']:
                            genre_stats[genre]['confusion_with'][confused_with] += 1
                        else:
                            genre_stats[genre]['confusion_with'][confused_with] = 1
                
                # If this is the predicted genre but not the true genre
                elif pred_class == i and genres.index(genre) != true_class:
                    genre_stats[genre]['false_positives'] += 1
    
    # Calculate calibration factors based on detailed statistics
    for genre in model_genres:
        if genre in genres and genre in genre_stats:
            stats = genre_stats[genre]
            
            # Average prediction score for this genre when it's the true genre
            avg_true_confidence = np.mean(stats['confidence']) if stats['confidence'] else 0.5
            
            # Average prediction across all samples
            avg_prediction = np.mean(stats['predictions']) if stats['predictions'] else 0.5
            
            # Calculate precision, avoid division by zero
            total_predictions = stats['true_positives'] + stats['false_positives']
            precision = stats['true_positives'] / total_predictions if total_predictions > 0 else 0
            
            # Calculate factor based on both confidence and precision ratio
            # If precision is low, we need to reduce the model's predictions more
            if avg_prediction > 0.01:
                confidence_factor = avg_true_confidence / avg_prediction
                precision_factor = 0.5 if precision == 0 else min(2.0, (precision + 0.5) / 1.5)
                
                # Combine the factors with more weight on precision for low-precision genres
                if precision < 0.3:
                    factor = confidence_factor * 0.3 + precision_factor * 0.7
                else:
                    factor = confidence_factor * 0.7 + precision_factor * 0.3
                
                # Limit to reasonable range
                factor = max(0.2, min(5.0, factor))
                
                # Special adjustments for commonly confused genres
                if genre == 'jazz' and 'false_positives' in stats and stats['false_positives'] > 0:
                    # Jazz tends to be overpredicted
                    factor = min(0.7, factor)
                elif genre == 'classical' and 'confusion_with' in stats and 'jazz' in stats['confusion_with']:
                    # If classical is confused with jazz, boost it slightly
                    factor = min(3.0, factor * 1.2)
                
                calibration_factors[genre] = factor
            else:
                calibration_factors[genre] = 1.0
        else:
            # Genre not in samples or stats, use default
            calibration_factors[genre] = 1.0
    
    # Clean up factors and ensure all model genres have factors
    for genre in model_genres:
        if genre not in calibration_factors:
            calibration_factors[genre] = 1.0
    
    logger.info("Calculated calibration factors:")
    for genre, factor in calibration_factors.items():
        logger.info(f"  {genre}: {factor:.2f}")
    
    return calibration_factors

def apply_calibration(model, X, y, genres, calibration_factors):
    """Apply calibration factors and check if they improve predictions"""
    # Make predictions
    all_probs = []
    for features in X:
        # Reshape if needed
        features_reshaped = features.reshape(1, -1)
        # Predict
        probs = model.predict(features_reshaped)
        all_probs.append(probs[0])
    
    # Get model genres
    model_genres = model.get_genres()
    
    # Convert to array
    all_probs = np.array(all_probs)
    
    # Apply calibration factors
    calibrated_probs = np.zeros_like(all_probs)
    for i, genre in enumerate(model_genres):
        factor = calibration_factors.get(genre, 1.0)
        calibrated_probs[:, i] = all_probs[:, i] * factor
    
    # Normalize to ensure sum is 1.0 for each prediction
    row_sums = calibrated_probs.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    calibrated_probs = calibrated_probs / row_sums
    
    # Get predicted classes
    y_pred = np.argmax(calibrated_probs, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, genres, title='After Calibration: Confusion Matrix', output_file='confusion_matrix_after_calibration.png')
    
    # Get unique classes in predictions and test set
    unique_classes = np.unique(np.concatenate([y, y_pred]))
    actual_genres = [genres[i] if i < len(genres) else f"Unknown-{i}" for i in unique_classes]
        
    # Generate classification report with actual labels
    try:
        report = classification_report(y, y_pred, labels=unique_classes, 
                                    target_names=actual_genres, output_dict=True)
        logger.info("Classification report after calibration:")
        for genre_name in actual_genres:
            if genre_name in report:
                logger.info(f"  {genre_name}: Precision={report[genre_name]['precision']:.2f}, "
                        f"Recall={report[genre_name]['recall']:.2f}, "
                        f"F1={report[genre_name]['f1-score']:.2f}")
    except Exception as e:
        logger.error(f"Error generating classification report: {str(e)}")
        report = {}
    
    return report

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', output_file='confusion_matrix.png'):
    """Plot confusion matrix with improved visualization"""
    # Ensure we have matching dimensions
    n_classes = cm.shape[0]
    
    # If we have more classes than labels, extend the labels
    if len(classes) < n_classes:
        classes = list(classes) + [f"Unknown-{i}" for i in range(len(classes), n_classes)]
    # If we have more labels than classes, truncate the labels
    elif len(classes) > n_classes:
        classes = classes[:n_classes]
    
    # Handle empty rows in confusion matrix (divide by zero)
    row_sums = cm.sum(axis=1)
    # Replace zeros with ones to avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
    
    # Create the normalized plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} (Normalized)')
    plt.tight_layout()
    plt.savefig(output_file)
    
    # Also create a counts version
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} (Counts)')
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_counts.png'))

def main():
    parser = argparse.ArgumentParser(description='Tune genre bias in the classifier')
    parser.add_argument('--sample-dir', type=str, default='samples', 
                        help='Directory containing genre samples')
    parser.add_argument('--output-file', type=str, default='app/models/genre_calibration.json',
                        help='Output file for calibration factors')
    parser.add_argument('--force', action='store_true',
                        help='Force recalculation even if calibration file exists')
    
    args = parser.parse_args()
    
    # Check if output file already exists
    if os.path.exists(args.output_file) and not args.force:
        logger.info(f"Calibration file {args.output_file} already exists. Use --force to recalculate.")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize the model
    logger.info("Loading genre classifier model")
    model = GenreClassifier()
    
    # Extract features from sample files
    logger.info(f"Extracting features from samples in {args.sample_dir}")
    X, y, genres = extract_sample_features(args.sample_dir)
    
    if X is None or len(X) == 0:
        logger.error("No features available for calibration")
        return
    
    # Calculate calibration factors
    logger.info("Calculating calibration factors")
    calibration_factors = calculate_calibration_factors(model, X, y, genres)
    
    # Apply calibration and check results
    logger.info("Applying calibration factors")
    apply_calibration(model, X, y, genres, calibration_factors)
    
    # Save calibration factors
    logger.info(f"Saving calibration factors to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(calibration_factors, f, indent=2)
    
    logger.info("Calibration complete!")

if __name__ == "__main__":
    main() 