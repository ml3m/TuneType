import os
import json
import numpy as np
from tqdm import tqdm
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
import seaborn as sns
from app.utils.feature_extractor import extract_features
from app.utils.model import GenreClassifier
import librosa
import soundfile as sf
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_or_extract_features(sample_dir, force_extract=False, use_augmentation=True):
    """Load features from cache or extract them from audio files with optional augmentation"""
    features_cache_file = os.path.join(sample_dir, 'features_cache.json')
    
    if os.path.exists(features_cache_file) and not force_extract:
        logger.info(f"Loading features from cache: {features_cache_file}")
        with open(features_cache_file, 'r') as f:
            cache = json.load(f)
            X = np.array(cache['features'])
            y = np.array(cache['labels'])
            genres = cache['genres']
        return X, y, genres
    
    logger.info("Extracting features from audio files...")
    
    # Get all genre directories
    genre_dirs = [d for d in os.listdir(sample_dir) 
                 if os.path.isdir(os.path.join(sample_dir, d))
                 and not d.startswith('.')]
    
    if not genre_dirs:
        logger.error(f"No genre directories found in {sample_dir}")
        return None, None, None
    
    logger.info(f"Found genres: {genre_dirs}")
    
    # Extract features from each audio file
    X = []
    y = []
    file_paths = []  # Store file paths for debugging/analysis
    
    for genre_idx, genre in enumerate(genre_dirs):
        genre_dir = os.path.join(sample_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg'))
                      and not f.startswith('.')]
        
        if not audio_files:
            logger.warning(f"No audio files found in {genre_dir}")
            continue
        
        logger.info(f"Processing {len(audio_files)} files for genre '{genre}'")
        
        for audio_file in tqdm(audio_files, desc=f"Extracting features for {genre}"):
            file_path = os.path.join(genre_dir, audio_file)
            
            try:
                # Extract features from multiple segments of the audio for better coverage
                num_segments = 5  # Increased from 3
                segment_duration = 30  # seconds
                
                # Check file size to determine approach
                if os.path.getsize(file_path) < 1000000:  # less than 1MB
                    # For small files, just extract from beginning
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(genre_idx)
                        file_paths.append(file_path)
                else:
                    # For larger files, extract from multiple segments
                    for i in range(num_segments):
                        # Extract from different parts of the audio
                        offset = i * segment_duration
                        features = extract_features(file_path, offset=offset)
                        if features is not None:
                            X.append(features)
                            y.append(genre_idx)
                            file_paths.append(f"{file_path}[{offset}s]")
                    
                    # Apply data augmentation if enabled
                    if use_augmentation:
                        # Load audio for augmentation
                        y_audio, sr = librosa.load(file_path, sr=22050, duration=60)
                        
                        # Only augment if we have enough audio
                        if len(y_audio) > sr * 30:
                            # Time stretching (slower)
                            y_slow = librosa.effects.time_stretch(y_audio, rate=0.85)
                            sf.write('tmp_slow.wav', y_slow, sr)
                            features_slow = extract_features('tmp_slow.wav')
                            if features_slow is not None:
                                X.append(features_slow)
                                y.append(genre_idx)
                                file_paths.append(f"{file_path}[time_stretch_slow]")
                            
                            # Time stretching (faster)
                            y_fast = librosa.effects.time_stretch(y_audio, rate=1.15)
                            sf.write('tmp_fast.wav', y_fast, sr)
                            features_fast = extract_features('tmp_fast.wav')
                            if features_fast is not None:
                                X.append(features_fast)
                                y.append(genre_idx)
                                file_paths.append(f"{file_path}[time_stretch_fast]")
                            
                            # Pitch shifting (up)
                            y_pitch_up = librosa.effects.pitch_shift(y_audio, sr=sr, n_steps=2)
                            sf.write('tmp_pitch_up.wav', y_pitch_up, sr)
                            features_pitch_up = extract_features('tmp_pitch_up.wav')
                            if features_pitch_up is not None:
                                X.append(features_pitch_up)
                                y.append(genre_idx)
                                file_paths.append(f"{file_path}[pitch_up]")
                            
                            # Pitch shifting (down)
                            y_pitch_down = librosa.effects.pitch_shift(y_audio, sr=sr, n_steps=-2)
                            sf.write('tmp_pitch_down.wav', y_pitch_down, sr)
                            features_pitch_down = extract_features('tmp_pitch_down.wav')
                            if features_pitch_down is not None:
                                X.append(features_pitch_down)
                                y.append(genre_idx)
                                file_paths.append(f"{file_path}[pitch_down]")
                            
                            # Add white noise
                            noise_factor = 0.005
                            noise = np.random.randn(len(y_audio))
                            y_noise = y_audio + noise_factor * noise
                            sf.write('tmp_noise.wav', y_noise, sr)
                            features_noise = extract_features('tmp_noise.wav')
                            if features_noise is not None:
                                X.append(features_noise)
                                y.append(genre_idx)
                                file_paths.append(f"{file_path}[noise]")
                            
                            # Clean up temp files
                            for temp_file in ['tmp_slow.wav', 'tmp_fast.wav', 'tmp_pitch_up.wav', 'tmp_pitch_down.wav', 'tmp_noise.wav']:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error extracting features from {file_path}: {e}")
    
    if not X:
        logger.error("No features were successfully extracted!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Cache the features
    cache = {
        'features': X.tolist(),
        'labels': y.tolist(),
        'genres': genre_dirs
    }
    
    with open(features_cache_file, 'w') as f:
        json.dump(cache, f)
    
    # Save file paths for debugging (not in the cache because it would be too large)
    with open(os.path.join(sample_dir, 'feature_files.txt'), 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")
    
    logger.info(f"Extracted features from {len(X)} audio segments across {len(genre_dirs)} genres")
    logger.info(f"Features shape: {X.shape}")
    
    return X, y, genre_dirs

def plot_confusion_matrix(cm, classes, output_file='confusion_matrix.png'):
    """Plot confusion matrix with improved visualization"""
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # More detailed visualization
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap="Blues", 
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                xticklabels=classes, yticklabels=classes)
    
    # Improve readability
    plt.ylabel('True Genre', fontsize=14)
    plt.xlabel('Predicted Genre', fontsize=14)
    plt.title('Normalized Confusion Matrix', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save high-quality figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {output_file}")
    
    # Also create a non-normalized version
    plt.figure(figsize=(14, 12))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", 
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Genre', fontsize=14)
    plt.xlabel('Predicted Genre', fontsize=14)
    plt.title('Confusion Matrix (Count)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_counts.png'), dpi=300, bbox_inches='tight')
    
    return ax

def plot_feature_importance(model, feature_names=None, output_file='feature_importance.png'):
    """Plot feature importance from the model if available"""
    try:
        if hasattr(model, 'estimators_'):
            # For ensemble models, try to get feature importance from a component
            for name, estimator in model.estimators:
                if hasattr(estimator, 'feature_importances_'):
                    importances = estimator.feature_importances_
                    break
        elif hasattr(model, 'feature_importances_'):
            # For single models
            importances = model.feature_importances_
        else:
            logger.warning("Model doesn't have feature importance attribute")
            return
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Get top 20 features
        top_indices = indices[:20]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title("Top 20 Feature Importances", fontsize=16)
        plt.bar(range(20), importances[top_indices], align="center")
        plt.xticks(range(20), [feature_names[i] for i in top_indices], rotation=90)
        plt.tight_layout()
        plt.ylabel("Importance", fontsize=14)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")

def train_model_with_samples(sample_dir, force_extract=False, test_size=0.2, use_cross_val=True, use_augmentation=True):
    """Train the genre classifier with audio samples"""
    # Extract features from audio samples
    X, y, genres = load_or_extract_features(sample_dir, force_extract, use_augmentation)
    
    if X is None or len(X) == 0:
        logger.error("No features available for training")
        return False, None
    
    # Count samples per genre and print
    unique, counts = np.unique(y, return_counts=True)
    genre_counts = dict(zip([genres[u] for u in unique], counts))
    logger.info("Samples per genre:")
    for genre, count in genre_counts.items():
        logger.info(f"  {genre}: {count} samples")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
    
    # Initialize the model with the correct genres
    model = GenreClassifier(genres=genres)
    
    # Perform cross-validation if requested
    if use_cross_val and len(X) >= 50:
        logger.info("Performing cross-validation...")
        try:
            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Get model instance for CV
            cv_model = model._build_model()
            
            # Compute cross-validation scores using balanced accuracy
            cv_scores = cross_val_score(
                cv_model, X, y, cv=cv, scoring='balanced_accuracy'
            )
            
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Plot learning curve if dataset is large enough
            if len(X) >= 100:
                logger.info("Generating learning curve...")
                from sklearn.model_selection import learning_curve
                
                # Calculate learning curve
                train_sizes, train_scores, test_scores = learning_curve(
                    cv_model, X, y, cv=cv, scoring='balanced_accuracy',
                    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
                )
                
                # Calculate mean and std of training and test scores
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                # Plot learning curve
                plt.figure(figsize=(10, 6))
                plt.title("Learning Curve", fontsize=16)
                plt.xlabel("Training examples", fontsize=14)
                plt.ylabel("Balanced accuracy score", fontsize=14)
                plt.grid()
                
                plt.fill_between(train_sizes, train_mean - train_std,
                                train_mean + train_std, alpha=0.1, color="r")
                plt.fill_between(train_sizes, test_mean - test_std,
                                test_mean + test_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_mean, 'o-', color="r",
                        label="Training score")
                plt.plot(train_sizes, test_mean, 'o-', color="g",
                        label="Cross-validation score")
                plt.legend(loc="best")
                plt.savefig("learning_curve.png", dpi=300, bbox_inches='tight')
                logger.info("Learning curve saved to learning_curve.png")
                
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
    
    # Train the model on training data
    history = model.train(X_train, y_train, cross_validation=use_cross_val)
    
    # Evaluate on test set
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate various metrics
    accuracy = np.mean(y_pred == y_test)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Test metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"  F1 Score (weighted): {f1:.4f}")
    
    # Generate detailed classification report
    report = classification_report(y_test, y_pred, target_names=genres, output_dict=True)
    logger.info("Classification Report:")
    for genre in genres:
        if genre in report:
            precision = report[genre]['precision']
            recall = report[genre]['recall']
            f1 = report[genre]['f1-score']
            support = report[genre]['support']
            logger.info(f"  {genre}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
    
    # Save classification report as CSV
    import pandas as pd
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv')
    logger.info("Classification report saved to classification_report.csv")
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, genres)
    
    # Try to plot feature importance
    plot_feature_importance(model.model)
    
    # Save the trained model
    model.save_model()
    
    # Return success and model
    return True, model

def main():
    parser = argparse.ArgumentParser(description='Train genre classifier with audio samples')
    parser.add_argument('--sample-dir', default='samples', help='Directory containing audio samples organized by genre')
    parser.add_argument('--force-extract', action='store_true', help='Force re-extraction of features even if cache exists')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--skip-cross-val', action='store_true', help='Skip cross-validation')
    parser.add_argument('--skip-augmentation', action='store_true', help='Skip data augmentation')
    
    args = parser.parse_args()
    
    logger.info(f"Training with samples from {args.sample_dir}")
    success, _ = train_model_with_samples(
        args.sample_dir, 
        force_extract=args.force_extract,
        test_size=args.test_size,
        use_cross_val=not args.skip_cross_val,
        use_augmentation=not args.skip_augmentation
    )
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main() 