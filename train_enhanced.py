#!/usr/bin/env python3
import os
import argparse
import numpy as np
import logging
from enhanced_genre_model import EnhancedGenreClassifier
from train_enhanced_model import load_and_augment_data, plot_learning_curve
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Music Genre Classifier')
    parser.add_argument('--sample-dir', type=str, default='samples',
                       help='Directory containing genre samples')
    parser.add_argument('--augment-factor', type=int, default=5,
                       help='Base augmentation factor for samples')
    parser.add_argument('--no-advanced-aug', action='store_true',
                       help='Disable advanced augmentation techniques')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--test-file', type=str,
                       help='Test file to classify after training')
    
    args = parser.parse_args()
    
    # Create model
    classifier = EnhancedGenreClassifier()
    
    # Train on existing samples
    logger.info(f"Preparing to train with samples from {args.sample_dir}")
    
    try:
        # Load and augment data
        X, y, genres = load_and_augment_data(
            args.sample_dir, 
            args.augment_factor,
            not args.no_advanced_aug
        )
        
        if X is None or len(X) == 0:
            logger.error("No features extracted, aborting")
            return
            
        # Train the model
        logger.info(f"Training enhanced model with {len(X)} samples for {args.epochs} epochs")
        history = classifier.train(
            X, y, 
            sample_dir=args.sample_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        # Plot learning curve
        plot_learning_curve(history)
        
        # Split data for tuning genre weights
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Tune genre weights
        logger.info("Tuning genre weights to improve accuracy")
        classifier.tune_genre_weights(X_val, y_val)
        
        logger.info("Model training complete!")
        
        # Test if a file was provided
        if args.test_file:
            from app.utils.feature_extractor import extract_multi_segment_features
            
            logger.info(f"Testing model on {args.test_file}")
            features = extract_multi_segment_features(args.test_file)
            
            if not features:
                logger.error(f"Failed to extract features from {args.test_file}")
                return
                
            # Make prediction
            result = classifier.predict(features)
            
            # Sort and display results
            sorted_genres = sorted(result.items(), key=lambda x: x[1], reverse=True)
            print(f"\nPredictions for {os.path.basename(args.test_file)}:")
            for genre, prob in sorted_genres:
                print(f"{genre}: {prob*100:.2f}%")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 