#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
import time
import matplotlib.pyplot as plt
from app.utils.model import GenreClassifier
from app.utils.feature_extractor import extract_features, extract_multi_segment_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_with_multi_segment_analysis(file_path, num_segments=5, use_deep_learning=True):
    """
    Predict genre using multi-segment analysis for more robust predictions
    
    Args:
        file_path (str): Path to the audio file
        num_segments (int): Number of segments to analyze
        use_deep_learning (bool): Whether to use deep learning model
        
    Returns:
        tuple: (predicted_genre, confidence, all_predictions)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None, 0, None
    
    # Extract features from multiple segments
    logger.info(f"Extracting features from {num_segments} segments of {file_path}")
    segment_features = extract_multi_segment_features(file_path, num_segments=num_segments, segment_duration=10)
    
    if not segment_features:
        logger.error("Failed to extract features from any segments")
        return None, 0, None
    
    logger.info(f"Successfully extracted features from {len(segment_features)} segments")
    
    # Initialize model with deep learning if requested
    model = GenreClassifier(use_deep_learning=use_deep_learning)
    genres = model.get_genres()
    
    # Predict using multi-segment analysis
    start_time = time.time()
    probabilities = model.predict(segment_features)
    prediction_time = time.time() - start_time
    
    # Get most confident prediction
    predicted_idx = np.argmax(probabilities[0])
    predicted_genre = genres[predicted_idx]
    confidence = probabilities[0][predicted_idx]
    
    # Get all predictions with probabilities above 10%
    significant_preds = {}
    for i, prob in enumerate(probabilities[0]):
        if prob >= 0.1:  # 10% threshold
            significant_preds[genres[i]] = float(prob)
    
    # Sort by probability
    sorted_preds = dict(sorted(significant_preds.items(), key=lambda x: x[1], reverse=True))
    
    logger.info(f"Prediction completed in {prediction_time:.3f} seconds")
    logger.info(f"Predicted genre: {predicted_genre} (confidence: {confidence:.2f})")
    logger.info(f"Top predictions: {sorted_preds}")
    
    # Return predicted genre, confidence, and all predictions
    return predicted_genre, confidence, probabilities[0]

def visualize_predictions(probabilities, genres, output_file=None):
    """
    Visualize prediction probabilities as a bar chart
    
    Args:
        probabilities (numpy.ndarray): Prediction probabilities
        genres (list): List of genre names
        output_file (str): Path to save the visualization, if None, display only
    """
    # Sort probabilities and genres by probability value
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_genres = [genres[i] for i in sorted_indices]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(genres)))
    
    # Plot only genres with at least 1% probability
    mask = sorted_probs >= 0.01
    plt.barh(
        [sorted_genres[i] for i in range(len(sorted_genres)) if mask[i]],
        [sorted_probs[i] for i in range(len(sorted_probs)) if mask[i]],
        color=[colors[i] for i in range(len(colors)) if mask[i]]
    )
    
    plt.xlabel('Probability')
    plt.ylabel('Genre')
    plt.title('Genre Prediction Probabilities')
    plt.xlim(0, min(1.0, max(sorted_probs) * 1.1))  # Add 10% margin
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Annotate with probabilities
    for i, (genre, prob) in enumerate(zip(sorted_genres, sorted_probs)):
        if prob >= 0.01:  # Only annotate if at least 1%
            plt.text(
                prob + 0.01, i, 
                f"{prob:.0%}",
                va='center'
            )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Visualization saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Multi-segment Music Genre Classification')
    parser.add_argument('file', type=str, help='Path to audio file to analyze')
    parser.add_argument('--segments', type=int, default=5, 
                       help='Number of segments to analyze')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of predictions')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for visualization (if --visualize is specified)')
    parser.add_argument('--deep-learning', action='store_true',
                       help='Use deep learning model (if available)')
    
    args = parser.parse_args()
    
    # Analyze file
    predicted_genre, confidence, all_probs = predict_with_multi_segment_analysis(
        args.file, 
        num_segments=args.segments,
        use_deep_learning=args.deep_learning
    )
    
    if predicted_genre is None:
        logger.error("Prediction failed")
        sys.exit(1)
    
    # Visualize if requested
    if args.visualize:
        model = GenreClassifier()
        genres = model.get_genres()
        visualize_predictions(all_probs, genres, args.output)
    
    # Print final result
    print(f"\nPredicted Genre: {predicted_genre}")
    print(f"Confidence: {confidence:.2%}")
    
    # Print top 3 alternatives if confidence is below 60%
    if confidence < 0.6:
        # Find top 3 excluding the predicted genre
        model = GenreClassifier()
        genres = model.get_genres()
        
        # Create sorted list of (genre, probability) pairs excluding top result
        pred_idx = genres.index(predicted_genre)
        alternatives = [(genres[i], all_probs[i]) for i in range(len(genres)) if i != pred_idx]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        print("\nAlternative possibilities:")
        for i, (genre, prob) in enumerate(alternatives[:3]):
            print(f"  {genre}: {prob:.2%}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 