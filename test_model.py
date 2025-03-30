#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from app.utils.feature_extractor import extract_features
from app.utils.model import GenreClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model(audio_file, visualize=True, output_dir="output"):
    """
    Test the trained model on an audio file
    
    Args:
        audio_file: Path to the audio file
        visualize: Whether to create visualizations
        output_dir: Directory to save visualizations
    
    Returns:
        Dictionary with prediction results
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return None
    
    try:
        # Create output directory if it doesn't exist
        if visualize and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Load the trained model
        model = GenreClassifier()
        model.load_or_build_model()
        
        # Extract features from the audio file
        logger.info(f"Extracting features from {audio_file}...")
        features = extract_features(audio_file)
        
        if features is None:
            logger.error("Feature extraction failed")
            return None
        
        # Make prediction
        logger.info("Making genre prediction...")
        probabilities = model.predict_proba(features.reshape(1, -1))
        
        # Get the top genres and their probabilities
        top_indices = np.argsort(-probabilities[0])
        top_genres = [model.genres[i] for i in top_indices]
        top_probs = probabilities[0][top_indices]
        
        # Display results
        logger.info(f"\nPrediction for {os.path.basename(audio_file)}:")
        logger.info("-" * 50)
        logger.info(f"Top genre: {top_genres[0]} ({top_probs[0]*100:.2f}%)")
        logger.info("\nAll genre probabilities:")
        
        for genre, prob in zip(top_genres, top_probs):
            logger.info(f"{genre}: {prob*100:.2f}%")
        
        # Create visualizations if requested
        if visualize:
            # 1. Load audio for waveform and spectrogram
            y, sr = librosa.load(audio_file, duration=60)  # Load up to 60 seconds
            
            # 2. Create figure with multiple subplots
            plt.figure(figsize=(15, 10))
            
            # 3. Plot waveform
            plt.subplot(3, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f"Waveform - {os.path.basename(audio_file)}")
            
            # 4. Plot spectrogram
            plt.subplot(3, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Log-frequency Power Spectrogram")
            
            # 5. Plot genre probabilities
            plt.subplot(3, 1, 3)
            # Only show top 6 genres for clarity
            top_n = min(6, len(top_genres))
            plt.barh(range(top_n), top_probs[:top_n])
            plt.yticks(range(top_n), [f"{g} ({p*100:.1f}%)" for g, p in zip(top_genres[:top_n], top_probs[:top_n])])
            plt.xlabel("Probability")
            plt.title("Genre Prediction Probabilities")
            plt.tight_layout()
            
            # Save visualization
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_analysis.png")
            plt.savefig(output_file)
            logger.info(f"Visualization saved to {output_file}")
            
            # Optionally show the plot (disable for non-interactive environments)
            # plt.show()
        
        # Return results
        return {
            'file': audio_file,
            'top_genre': top_genres[0],
            'top_probability': top_probs[0],
            'genres': top_genres,
            'probabilities': top_probs.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description='Test the genre classification model on an audio file')
    parser.add_argument('audio_file', help='Path to the audio file to test')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualizations')
    parser.add_argument('--output-dir', default='output', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Test the model
    result = test_model(args.audio_file, visualize=not args.no_vis, output_dir=args.output_dir)
    
    if result:
        # Exit with success
        sys.exit(0)
    else:
        # Exit with error
        sys.exit(1)

if __name__ == "__main__":
    main() 