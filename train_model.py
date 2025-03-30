import os
import numpy as np
import logging
import requests
import zipfile
import librosa
from tqdm import tqdm
from app.utils.model import GenreClassifier
from app.utils.feature_extractor import extract_features

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL for GTZAN dataset (small sample)
DATASET_URL = "https://storage.googleapis.com/open-source-datasets/GTZAN/GTZAN_sample.zip"
DATASET_DIR = "datasets/gtzan_sample"

def download_dataset():
    """Download a sample of the GTZAN dataset if it doesn't exist"""
    if os.path.exists(DATASET_DIR):
        logger.info(f"Dataset already exists at {DATASET_DIR}")
        return
    
    logger.info(f"Downloading dataset from {DATASET_URL}")
    os.makedirs("datasets", exist_ok=True)
    
    # Download zip file
    response = requests.get(DATASET_URL, stream=True)
    zip_path = "datasets/gtzan_sample.zip"
    with open(zip_path, "wb") as f:
        total_length = int(response.headers.get('content-length', 0))
        for chunk in tqdm(response.iter_content(chunk_size=1024), 
                          total=total_length//1024, 
                          unit='KB'):
            if chunk:
                f.write(chunk)
    
    # Extract the zip file
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("datasets")
    
    # Remove the zip file
    os.remove(zip_path)
    logger.info("Dataset extracted and ready for use")

def extract_features_from_file(file_path, genre_label, genres):
    """Extract features from an audio file and return with label"""
    try:
        features = extract_features(file_path)
        label = genres.index(genre_label)
        return features, label
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None, None

def prepare_dataset():
    """Prepare the dataset for training"""
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    features_list = []
    labels_list = []
    
    logger.info("Extracting features from audio files...")
    
    # Process each genre folder
    for genre in genres:
        genre_dir = os.path.join(DATASET_DIR, genre)
        if not os.path.exists(genre_dir):
            logger.warning(f"Genre directory not found: {genre_dir}")
            continue
            
        # Process each audio file in the genre directory
        for filename in tqdm(os.listdir(genre_dir), desc=f"Processing {genre}"):
            if filename.endswith('.wav') or filename.endswith('.au') or filename.endswith('.mp3'):
                file_path = os.path.join(genre_dir, filename)
                features, label = extract_features_from_file(file_path, genre, genres)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
    
    # Convert lists to numpy arrays
    X = np.vstack(features_list)
    y = np.array(labels_list)
    
    logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def train_model():
    """Download dataset, extract features and train the model"""
    # Download the dataset
    download_dataset()
    
    # Prepare the dataset
    X, y = prepare_dataset()
    
    # Initialize the model
    classifier = GenreClassifier()
    
    # Train the model
    logger.info("Training the model...")
    classifier.train(X, y)
    
    logger.info("Model training complete!")
    
if __name__ == "__main__":
    logger.info("Starting model training script...")
    train_model() 