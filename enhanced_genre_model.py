#!/usr/bin/env python3
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from app.utils.feature_extractor import extract_features, extract_multi_segment_features
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedGenreClassifier:
    """
    Enhanced genre classifier using deep learning with advanced architecture
    """
    def __init__(self, genres=None, model_dir='app/models', feature_dim=135):
        """
        Initialize the enhanced genre classifier
        
        Args:
            genres (list): List of genres to classify
            model_dir (str): Directory to save/load models
            feature_dim (int): Dimension of input features
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Default genres if none provided
        self.genres = genres or [
            'blues', 'classical', 'country', 'electronic', 'folk', 
            'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'world'
        ]
        
        self.num_genres = len(self.genres)
        self.feature_dim = feature_dim
        self.model_path = os.path.join(model_dir, 'enhanced_genre_model.h5')
        self.scaler_path = os.path.join(model_dir, 'enhanced_scaler.joblib')
        self.genre_weights_path = os.path.join(model_dir, 'genre_weights.joblib')
        
        # Initialize models
        self.model = None
        self.scaler = None
        self.genre_weights = None
        
        # Try to load existing model or create new one
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create a new one"""
        if os.path.exists(self.model_path):
            logger.info("Loading existing enhanced genre model")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                if os.path.exists(self.genre_weights_path):
                    self.genre_weights = joblib.load(self.genre_weights_path)
                
                logger.info("Enhanced model loaded successfully")
                return
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
        
        logger.info("Creating new enhanced genre model")
        self.model = self._build_model()
        
        # Initialize with equal weights
        self.genre_weights = {genre: 1.0 for genre in self.genres}
    
    def _build_model(self):
        """Build the enhanced deep learning model with advanced architecture"""
        # Input layer
        inputs = layers.Input(shape=(self.feature_dim,))
        
        # Initial layer with stronger regularization
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-3))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)  # Increase dropout to prevent overfitting
        
        # First residual block with stronger regularization
        res_input = x
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, res_input])  # Residual connection
        
        # Second layer with skip connection
        res_input = x
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Project residual to match dimensions
        res_input = layers.Dense(256, use_bias=False)(res_input)
        x = layers.Add()([x, res_input])  # Residual connection
        
        # Third layer with skip connection
        res_input = x
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Project residual to match dimensions
        res_input = layers.Dense(128, use_bias=False)(res_input)
        x = layers.Add()([x, res_input])  # Residual connection
        
        # Final layers
        x = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer with softmax activation
        outputs = layers.Dense(self.num_genres, activation="softmax")(x)
        
        # Create and compile model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Use Adam optimizer with weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=1e-4,  # Increase weight decay
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        logger.info(f"Enhanced genre model created with {model.count_params()} parameters")
        return model
    
    def train(self, features, labels, sample_dir=None, epochs=200, batch_size=32):
        """
        Train the enhanced genre model
        
        Args:
            features (np.ndarray): Feature vectors for training
            labels (np.ndarray): Genre labels (integers) for training
            sample_dir (str): Optional directory with genre samples to extract more features
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        from sklearn.preprocessing import StandardScaler
        
        # Initialize scaler if not already done
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Log the actual number of samples per genre
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_distribution = {self.genres[label]: count for label, count in zip(unique_labels, counts)}
        logger.info(f"Class distribution in training data: {class_distribution}")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Calculate class weights to balance training
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        max_count = np.max(class_counts)
        
        # More aggressive balancing by using squared weights
        class_weights = {i: (max_count / count)**1.5 
                        for i, count in enumerate(class_counts) if count > 0}
        
        logger.info(f"Using class weights: {class_weights}")
        
        # Define callbacks with more patience for early stopping
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info(f"Training enhanced genre model with {len(X_train)} samples for {epochs} epochs")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save the scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Evaluate on validation set
        val_loss, val_acc = self.model.evaluate(X_val, y_val)
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        # Initialize genre weights based on inverse frequency in training data
        genre_weights = {}
        for label, count in zip(unique_labels, counts):
            genre = self.genres[label]
            # Use inverse frequency to give more weight to rare genres
            genre_weights[genre] = min(2.0, max(0.5, (total_samples / count) / len(unique_labels)))
        
        # Add default weights for any missing genres
        for genre in self.genres:
            if genre not in genre_weights:
                genre_weights[genre] = 1.0
        
        logger.info(f"Initial genre weights based on data distribution: {genre_weights}")
        joblib.dump(genre_weights, self.genre_weights_path)
        self.genre_weights = genre_weights
        
        # Create and save confusion matrix
        self._create_confusion_matrix(X_val, y_val)
        
        return history.history
    
    def _create_confusion_matrix(self, X_val, y_val):
        """Create and save confusion matrix visualization"""
        # Get predictions
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Determine actual classes present in the data
        unique_classes = np.unique(y_val)
        used_genres = [self.genres[i] for i in unique_classes]
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=used_genres, yticklabels=used_genres)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized)')
        plt.savefig(os.path.join(self.model_dir, 'enhanced_confusion_matrix.png'))
        
        # Also create a count-based confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=used_genres, yticklabels=used_genres)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Counts)')
        plt.savefig(os.path.join(self.model_dir, 'enhanced_confusion_matrix_counts.png'))
        
        # Generate and save classification report
        report = classification_report(y_val, y_pred_classes, target_names=used_genres, output_dict=True)
        
        # Save report as CSV
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(os.path.join(self.model_dir, 'enhanced_classification_report.csv'))
        
        logger.info("Confusion matrix and classification report saved")
    
    def tune_genre_weights(self, X_val, y_val):
        """
        Tune genre weights to improve accuracy for underperforming genres
        
        Args:
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
        """
        # Get predictions
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate per-genre accuracy
        genre_accuracy = {}
        for genre_idx, genre in enumerate(self.genres):
            # Get samples of this genre
            genre_samples = (y_val == genre_idx)
            if not any(genre_samples):
                genre_accuracy[genre] = 1.0  # No samples, assume perfect
                continue
                
            # Calculate accuracy for this genre
            correct = (y_val[genre_samples] == y_pred_classes[genre_samples])
            accuracy = correct.sum() / len(correct)
            genre_accuracy[genre] = accuracy
        
        # Calculate average accuracy
        avg_accuracy = np.mean(list(genre_accuracy.values()))
        
        # Adjust weights for each genre
        for genre, accuracy in genre_accuracy.items():
            if accuracy < avg_accuracy:
                # Increase weight for underperforming genres
                self.genre_weights[genre] = min(2.0, self.genre_weights.get(genre, 1.0) * (avg_accuracy / max(0.01, accuracy)))
            else:
                # Slightly decrease weight for overperforming genres
                self.genre_weights[genre] = max(0.5, self.genre_weights.get(genre, 1.0) * 0.95)
        
        # Normalize weights to keep the average close to 1.0
        weight_sum = sum(self.genre_weights.values())
        norm_factor = len(self.genres) / weight_sum
        for genre in self.genre_weights:
            self.genre_weights[genre] *= norm_factor
        
        # Save updated weights
        joblib.dump(self.genre_weights, self.genre_weights_path)
        logger.info(f"Genre weights tuned and saved: {self.genre_weights}")
    
    def predict(self, features):
        """
        Predict the genre probabilities for a given feature vector
        
        Args:
            features: The feature vector to predict, can be a single vector or multiple segments' features
            
        Returns:
            dict: The predicted probabilities for each genre
        """
        try:
            # Check if we have multiple segments
            if isinstance(features, list) and len(features) > 0:
                # Process each segment and average results
                segment_probs = []
                raw_segment_probs = []  # Store raw probabilities without weights
                
                for segment_features in features:
                    # Scale features
                    segment_features_scaled = self.scaler.transform(segment_features.reshape(1, -1))
                    
                    # Get predictions
                    segment_pred = self.model.predict(segment_features_scaled, verbose=0)
                    segment_probs.append(segment_pred[0])
                    raw_segment_probs.append(segment_pred[0])
                
                # Calculate the confidence-weighted average
                segment_probs = np.array(segment_probs)
                raw_segment_probs = np.array(raw_segment_probs)
                
                # Calculate entropy (uncertainty) of each segment prediction
                entropies = -np.sum(segment_probs * np.log2(np.clip(segment_probs, 1e-10, 1.0)), axis=1)
                max_entropy = -np.log2(1.0/len(self.genres))  # Maximum possible entropy
                
                # Convert entropy to confidence (lower entropy = higher confidence)
                confidences = 1.0 - (entropies / max_entropy)
                
                # Weight predictions by confidence (higher confidence = higher weight)
                # Add a small constant to ensure even low confidence segments contribute
                segment_weights = confidences + 0.1
                segment_weights = segment_weights / np.sum(segment_weights)
                
                logger.info(f"Segment confidences: {confidences}")
                logger.info(f"Segment weights: {segment_weights}")
                
                # Apply weights
                probs = np.zeros_like(segment_probs[0])
                for i, p in enumerate(segment_probs):
                    probs += p * segment_weights[i]
            else:
                # Process single feature vector
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                    
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Get predictions
                probs = self.model.predict(features_scaled, verbose=0)[0]
            
            # Apply genre weights
            weighted_probs = np.zeros_like(probs)
            for i, genre in enumerate(self.genres):
                weight = self.genre_weights.get(genre, 1.0)
                weighted_probs[i] = probs[i] * weight
            
            # Normalize to ensure sum is 1.0
            weighted_probs = weighted_probs / weighted_probs.sum()
            
            # Create result dictionary
            result = {genre: float(weighted_probs[i]) for i, genre in enumerate(self.genres)}
            
            # Log top predictions
            sorted_probs = sorted(result.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_probs[:3]
            logger.info(f"Top 3 predictions: {top_3}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return uniform distribution in case of error
            return {genre: 1.0 / len(self.genres) for genre in self.genres}

def main():
    """Test the enhanced genre classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Music Genre Classification')
    parser.add_argument('--sample-dir', type=str, default='samples',
                       help='Directory containing genre samples')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--test-file', type=str,
                       help='Audio file to test classification on')
    
    args = parser.parse_args()
    
    # Create the enhanced classifier
    classifier = EnhancedGenreClassifier()
    
    if args.test_file:
        # Extract features from test file
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
    else:
        logger.info("No test file specified. Use --test-file to classify an audio file.")

if __name__ == "__main__":
    main() 