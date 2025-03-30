import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer
import warnings

# Try to import TensorFlow, but make it optional
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    # Suppress TensorFlow warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TensorFlow is available and will be used for deep learning")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow is not available. Deep learning features will be disabled.")

# Comprehensive list of major music genres
DEFAULT_GENRES = ['blues', 'classical', 'country', 'electronic', 'folk', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock', 'world']

class GenreClassifier:
    def __init__(self, genres=None, use_deep_learning=False):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.genres = genres if genres is not None else DEFAULT_GENRES
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        if use_deep_learning and not TENSORFLOW_AVAILABLE:
            logger.warning("Deep learning was requested but TensorFlow is not available. Falling back to ensemble model only.")
        self.deep_model = None
        self.calibration_factors = None
        
        # Create model directory if it doesn't exist
        os.makedirs('app/models', exist_ok=True)
        
        self.model_path = 'app/models/genre_model.pkl'
        self.scaler_path = 'app/models/scaler.pkl'
        self.feature_selector_path = 'app/models/feature_selector.pkl'
        self.genres_path = 'app/models/genres.pkl'
        self.deep_model_path = 'app/models/deep_genre_model'
        self.calibration_path = 'app/models/calibration_factors.json'
        
        # Try to load the model or create a new one
        self._load_or_build_model()
    
    def _build_model(self):
        """Build an improved ensemble model for better genre classification"""
        logger.info("Building enhanced ensemble model for genre classification")
        
        # Random Forest with tuned parameters for more balanced predictions
        rf = RandomForestClassifier(
            n_estimators=800,  # Increased from 500
            max_depth=40,      # Limit depth to prevent overfitting
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample',  # Better for imbalanced genres
            random_state=42,
            n_jobs=-1,
            criterion='entropy'  # Changed from default 'gini'
        )
        
        # Gradient Boosting with tuned parameters
        gb = GradientBoostingClassifier(
            n_estimators=300,  # Increased from 250
            learning_rate=0.03,  # Reduced from 0.05 for better generalization
            max_depth=8,      # Reduced from 12
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,    # Reduced from 0.85
            max_features=0.7, # Use 70% of features
            random_state=42
        )
        
        # Extra Trees - similar to Random Forest but with more randomness
        et = ExtraTreesClassifier(
            n_estimators=800,  # Increased from 500
            max_depth=40,      # Limit depth to prevent overfitting
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample',  # Better for imbalanced genres
            random_state=42,
            n_jobs=-1,
            criterion='entropy'
        )
        
        # K-Nearest Neighbors with distance weighting and optimized N
        knn = KNeighborsClassifier(
            n_neighbors=7,    # Increased from 5
            weights='distance',
            metric='minkowski',
            p=1,              # Manhattan distance works better for audio features
            n_jobs=-1,
            leaf_size=10      # Optimized for audio features
        )
        
        # Support Vector Classifier with RBF kernel - calibrated for better probabilities
        svm_base = SVC(
            probability=True,
            C=15.0,           # Increased from 10.0
            gamma='auto',     # Changed from 'scale'
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            decision_function_shape='ovr',  # One-vs-rest for multiclass
            cache_size=1000   # Increase cache for faster training
        )
        
        # Use calibration to improve probability estimates
        svm = CalibratedClassifierCV(
            base_estimator=svm_base,
            method='sigmoid',  # Platt scaling
            cv=3,
            n_jobs=-1
        )
        
        # Neural Network Classifier with deeper architecture
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Deeper network
            activation='relu',
            solver='adam',
            alpha=0.0005,      # Increased regularization
            batch_size=64,     # Explicit batch size
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,     # Double iterations
            early_stopping=True, # Add early stopping
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        
        # AdaBoost Classifier
        ada = AdaBoostClassifier(
            n_estimators=150,  # Increased from 100
            learning_rate=0.1,
            random_state=42,
            algorithm='SAMME.R'  # Use real-valued prediction
        )
        
        # Create a more sophisticated voting classifier that combines all models
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('et', et),
                ('knn', knn),
                ('svm', svm),
                ('mlp', mlp),
                ('ada', ada)
            ],
            voting='soft',  # Use probabilities for weighted voting
            weights=[3, 2, 3, 1, 3, 2, 1],  # Higher weights for RF, ET, and SVM
            n_jobs=-1       # Use all cores
        )
        
        # Create feature selector using Recursive Feature Elimination with CV
        self.feature_selector = SelectFromModel(
            estimator=ExtraTreesClassifier(
                n_estimators=500, 
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ),
            threshold="1.25*mean"  # More permissive threshold to keep more features
        )
        
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler(quantile_range=(2.5, 97.5))  # More robust to extreme outliers
        
        return voting_clf
    
    def _build_deep_learning_model(self, input_shape=(135,), num_classes=12):
        """Build a deep learning model for better genre classification"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("Cannot build deep learning model: TensorFlow is not available")
            return None
            
        logger.info("Building deep learning model for genre classification")
        
        # Create a neural network with residual connections
        inputs = tf.keras.Input(shape=input_shape)
        
        # First dense block
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual blocks
        for i in range(3):
            # Store the input to the block
            block_input = x
            
            # Apply transformations
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # Project input if needed (if shapes don't match)
            if block_input.shape[-1] != x.shape[-1]:
                block_input = layers.Dense(256, use_bias=False)(block_input)
            
            # Add the residual connection
            x = layers.Add()([x, block_input])
            x = layers.Activation('relu')(x)
        
        # Final layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer with softmax activation
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with advanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=True
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Deep learning model created with {model.count_params()} parameters")
        return model
    
    def _load_or_build_model(self):
        """Try to load a saved model or build a new one if not available"""
        try:
            if os.path.exists(self.model_path):
                logger.info("Loading existing model")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                if os.path.exists(self.feature_selector_path):
                    self.feature_selector = joblib.load(self.feature_selector_path)
                
                # Try to load custom genres if available
                if os.path.exists(self.genres_path):
                    self.genres = joblib.load(self.genres_path)
                    logger.info(f"Loaded custom genres: {self.genres}")
                
                # Try to load deep learning model if available
                if self.use_deep_learning and os.path.exists(self.deep_model_path + ".h5"):
                    try:
                        self.deep_model = tf.keras.models.load_model(self.deep_model_path + ".h5")
                        logger.info("Deep learning model loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading deep learning model: {str(e)}")
                        self.deep_model = None
                
                # Try to load calibration factors if available
                if os.path.exists(self.calibration_path):
                    try:
                        import json
                        with open(self.calibration_path, 'r') as f:
                            self.calibration_factors = json.load(f)
                        logger.info(f"Loaded calibration factors for {len(self.calibration_factors)} genres")
                    except Exception as e:
                        logger.error(f"Error loading calibration factors: {str(e)}")
                        self.calibration_factors = None
                
                logger.info("Model loaded successfully")
            else:
                logger.info("No existing model found, building a new one")
                self.model = self._build_model()
                if self.use_deep_learning:
                    self.deep_model = self._build_deep_learning_model(num_classes=len(self.genres))
                self._train_dummy_data()  # Train on dummy data to initialize
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Building a new model")
            self.model = self._build_model()
            if self.use_deep_learning and TENSORFLOW_AVAILABLE:
                self.deep_model = self._build_deep_learning_model(num_classes=len(self.genres))
            self._train_dummy_data()
    
    def _train_dummy_data(self):
        """Train on dummy data to initialize the model"""
        logger.info("Training model with dummy data for initialization")
        
        # Create dummy data with appropriate feature count (matches our feature extractor)
        X_dummy = np.random.rand(500, 135)  # Match with actual feature count (135)
        
        # Create balanced dummy labels for all genres
        samples_per_genre = 500 // len(self.genres)
        y_dummy = np.array([g for g in range(len(self.genres)) for _ in range(samples_per_genre)])
        
        # If needed, add a few more samples to reach 500
        remaining = 500 - len(y_dummy)
        if remaining > 0:
            y_dummy = np.append(y_dummy, np.random.randint(0, len(self.genres), remaining))
        
        # Fit the feature selector on dummy data
        if self.feature_selector is not None:
            try:
                X_dummy_selected = self.feature_selector.fit_transform(X_dummy, y_dummy)
                logger.info(f"Feature selector reduced features from {X_dummy.shape[1]} to {X_dummy_selected.shape[1]}")
            except Exception as e:
                logger.warning(f"Could not fit feature selector on dummy data: {str(e)}")
                X_dummy_selected = X_dummy
        else:
            X_dummy_selected = X_dummy
            
        # Fit the scaler on dummy data
        X_dummy_scaled = self.scaler.fit_transform(X_dummy_selected)
        
        # Train the model on dummy data
        try:
            self.model.fit(X_dummy_scaled, y_dummy)
            logger.info("Model initialized with dummy data")
            
            # Train deep learning model if available
            if self.use_deep_learning and self.deep_model is not None and TENSORFLOW_AVAILABLE:
                # Basic validation split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_dummy_scaled, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy
                )
                
                # Early stopping to prevent overfitting
                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # Train for a small number of epochs for initialization
                self.deep_model.fit(
                    X_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                logger.info("Deep learning model initialized with dummy data")
            
            # Save the initialized model
            self.save_model()
        except Exception as e:
            logger.error(f"Error training model on dummy data: {str(e)}")
    
    def save_model(self):
        """Save the trained model to disk"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, self.feature_selector_path)
        joblib.dump(self.genres, self.genres_path)
        
        # Save deep learning model if available
        if self.use_deep_learning and self.deep_model is not None and TENSORFLOW_AVAILABLE:
            try:
                self.deep_model.save(self.deep_model_path + ".h5")
                logger.info(f"Deep learning model saved to {self.deep_model_path}.h5")
            except Exception as e:
                logger.error(f"Error saving deep learning model: {str(e)}")
        
        # Save calibration factors if available
        if self.calibration_factors is not None:
            try:
                import json
                with open(self.calibration_path, 'w') as f:
                    json.dump(self.calibration_factors, f)
                logger.info(f"Calibration factors saved to {self.calibration_path}")
            except Exception as e:
                logger.error(f"Error saving calibration factors: {str(e)}")
        
        logger.info(f"Model and related components saved to {os.path.dirname(self.model_path)}")
    
    def predict(self, features):
        """
        Predict the genre probabilities for a given feature vector
        
        Args:
            features: The feature vector to predict, can be a single vector or multiple segments' features
            
        Returns:
            numpy.ndarray: The predicted probabilities for each genre
        """
        # Check if we have multiple segments' worth of features
        if isinstance(features, list) and len(features) > 0:
            # We have multiple segments, predict each one
            segment_probs = []
            for segment_features in features:
                # Get probabilities for this segment
                if segment_features.ndim == 1:
                    segment_features = segment_features.reshape(1, -1)
                segment_probs.append(self._predict_single(segment_features))
            
            # Average probabilities across segments
            segment_probs = np.array(segment_probs)
            
            # Weight predictions by confidence
            # Higher max values get more weight
            segment_max_probs = np.max(segment_probs, axis=1)
            segment_weights = segment_max_probs / np.sum(segment_max_probs)
            
            # Apply weights
            weighted_probs = np.zeros_like(segment_probs[0])
            for i, probs in enumerate(segment_probs):
                weighted_probs += probs * segment_weights[i]
            
            return weighted_probs
        else:
            # Single feature vector, use standard prediction
            return self._predict_single(features)
    
    def _predict_single(self, features):
        """Make a prediction using a single feature vector"""
        try:
            # Process features
            features_reshaped = features.reshape(1, -1) if features.ndim == 1 else features
            
            # Apply feature selection if available
            if self.feature_selector is not None:
                try:
                    features_selected = self.feature_selector.transform(features_reshaped)
                except Exception as e:
                    logger.warning(f"Feature selection failed: {str(e)}. Using all features.")
                    features_selected = features_reshaped
            else:
                features_selected = features_reshaped
            
            # Scale features
            features_scaled = self.scaler.transform(features_selected)
            
            # Get base predictions from ensemble model
            base_probs = self.model.predict_proba(features_scaled)
            
            # Get predictions from deep learning model if available
            if self.use_deep_learning and self.deep_model is not None and TENSORFLOW_AVAILABLE:
                try:
                    # Predict with deep model
                    deep_probs = self.deep_model.predict(features_scaled, verbose=0)
                    
                    # Combine predictions
                    # We give more weight to the deep model
                    combined_probs = 0.4 * base_probs + 0.6 * deep_probs
                except Exception as e:
                    logger.error(f"Deep model prediction failed: {str(e)}. Using only ensemble model.")
                    combined_probs = base_probs
            else:
                combined_probs = base_probs
            
            # Apply calibration if available
            if self.calibration_factors is not None:
                try:
                    calibrated_probs = np.zeros_like(combined_probs)
                    for i, genre in enumerate(self.genres):
                        if genre in self.calibration_factors:
                            # Apply calibration for this genre
                            calibration_factor = self.calibration_factors[genre]
                            calibrated_probs[0, i] = combined_probs[0, i] * calibration_factor
                        else:
                            # No calibration available
                            calibrated_probs[0, i] = combined_probs[0, i]
                    
                    # Renormalize to ensure probabilities sum to 1
                    row_sums = calibrated_probs.sum(axis=1, keepdims=True)
                    calibrated_probs = calibrated_probs / row_sums
                    
                    return calibrated_probs
                except Exception as e:
                    logger.error(f"Calibration failed: {str(e)}. Using uncalibrated probabilities.")
                    return combined_probs
            else:
                return combined_probs
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return uniform distribution in case of error
            uniform = np.ones((1, len(self.genres))) / len(self.genres)
            return uniform
    
    def train(self, features, labels, epochs=None, batch_size=None, cross_validation=True):
        """
        Train the model with the provided features and labels
        
        Args:
            features: Feature vectors for training
            labels: Genre labels for training
            epochs: Number of epochs for deep learning model
            batch_size: Batch size for deep learning model
            cross_validation: Whether to use cross-validation for performance estimation
            
        Returns:
            dict: Training results including accuracy metrics
        """
        logger.info(f"Training genre classification model with {len(features)} samples")
        
        try:
            # Apply feature selection if needed
            if self.feature_selector is not None:
                logger.info("Applying feature selection")
                features_selected = self.feature_selector.fit_transform(features, labels)
                logger.info(f"Features reduced from {features.shape[1]} to {features_selected.shape[1]}")
            else:
                features_selected = features
            
            # Scale features
            logger.info("Scaling features")
            features_scaled = self.scaler.fit_transform(features_selected)
            
            # Train the ensemble model
            results = {}
            if cross_validation:
                # Stratified cross-validation for more reliable evaluation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Compute cross-validation scores
                logger.info("Performing cross-validation")
                cv_scores = cross_val_score(
                    self.model, features_scaled, labels, 
                    cv=cv, scoring='balanced_accuracy',
                    n_jobs=-1
                )
                
                # Log cross-validation results
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                logger.info(f"Cross-validation balanced accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
                
                results['cv_scores'] = cv_scores
                results['mean_cv_score'] = mean_cv_score
                results['std_cv_score'] = std_cv_score
            
            # Train the model on the entire dataset
            logger.info("Training final model on entire dataset")
            self.model.fit(features_scaled, labels)
            
            # If using deep learning, split and train deep model
            if self.use_deep_learning and self.deep_model is not None and TENSORFLOW_AVAILABLE:
                logger.info("Training deep learning model")
                
                # Split data for validation
                X_train, X_val, y_train, y_val = train_test_split(
                    features_scaled, labels, test_size=0.15, random_state=42, stratify=labels
                )
                
                # Set up callbacks
                callback_list = [
                    callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=20,
                        restore_best_weights=True
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=1e-6
                    )
                ]
                
                # Set default epochs and batch size if not provided
                if epochs is None:
                    epochs = 200  # More epochs with early stopping
                if batch_size is None:
                    batch_size = 32
                
                # Fit the model
                deep_history = self.deep_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=callback_list,
                    verbose=1
                )
                
                # Evaluate deep model on validation set
                deep_val_loss, deep_val_acc = self.deep_model.evaluate(X_val, y_val, verbose=0)
                logger.info(f"Deep learning validation accuracy: {deep_val_acc:.4f}")
                
                results['deep_val_accuracy'] = deep_val_acc
                results['deep_val_loss'] = deep_val_loss
            
            # Save the trained model
            self.save_model()
            
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def get_genres(self):
        """Return the list of genres supported by the model"""
        return self.genres
    
    def set_calibration_factors(self, calibration_factors):
        """Set calibration factors for adjusting predictions"""
        self.calibration_factors = calibration_factors
        self.save_model()  # Save to persist calibration factors
    
    def get_calibration_factors(self):
        """Get the current calibration factors"""
        return self.calibration_factors 