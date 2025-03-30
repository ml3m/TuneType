import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import joblib
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer

logger = logging.getLogger(__name__)

# Comprehensive list of major music genres
DEFAULT_GENRES = ['blues', 'classical', 'country', 'electronic', 'folk', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock', 'world']

class GenreClassifier:
    def __init__(self, genres=None):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.genres = genres if genres is not None else DEFAULT_GENRES
        
        # Create model directory if it doesn't exist
        os.makedirs('app/models', exist_ok=True)
        
        self.model_path = 'app/models/genre_model.pkl'
        self.scaler_path = 'app/models/scaler.pkl'
        self.feature_selector_path = 'app/models/feature_selector.pkl'
        self.genres_path = 'app/models/genres.pkl'
        
        # Try to load the model or create a new one
        self._load_or_build_model()
    
    def _build_model(self):
        """Build an improved ensemble model for better genre classification"""
        logger.info("Building new enhanced ensemble model for genre classification")
        
        # Random Forest with tuned parameters
        rf = RandomForestClassifier(
            n_estimators=500,  # Increased from 300
            max_depth=None,    # Allow full depth for some trees
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            criterion='entropy'  # Changed from default 'gini'
        )
        
        # Gradient Boosting with tuned parameters
        gb = GradientBoostingClassifier(
            n_estimators=250,  # Increased from 200
            learning_rate=0.05,  # Reduced from 0.1 for better generalization
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.85, 
            max_features='sqrt',
            random_state=42
        )
        
        # Extra Trees - similar to Random Forest but with more randomness
        et = ExtraTreesClassifier(
            n_estimators=500,  # Increased from 300
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            criterion='entropy'
        )
        
        # K-Nearest Neighbors with distance weighting
        knn = KNeighborsClassifier(
            n_neighbors=5,  # Reduced from 7
            weights='distance',
            metric='minkowski',  # Changed from euclidean
            p=2,  # Euclidean distance
            n_jobs=-1
        )
        
        # Support Vector Classifier with RBF kernel
        svm = SVC(
            probability=True,
            C=10.0,
            gamma='scale',
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            decision_function_shape='ovr'  # One-vs-rest for multiclass
        )
        
        # Neural Network Classifier
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        
        # AdaBoost Classifier
        ada = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
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
            weights=[3, 2, 3, 1, 3, 2, 1]  # Higher weights for RF, ET, and SVM
        )
        
        # Create feature selector using Recursive Feature Elimination with CV
        # This will be fit separately during training
        self.feature_selector = SelectFromModel(
            estimator=ExtraTreesClassifier(n_estimators=300, random_state=42),
            threshold="1.5*mean"
        )
        
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
        
        return voting_clf
    
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
                
                logger.info("Model loaded successfully")
            else:
                logger.info("No existing model found, building a new one")
                self.model = self._build_model()
                self._train_dummy_data()  # Train on dummy data to initialize
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Building a new model")
            self.model = self._build_model()
            self._train_dummy_data()
    
    def _train_dummy_data(self):
        """Train on dummy data to initialize the model"""
        logger.info("Training model with dummy data for initialization")
        
        # Create dummy data with appropriate feature count (matches our feature extractor)
        X_dummy = np.random.rand(500, 135)  # Match with actual feature count (135)
        
        # Create dummy labels (genres.length classes)
        y_dummy = np.random.randint(0, len(self.genres), 500)
        
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
        self.model.fit(X_dummy_scaled, y_dummy)
        
        # Save the model and scaler
        self.save_model()
        
        logger.info("Dummy training completed and model initialized")
    
    def save_model(self):
        """Save the model and scaler for future use"""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            if self.feature_selector is not None:
                joblib.dump(self.feature_selector, self.feature_selector_path)
            joblib.dump(self.genres, self.genres_path)
            logger.info("Model, scaler, and genres saved")
    
    def tune_hyperparameters(self, X_train, y_train):
        """
        Tune the hyperparameters of the model using grid search
        
        Args:
            X_train: Features for training
            y_train: Labels for training
            
        Returns:
            Best parameters dict
        """
        logger.info("Tuning hyperparameters for best performance...")
        
        # We'll tune the Random Forest component of our ensemble
        # Using a more comprehensive grid search
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Define more comprehensive parameter grid
        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2']
        }
        
        # Create scorer that balances accuracy for all classes
        balanced_scorer = make_scorer(balanced_accuracy_score)
        
        # Create grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=balanced_scorer,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Return best parameters
        return best_params
    
    def predict(self, features):
        """
        Predict the genre of a song based on extracted features
        
        Args:
            features: numpy array of extracted features
            
        Returns:
            numpy array of probabilities for each genre
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Handle feature selection if available
        if self.feature_selector is not None:
            try:
                # Check if feature selector is fitted
                if hasattr(self.feature_selector, 'get_support'):
                    features = self.feature_selector.transform(features)
                    logger.debug(f"Applied feature selection, using {features.shape[1]} features")
            except Exception as e:
                logger.warning(f"Failed to apply feature selection: {str(e)}")
                # Continue with all features
                pass
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        # Make prediction
        try:
            prediction_proba = self.model.predict_proba(scaled_features)
            logger.info(f"Prediction made with shape: {prediction_proba.shape}")
            return prediction_proba
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Fallback to hard voting if soft fails
            try:
                if hasattr(self.model, 'set_params'):
                    self.model.set_params(voting='hard')
                    class_prediction = self.model.predict(scaled_features)
                    # Convert to one-hot like format
                    proba = np.zeros((len(class_prediction), len(self.genres)))
                    for i, label in enumerate(class_prediction):
                        proba[i, label] = 1.0
                    
                    # Reset to soft voting
                    self.model.set_params(voting='soft')
                    return proba
                else:
                    raise e
            except:
                logger.error("Fallback prediction also failed")
                raise e
    
    def train(self, features, labels, epochs=None, batch_size=None, cross_validation=True):
        """
        Train the model on new data
        
        Args:
            features: numpy array of features
            labels: numpy array of labels (class indices or one-hot encoded)
            epochs: not used, kept for API compatibility
            batch_size: not used, kept for API compatibility
            cross_validation: whether to perform cross-validation
        """
        if self.model is None:
            self.model = self._build_model()
        
        # Convert one-hot encoded labels to class indices if necessary
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)
        
        # Apply feature selection if needed
        if self.feature_selector is not None:
            try:
                # Do feature selection on training data
                self.feature_selector.fit(features, labels)
                selected_features = self.feature_selector.transform(features)
                feature_count_before = features.shape[1]
                feature_count_after = selected_features.shape[1]
                logger.info(f"Applied feature selection, reduced from {feature_count_before} to {feature_count_after} features")
                
                # Get feature importances for logging
                if hasattr(self.feature_selector, 'estimator_') and hasattr(self.feature_selector.estimator_, 'feature_importances_'):
                    importances = self.feature_selector.estimator_.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    top_indices = indices[:min(10, len(indices))]
                    logger.info("Top 10 features by importance:")
                    for i, idx in enumerate(top_indices):
                        logger.info(f"{i+1}. Feature {idx}: {importances[idx]:.6f}")
                
                features = selected_features
            except Exception as e:
                logger.warning(f"Failed to apply feature selection: {str(e)}")
                # Continue without feature selection
        
        # Update the scaler with new data and scale the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train the model
        logger.info(f"Training ensemble model on {features.shape[0]} samples with {features.shape[1]} features")
        
        if cross_validation and features.shape[0] >= 50:
            # Perform cross-validation and parameter tuning
            try:
                # Use stratified k-fold to handle imbalanced classes
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Record scores for different models in the ensemble
                scores = {}
                
                # Cross-validate individual models
                for name, estimator in self.model.estimators:
                    logger.info(f"Cross-validating {name} classifier")
                    cv_scores = cross_val_score(
                        estimator, scaled_features, labels, 
                        cv=cv, scoring='balanced_accuracy'
                    )
                    avg_score = cv_scores.mean()
                    scores[name] = avg_score
                    logger.info(f"{name} CV score: {avg_score:.4f} (±{cv_scores.std():.4f})")
                
                # Tune the best performing model
                best_model = max(scores.items(), key=lambda x: x[1])[0]
                logger.info(f"Tuning hyperparameters for {best_model} (best CV score: {scores[best_model]:.4f})")
                
                # Get best parameters
                best_params = self.tune_hyperparameters(scaled_features, labels)
                
                # Update the model with best parameters
                for name, estimator in self.model.estimators:
                    if name == best_model:
                        logger.info(f"Updating {name} with tuned parameters")
                        for param, value in best_params.items():
                            if hasattr(estimator, param):
                                setattr(estimator, param, value)
            except Exception as e:
                logger.warning(f"Error during cross-validation: {str(e)}")
                logger.info("Continuing with default parameters")
        
        # Train the model with all data
        self.model.fit(scaled_features, labels)
        
        # Evaluate on training data (this is just to get a baseline, not for actual evaluation)
        train_preds = self.model.predict(scaled_features)
        train_accuracy = accuracy_score(labels, train_preds)
        balanced_acc = balanced_accuracy_score(labels, train_preds)
        f1 = f1_score(labels, train_preds, average='weighted')
        
        logger.info(f"Training metrics - Accuracy: {train_accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}, F1: {f1:.4f}")
        
        # Save the updated model
        self.save_model()
        
        logger.info("Model training complete")
        
        # Return history for API compatibility
        return {'accuracy': [train_accuracy], 'balanced_accuracy': [balanced_acc], 'f1': [f1]} 