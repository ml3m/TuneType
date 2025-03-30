# Music Genre Classification - Accuracy Improvements

This document outlines the improvements made to address the issue of inaccurate genre classification, specifically the tendency of the model to classify most songs as "jazz".

## Summary of Improvements

We've made several significant changes to improve the accuracy and reliability of the genre classification system:

1. **Enhanced Feature Extraction**: Added more genre-specific audio features to better differentiate between genres
2. **Improved Model Architecture**: Modified the model to reduce bias toward specific genres
3. **Balanced Training**: Implemented aggressive data augmentation techniques to create a more balanced dataset
4. **Calibration System**: Added a post-processing calibration layer to correct for prediction biases
5. **Multiple Segment Analysis**: Now analyzing multiple segments of each audio file for more accurate predictions

## Technical Details

### 1. Enhanced Feature Extraction

We improved the feature extraction pipeline to capture more genre-specific characteristics:

- **Added Spectral Flatness**: This feature helps detect electronic/synthetic music
- **Improved Rhythm Features**: More detailed analysis of beat patterns to better distinguish dance genres
- **Enhanced Onset Detection**: Better detection of note attacks, helping distinguish sharper genres (rock, metal) from smoother ones (jazz, classical)
- **Genre-Specific EQ Analysis**: Analyzing specific frequency bands relevant to different genres

### 2. Improved Model Architecture

We modified the machine learning model to be more balanced and accurate:

- **Deeper Neural Network**: Increased hidden layer sizes from `(100, 50)` to `(256, 128, 64)` for better feature learning
- **Calibrated SVM**: Added probability calibration to the Support Vector Machine component
- **Optimized KNN**: Changed from Euclidean to Manhattan distance for audio features
- **Temperature Scaling**: Applied temperature scaling to sharpen predictions and reduce hedging
- **Enhanced Class Weighting**: Used `balanced_subsample` instead of `balanced` for better handling of imbalanced genres

### 3. Balanced Training

To address imbalances in the training data:

- **Dynamic Augmentation**: Automatically detects underrepresented genres and applies more augmentation to them
- **Advanced Augmentation Techniques**:
  - Time stretching (variable rates)
  - Pitch shifting (random semitones)
  - EQ simulation (genre-appropriate frequency adjustments)
  - Reverb simulation (for spatial characteristics)
  - Combined transformations for extreme diversity

### 4. Calibration System

We added a calibration layer to correct for known biases:

- **Automatic Calibration Script**: `tune_genre_bias.py` analyzes prediction patterns and computes correction factors
- **Jazz Bias Correction**: Special handling for jazz with an aggressive reduction factor
- **Persistent Calibration**: Calibration factors are saved and applied consistently across all predictions
- **User Feedback System**: API endpoint for updating calibration based on user feedback

### 5. Multiple Segment Analysis

Rather than analyzing just one section of a song:

- **Multiple Sample Points**: Now takes samples from beginning, middle, and end of each track
- **Weighted Combining**: Combines predictions from all segments with appropriate weighting
- **Confidence Calculation**: Calculates prediction confidence based on consistency across segments

## Using the Improved System

### Training with Better Accuracy

To train the model with all improvements:

```bash
python train_with_samples.py --force-extract
```

### Calibrating the Model

To run the automatic calibration (requires sample files):

```bash
python tune_genre_bias.py --sample-dir samples
```

### Running the Web Application

The web application will automatically use all improvements:

```bash
python app.py
```

## Results

The improvements significantly reduce the tendency to classify everything as jazz. The system now:

1. Provides more accurate genre classifications
2. Shows reliable confidence scores
3. Suggests alternative genres when confidence is low
4. Adapts over time through the calibration feedback system

## Future Improvements

While these changes significantly improve accuracy, further enhancements could include:

1. Subgenre detection for more specific classification
2. Integration with music databases for additional metadata-based features
3. Transfer learning with larger pre-trained audio models
4. Active learning components to continuously improve from user feedback 