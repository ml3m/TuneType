import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from app.utils.feature_extractor import extract_features
from app.utils.model import GenreClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app with correct template and static folders
app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')
app.config['UPLOAD_FOLDER'] = 'tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model
model = GenreClassifier()

# Log the loaded genres
logger.info(f"Model loaded with genres: {model.genres}")

# Load genre confidence calibration factors if available
# This helps prevent bias toward specific genres
CALIBRATION_FACTORS_FILE = 'app/models/genre_calibration.json'
calibration_factors = {}

try:
    if os.path.exists(CALIBRATION_FACTORS_FILE):
        with open(CALIBRATION_FACTORS_FILE, 'r') as f:
            calibration_factors = json.load(f)
        logger.info(f"Loaded genre calibration factors: {calibration_factors}")
    else:
        # Initialize with default values (1.0) for all genres
        calibration_factors = {genre: 1.0 for genre in model.genres}
        # Save default calibration factors
        with open(CALIBRATION_FACTORS_FILE, 'w') as f:
            json.dump(calibration_factors, f, indent=2)
        logger.info("Initialized default genre calibration factors")
except Exception as e:
    logger.error(f"Error loading calibration factors: {str(e)}")
    # Fallback to default values
    calibration_factors = {genre: 1.0 for genre in model.genres}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page"""
    # Get available genres from the model
    genres = model.genres
    return render_template(
        'index.html', 
        genres=genres,
        title="Music Genre Classifier"
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make prediction"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not allowed. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved to {filepath}")
        
        # Extract features from multiple segments for more accurate prediction
        segments = 3  # Extract from beginning, middle, and end
        segment_duration = 30  # seconds
        all_features = []
        
        # Get file duration
        import librosa
        duration = librosa.get_duration(path=filepath)
        logger.info(f"File duration: {duration:.2f} seconds")
        
        if duration <= segment_duration:
            # File is shorter than segment duration, just analyze once
            logger.info("Short file, extracting features once")
            features = extract_features(filepath)
            if features is not None:
                all_features.append(features)
        else:
            # Sample from multiple segments
            logger.info(f"Extracting features from {segments} segments")
            
            # Start segment
            features_start = extract_features(filepath, offset=0)
            if features_start is not None:
                all_features.append(features_start)
            
            # Middle segment
            mid_offset = max(0, (duration - segment_duration) / 2)
            features_mid = extract_features(filepath, offset=mid_offset)
            if features_mid is not None:
                all_features.append(features_mid)
            
            # End segment (if duration is sufficient)
            if duration > segment_duration * 2:
                end_offset = max(0, duration - segment_duration)
                features_end = extract_features(filepath, offset=end_offset)
                if features_end is not None:
                    all_features.append(features_end)
        
        if not all_features:
            return jsonify({'error': 'Failed to extract features from audio file'}), 400
        
        # Convert all features to numpy array
        all_features = np.array(all_features)
        
        # Make prediction for each segment and combine
        logger.info(f"Making predictions with {len(all_features)} feature sets")
        
        # Process each feature set
        all_probabilities = []
        for features in all_features:
            # Reshape if needed
            features_reshaped = features.reshape(1, -1)
            
            # Make prediction
            probabilities = model.predict(features_reshaped)
            all_probabilities.append(probabilities[0])
        
        # Average predictions from all segments
        average_probabilities = np.mean(all_probabilities, axis=0)
        
        # Apply calibration factors to combat genre bias
        calibrated_probs = np.zeros_like(average_probabilities)
        for i, genre in enumerate(model.genres):
            factor = calibration_factors.get(genre, 1.0)
            calibrated_probs[i] = average_probabilities[i] * factor
        
        # Normalize to ensure sum is 1.0
        calibrated_probs = calibrated_probs / calibrated_probs.sum()
        
        # Sort genres by probability
        sorted_indices = np.argsort(-calibrated_probs)
        sorted_genres = [model.genres[i] for i in sorted_indices]
        sorted_probs = calibrated_probs[sorted_indices].tolist()
        
        # Calculate confidence level
        top_prob = sorted_probs[0]
        second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
        confidence_margin = top_prob - second_prob
        
        # Determine confidence level text
        if confidence_margin > 0.3:
            confidence_text = "Very High"
        elif confidence_margin > 0.2:
            confidence_text = "High"
        elif confidence_margin > 0.1:
            confidence_text = "Moderate"
        else:
            confidence_text = "Low"
        
        # Format response
        results = []
        for genre, prob in zip(sorted_genres, sorted_probs):
            results.append({
                'genre': genre,
                'probability': prob,
                'percentage': f"{prob * 100:.2f}%"
            })
        
        # Check if top prediction is likely to be correct
        is_reliable = confidence_margin >= 0.1
        
        logger.info(f"Top predicted genre: {results[0]['genre']} ({results[0]['percentage']}) - Confidence: {confidence_text}")
        
        # If confidence is very low, suggest multiple genres
        predicted_genres = [sorted_genres[0]]
        if confidence_margin < 0.1 and len(sorted_genres) > 1:
            predicted_genres.append(sorted_genres[1])
            if confidence_margin < 0.05 and len(sorted_genres) > 2:
                predicted_genres.append(sorted_genres[2])
        
        # Create better response with more information
        response = {
            'success': True,
            'filename': filename,
            'top_genre': results[0]['genre'],
            'top_probability': results[0]['probability'],
            'confidence': confidence_text,
            'is_reliable': is_reliable,
            'suggested_genres': predicted_genres,
            'results': results,
            'analyzed_segments': len(all_features)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/visualize/<filename>')
def visualize(filename):
    """Render visualization page for a file"""
    return render_template('visualize.html', filename=filename)

@app.route('/about')
def about():
    """Render about page with model information"""
    model_info = {
        'genres': model.genres,
        'num_genres': len(model.genres),
        'feature_dim': model.model.estimators_[0].feature_importances_.shape[0] if hasattr(model.model, 'estimators_') else None,
    }
    return render_template('about.html', model_info=model_info)

@app.route('/update-calibration', methods=['POST'])
def update_calibration():
    """
    Allow updating calibration factors based on feedback.
    This helps improve the model's performance over time.
    """
    try:
        data = request.json
        if not data or 'genre' not in data or 'adjustment' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        genre = data['genre']
        adjustment = float(data['adjustment'])
        
        # Validate the adjustment value (prevent extreme values)
        if adjustment < 0.5 or adjustment > 2.0:
            return jsonify({'error': 'Adjustment value must be between 0.5 and 2.0'}), 400
        
        # Only allow adjusting known genres
        if genre not in model.genres:
            return jsonify({'error': f'Unknown genre: {genre}'}), 400
        
        # Update the calibration factor
        current_factor = calibration_factors.get(genre, 1.0)
        new_factor = current_factor * adjustment
        
        # Limit to reasonable range (0.25 to 4.0)
        new_factor = max(0.25, min(4.0, new_factor))
        
        calibration_factors[genre] = new_factor
        
        # Save updated calibration factors
        with open(CALIBRATION_FACTORS_FILE, 'w') as f:
            json.dump(calibration_factors, f, indent=2)
        
        logger.info(f"Updated calibration factor for {genre}: {current_factor} -> {new_factor}")
        
        return jsonify({
            'success': True,
            'genre': genre,
            'previous_factor': current_factor,
            'new_factor': new_factor
        })
        
    except Exception as e:
        logger.error(f"Error updating calibration: {str(e)}")
        return jsonify({'error': f'Error updating calibration: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 