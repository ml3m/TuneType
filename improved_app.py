#!/usr/bin/env python3
import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from app.utils.feature_extractor import extract_multi_segment_features
from enhanced_genre_model import EnhancedGenreClassifier

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
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg', 'flac'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the enhanced model
model = EnhancedGenreClassifier()

# Log the loaded genres
logger.info(f"Enhanced model loaded with genres: {model.genres}")

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
        title="Enhanced Music Genre Classifier"
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make prediction using the enhanced model"""
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
        
        # Extract features from multiple segments for more robust analysis
        logger.info("Extracting features from multiple segments")
        features = extract_multi_segment_features(filepath, num_segments=5)
        
        if not features:
            return jsonify({'error': 'Failed to extract features from audio file'}), 400
        
        # Make prediction using the enhanced model
        logger.info(f"Making predictions with {len(features)} feature sets")
        result = model.predict(features)
        
        # Sort genres by probability
        sorted_genres = sorted(result.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate confidence level
        top_genre, top_prob = sorted_genres[0]
        second_prob = sorted_genres[1][1] if len(sorted_genres) > 1 else 0
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
        formatted_results = []
        for genre, prob in sorted_genres:
            formatted_results.append({
                'genre': genre,
                'probability': prob,
                'percentage': f"{prob * 100:.2f}%"
            })
        
        # Check if reliable
        is_reliable = confidence_margin >= 0.1
        
        logger.info(f"Top genre: {top_genre} ({top_prob*100:.2f}%) - Confidence: {confidence_text}")
        
        # Return the results
        return jsonify({
            'success': True,
            'results': formatted_results,
            'confidence': {
                'level': confidence_text,
                'margin': confidence_margin,
                'reliable': is_reliable
            },
            'file': filename
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

@app.route('/compare', methods=['POST'])
def compare():
    """Compare prediction with user feedback to improve the model"""
    try:
        data = request.get_json()
        
        if not data or 'predicted' not in data or 'actual' not in data:
            return jsonify({'error': 'Missing prediction or actual genre information'}), 400
        
        predicted_genre = data['predicted']
        actual_genre = data['actual']
        
        # Verify genres exist in our model
        if predicted_genre not in model.genres or actual_genre not in model.genres:
            return jsonify({'error': 'Invalid genre provided'}), 400
        
        # Adjust genre weights based on feedback
        if predicted_genre != actual_genre:
            # Increase weight for the actual genre
            pred_idx = model.genres.index(predicted_genre)
            actual_idx = model.genres.index(actual_genre)
            
            # Reduce weight for incorrectly predicted genre
            model.genre_weights[predicted_genre] = max(0.5, model.genre_weights.get(predicted_genre, 1.0) * 0.9)
            
            # Increase weight for the correct genre
            model.genre_weights[actual_genre] = min(2.0, model.genre_weights.get(actual_genre, 1.0) * 1.1)
            
            # Save updated weights
            import joblib
            joblib.dump(model.genre_weights, model.genre_weights_path)
            
            logger.info(f"Updated genre weights based on feedback: {predicted_genre} -> {actual_genre}")
            
            return jsonify({
                'success': True,
                'message': 'Thank you for your feedback! Model weights have been adjusted.'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Thank you for confirming the prediction!'
            })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': f'Error processing feedback: {str(e)}'}), 500

@app.route('/genres')
def get_genres():
    """Return the list of supported genres"""
    return jsonify({
        'genres': model.genres
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 