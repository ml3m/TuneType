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
# The method is called _load_or_build_model() internally in the constructor
# so we don't need to call it explicitly

# Log the loaded genres
logger.info(f"Model loaded successfully. Genres: {model.genres}")

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
        
        # Extract features
        logger.info(f"Extracting features from {filename}")
        features = extract_features(filepath)
        
        if features is None:
            return jsonify({'error': 'Failed to extract features from audio file'}), 400
        
        # Make prediction - make sure features are correctly shaped
        logger.info(f"Making prediction with features shape: {features.shape}")
        
        # Reshape if needed - our new extractor returns a 1D array of 135 features
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Check feature count
        if features.shape[1] != 135:
            logger.warning(f"Feature count mismatch: got {features.shape[1]}, expected 135")
            # Try to fix by padding or truncating
            if features.shape[1] > 135:
                features = features[:, :135]
            else:
                padding = np.zeros((features.shape[0], 135 - features.shape[1]))
                features = np.hstack((features, padding))
        
        probabilities = model.predict(features)
        
        # Sort genres by probability
        sorted_indices = np.argsort(-probabilities[0])
        sorted_genres = [model.genres[i] for i in sorted_indices]
        sorted_probs = probabilities[0][sorted_indices].tolist()
        
        # Format response
        results = []
        for genre, prob in zip(sorted_genres, sorted_probs):
            results.append({
                'genre': genre,
                'probability': prob,
                'percentage': f"{prob * 100:.2f}%"
            })
        
        logger.info(f"Top predicted genre: {results[0]['genre']} ({results[0]['percentage']})")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'top_genre': results[0]['genre'],
            'top_probability': results[0]['probability'],
            'results': results
        })
    
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

if __name__ == '__main__':
    app.run(debug=True) 