# TuneType

<div align="center">
  <img src="logo.png" alt="TuneType Logo" width="200"/>
  <h3>Advanced Music Genre Classification System</h3>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![librosa](https://img.shields.io/badge/librosa-0.9+-red.svg)](https://librosa.org/)

## 🎵 Overview

TuneType is an advanced music genre classification system that leverages machine learning and audio signal processing to accurately identify the genre of any music track. The system analyzes audio features such as spectral characteristics, rhythm patterns, harmonic content, and tonal features to classify songs into major music genres.

### 📊 Key Features

- **High-Accuracy Classification**: Ensemble ML model combining multiple classifiers for robust genre detection
- **Support for Major Genres**: Classifies music into 12 major genres including rock, pop, jazz, classical, hip-hop, and more
- **Comprehensive Audio Analysis**: Extracts 135+ audio features for nuanced genre identification
- **Interactive Web Interface**: User-friendly web app for uploading and analyzing music files
- **Detailed Visualizations**: Genre probability distribution and audio feature visualizations
- **Cross-Platform Compatibility**: Processes common audio formats (.mp3, .wav, .ogg)

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TuneType.git
cd TuneType
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download genre samples for model training:
```bash
python download_samples.py
```

5. Train the classification model:
```bash
python train_with_samples.py
```

6. Start the web application:
```bash
python app.py
```

7. Access the application in your browser at:
```
http://localhost:5001
```

## 📝 Usage

1. **Upload a Music File**: Click the upload button and select an audio file (.mp3, .wav, .ogg)
2. **Analyze**: Click the "Analyze" button to process the audio and detect the genre
3. **View Results**: The system will display:
   - Top predicted genre
   - Confidence scores for all genres
   - Detailed audio feature visualizations
4. **Explore Features**: Navigate to the visualization page for deeper analysis of audio characteristics

## 🧠 How It Works

TuneType operates through a multi-stage pipeline:

1. **Feature Extraction**: Converts raw audio into 135+ numerical features including:
   - Mel-frequency cepstral coefficients (MFCCs)
   - Spectral characteristics (centroid, rolloff, contrast)
   - Rhythm features (beat statistics, tempo)
   - Harmonic/percussive component analysis
   - Tonal features (chroma, key detection)

2. **Classification**: Uses an ensemble model combining:
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Extra Trees Classifier
   - Support Vector Machines
   - Neural Networks
   - K-Nearest Neighbors

3. **Result Presentation**: Displays genre predictions with confidence scores and interactive visualizations

## 🛠️ Technical Architecture

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn, librosa, numpy
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Audio Processing**: librosa, soundfile

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [librosa](https://librosa.org/) - Audio and music processing in Python
- [scikit-learn](https://scikit-learn.org/) - Machine learning in Python
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Internet Archive](https://archive.org/) - Source for public domain music samples 
