# genre_classifier/views.py
import os
import librosa
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from .forms import UploadAudioForm


# Load the model globally
MODEL_PATH = os.path.join(settings.BASE_DIR, 'genre_classifier',
 'tfjs_model') # Path to your tfjs model
model = tf.keras.models.load_model(MODEL_PATH)


def extract_features(audio_path):
 y, sr = librosa.load(audio_path, duration=30)
 mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
 mfccs_processed = np.mean(mfccs.T, axis=0)
 return mfccs_processed


def predict_genre(features):
 features = features.reshape(1, -1)  # Reshape to (1, num_features)
 prediction = model.predict(features)
 predicted_class = np.argmax(prediction[0])
 genres = ['Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Pop',
 'Rock']  # Replace with your actual genres
 return genres[predicted_class]


def upload_audio(request):
 if request.method == 'POST':
 form = UploadAudioForm(request.POST, request.FILES)
 if form.is_valid():
 audio_file = request.FILES['audio_file']
 # Save the uploaded file to a temporary location
 temp_file_path = os.path.join(settings.MEDIA_ROOT,
 audio_file.name)
 with open(temp_file_path, 'wb+') as destination:
 for chunk in audio_file.chunks():
 destination.write(chunk)


 # Extract features and predict genre
 try:
 features = extract_features(temp_file_path)
 genre = predict_genre(features)
 except Exception as e:
 genre = f"Error processing audio: {e}"
 finally:
 os.remove(temp_file_path)  # Clean up the temporary file


 return render(request, 'genre_classifier/result.html',
 {'genre': genre})
 else:
 return render(request, 'genre_classifier/index.html',
 {'form': form})
 else:
 form = UploadAudioForm()
 return render(request, 'genre_classifier/index.html', {'form': form})
