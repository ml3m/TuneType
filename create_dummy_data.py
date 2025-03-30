import os
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050  # Standard sample rate used in our project
DURATION = 10  # Duration in seconds for each synthetic sample

def create_sine_wave(freq, duration, sr=SAMPLE_RATE):
    """Creates a sine wave of a given frequency"""
    t = np.linspace(0, duration, int(sr * duration), False)
    return np.sin(2 * np.pi * freq * t)

def add_noise(signal, noise_level=0.1):
    """Adds white noise to a signal"""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def create_classical_sample():
    """Creates a synthetic classical music sample"""
    # Classical: clean sine waves with harmonics
    base_tone = create_sine_wave(440, DURATION)  # A4
    overtone1 = create_sine_wave(880, DURATION) * 0.5  # A5 (first overtone)
    overtone2 = create_sine_wave(1320, DURATION) * 0.3  # E6 (second overtone)
    
    # Mix the tones with gentle amplitude envelope
    t = np.linspace(0, 1, int(SAMPLE_RATE * DURATION), False)
    envelope = np.sin(np.pi * t) * 0.8 + 0.2
    
    signal = (base_tone + overtone1 + overtone2) * envelope
    
    # Add very light noise
    return add_noise(signal, 0.02)

def create_jazz_sample():
    """Creates a synthetic jazz music sample"""
    # Jazz: more complex chord structure with swing rhythm
    base_tone = create_sine_wave(349.23, DURATION)  # F4
    chord_tone1 = create_sine_wave(440, DURATION) * 0.7  # A4
    chord_tone2 = create_sine_wave(523.25, DURATION) * 0.5  # C5
    
    # Add swing rhythm pattern
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    rhythm = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * t + np.sin(2 * np.pi * 1 * t))
    
    signal = (base_tone + chord_tone1 + chord_tone2) * rhythm
    
    # Add moderate noise for brass-like quality
    return add_noise(signal, 0.15)

def create_rock_sample():
    """Creates a synthetic rock music sample"""
    # Rock: distorted wave with strong rhythm
    base_tone = create_sine_wave(196, DURATION)  # G3
    
    # Create distortion by clipping
    distortion = np.clip(create_sine_wave(196, DURATION) * 2.5, -0.8, 0.8)
    
    # Add drum-like beats
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    beats = np.sin(2 * np.pi * 2 * t) > 0.7
    beat_signal = beats.astype(float) * 0.5
    
    signal = (base_tone * 0.3 + distortion * 0.7 + beat_signal)
    
    # Add noise for grit
    return add_noise(signal, 0.2)

def create_electronic_sample():
    """Creates a synthetic electronic music sample"""
    # Electronic: square waves, synthesizer sounds
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Create square wave
    square = np.sign(np.sin(2 * np.pi * 220 * t))
    
    # Add a sweeping filter effect
    filter_mod = np.sin(2 * np.pi * 0.2 * t)
    sweep = create_sine_wave(110 + 50 * filter_mod, DURATION)
    
    # Add a steady beat
    beat = ((np.sin(2 * np.pi * 4 * t) > 0.9) * 0.8).astype(float)
    
    signal = square * 0.3 + sweep * 0.3 + beat * 0.4
    
    # Add a bit of noise
    return add_noise(signal, 0.05)

def create_pop_sample():
    """Creates a synthetic pop music sample"""
    # Pop: clean tones with regular beat pattern
    base_tone = create_sine_wave(261.63, DURATION)  # C4
    chord_tone = create_sine_wave(329.63, DURATION) * 0.5  # E4
    
    # Create regular beat pattern
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    beat_pattern = np.sin(2 * np.pi * 2 * t) > 0.7
    beat_signal = beat_pattern.astype(float) * 0.4
    
    # Create simple melody pattern
    melody = np.sin(2 * np.pi * (261.63 + 30 * np.sin(2 * np.pi * 0.5 * t)) * t) * 0.5
    
    signal = base_tone * 0.3 + chord_tone * 0.2 + beat_signal + melody * 0.3
    
    # Add moderate noise
    return add_noise(signal, 0.1)

def create_blues_sample():
    """Creates a synthetic blues music sample"""
    # Blues: bent notes, blue notes, slower rhythm
    base_tone = create_sine_wave(196, DURATION)  # G3
    
    # Add blue note bends
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Frequency modulation to simulate note bending
    freq_mod = 196 + 15 * np.sin(2 * np.pi * 0.5 * t)
    bent_note = np.sin(2 * np.pi * freq_mod * t)
    
    # Slow rhythm pattern
    rhythm = 0.8 + 0.2 * np.sin(2 * np.pi * 1 * t)
    
    signal = (base_tone * 0.4 + bent_note * 0.6) * rhythm
    
    # Add noise for "gritty" feeling
    return add_noise(signal, 0.18)

def create_country_sample():
    """Creates a synthetic country music sample"""
    # Country: twangy guitar sounds with steady rhythm
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Base guitar tone
    base_tone = create_sine_wave(196, DURATION)  # G3
    
    # Add twangy-ness (rapid decay on harmonics)
    decay = np.exp(-5 * t)
    twang = create_sine_wave(392, DURATION) * decay * 0.6
    
    # Add steady beat pattern (similar to 4/4 time signature)
    beat_pattern = np.sin(2 * np.pi * 1.5 * t) > 0.7
    beat_signal = beat_pattern.astype(float) * 0.3
    
    signal = base_tone * 0.4 + twang + beat_signal
    
    # Add moderate noise for authenticity
    return add_noise(signal, 0.15)

def create_metal_sample():
    """Creates a synthetic metal music sample"""
    # Metal: heavily distorted with fast rhythm
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Base low note (power chord like)
    base_tone = create_sine_wave(82.41, DURATION)  # E2
    power_chord = create_sine_wave(123.47, DURATION) * 0.7  # B2
    
    # Heavy distortion
    distortion = np.clip((base_tone + power_chord) * 5.0, -0.8, 0.8)
    
    # Fast drum beats (double bass)
    beats = np.sin(2 * np.pi * 5 * t) > 0.5  # Fast tempo
    beat_signal = beats.astype(float) * 0.8
    
    signal = distortion * 0.7 + beat_signal * 0.3
    
    # Add significant noise for extra grit
    return add_noise(signal, 0.3)

def create_hiphop_sample():
    """Creates a synthetic hip-hop music sample"""
    # Hip-hop: strong bass and beats
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Heavy bass line
    bass = create_sine_wave(60, DURATION)  # Deep bass
    
    # Filtered version of bass for sub-bass effect
    sub_bass = np.sin(2 * np.pi * 40 * t) * np.exp(-0.5 * t) * 0.8
    
    # Beat pattern with emphasis on 1 and 3
    beat_pattern1 = ((np.sin(2 * np.pi * 1 * t) > 0.8) * 0.9).astype(float)  # Main beats
    beat_pattern2 = ((np.sin(2 * np.pi * 2 * t + 0.5) > 0.9) * 0.5).astype(float)  # Off beats
    
    signal = bass * 0.5 + sub_bass * 0.3 + beat_pattern1 + beat_pattern2 * 0.5
    
    # Add some noise for vinyl-like effect
    return add_noise(signal, 0.1)

def create_reggae_sample():
    """Creates a synthetic reggae music sample"""
    # Reggae: offbeat rhythms, laid back tempo
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Base tone
    base_tone = create_sine_wave(196, DURATION) * 0.3  # G3 (not too loud)
    
    # Characteristic offbeat "skank" pattern
    skank_pattern = ((np.sin(2 * np.pi * 1 * t + 0.5) > 0.7) * 0.8).astype(float)
    
    # Bass line (strong)
    bass_line = create_sine_wave(98, DURATION) * 0.7  # G2 (strong bass)
    
    # Combine with laid-back tempo feeling
    rhythm = 0.7 + 0.3 * np.sin(2 * np.pi * 0.6 * t)  # Slower, relaxed tempo
    
    signal = (base_tone + skank_pattern + bass_line) * rhythm
    
    # Add moderate noise
    return add_noise(signal, 0.12)

def create_folk_sample():
    """Creates a synthetic folk music sample"""
    # Folk: acoustic sounds, natural tones
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Simple acoustic guitar-like tone
    base_tone = create_sine_wave(294.66, DURATION)  # D4
    overtone = create_sine_wave(440, DURATION) * 0.4  # A4
    
    # Add gentle finger-picking pattern
    pattern = np.zeros_like(t)
    for i in range(1, 6):
        offset = (i - 1) * 0.2
        pattern += 0.2 * (t % 1.0 > offset) * (t % 1.0 < offset + 0.1)
    
    # Natural dynamics
    dynamics = 0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * t)
    
    signal = (base_tone + overtone) * pattern * dynamics
    
    # Add minimal noise for acoustic feel
    return add_noise(signal, 0.08)

def create_world_sample():
    """Creates a synthetic world music sample"""
    # World: diverse rhythms, ethnic instruments simulation
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Exotic scale tones
    tone1 = create_sine_wave(261.63, DURATION) * 0.3  # C4
    tone2 = create_sine_wave(311.13, DURATION) * 0.3  # D#4/Eb4
    tone3 = create_sine_wave(392.00, DURATION) * 0.3  # G4
    
    # Complex polyrhythm
    rhythm1 = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    rhythm2 = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    
    # Percussion patterns
    perc = ((np.sin(2 * np.pi * 8 * t) > 0.7) * 0.6).astype(float)
    
    signal = tone1 * rhythm1 + tone2 * rhythm2 + tone3 + perc
    
    # Add moderate noise for texture
    return add_noise(signal, 0.15)

def create_sample_for_genre(genre):
    """Creates a synthetic audio sample for a specific genre"""
    if genre == 'classical':
        return create_classical_sample()
    elif genre == 'jazz':
        return create_jazz_sample()
    elif genre == 'rock':
        return create_rock_sample()
    elif genre == 'electronic':
        return create_electronic_sample()
    elif genre == 'pop':
        return create_pop_sample()
    elif genre == 'blues':
        return create_blues_sample()
    elif genre == 'country':
        return create_country_sample()
    elif genre == 'metal':
        return create_metal_sample()
    elif genre == 'hiphop':
        return create_hiphop_sample()
    elif genre == 'reggae':
        return create_reggae_sample()
    elif genre == 'folk':
        return create_folk_sample()
    elif genre == 'world':
        return create_world_sample()
    else:
        raise ValueError(f"Unknown genre: {genre}")

def create_dummy_data(num_samples=50, genres=None):
    """Creates synthetic audio samples for training"""
    # Define genres if not specified
    if genres is None:
        genres = ['classical', 'jazz', 'rock', 'electronic', 'pop', 'blues', 
                  'country', 'metal', 'hiphop', 'reggae', 'folk', 'world']
    
    # Create samples directory
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create synthetic samples for each genre
    for genre in genres:
        genre_dir = os.path.join(samples_dir, genre)
        os.makedirs(genre_dir, exist_ok=True)
        
        logger.info(f"Generating {num_samples} synthetic samples for {genre}...")
        
        for i in tqdm(range(num_samples), desc=f"Creating {genre} samples"):
            # Create a synthetic sample for this genre
            signal = create_sample_for_genre(genre)
            
            # Add some variation to each sample
            variation = np.random.uniform(0.9, 1.1)
            signal = signal * variation
            
            # Normalize the signal
            signal = signal / np.max(np.abs(signal))
            
            # Save as WAV file
            filename = os.path.join(genre_dir, f"synthetic_{genre}_{i+1:03d}.wav")
            sf.write(filename, signal, SAMPLE_RATE)
    
    logger.info(f"Created {num_samples * len(genres)} synthetic audio samples")
    return genres

if __name__ == "__main__":
    # Create synthetic data with 50 samples per genre
    create_dummy_data(50) 