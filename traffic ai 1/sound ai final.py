# Change the sample_file.wav to a file from the dataset.

import librosa
import numpy as np
from joblib import load

def process_wav_file(filename):
    try:
        y, sr = librosa.load(filename, sr=44100)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    return mfccs

loaded_model = load("sound_ai_model.joblib")

mfccs = process_wav_file("F:/project/projects/road ai/traffic ai 1/sample1.wav")

if mfccs is not None:
    prediction = loaded_model.predict([mfccs])[0]
    print("Prediction: ", prediction)
else:
    print("Failed to process the audio file.")
