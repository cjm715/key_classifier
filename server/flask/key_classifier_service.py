import numpy as np
import flasgger
from flasgger import Swagger
import random
import os
import tensorflow.keras as keras
import librosa
import tensorflow as tf


PARENT_FOLDER = ''
MODEL_PATH = PARENT_FOLDER + 'static/model.h5'
tf.config.set_visible_devices([], 'GPU')

class KeyClassifierService:
    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH)
        self.key_id_to_key_symbol = {
            0:  'A',
            1:  'Bb',
            2:  'B',
            3:  'C',
            4:  'C#',
            5:  'D',
            6:  'Eb',
            7:  'E',
            8:  'F',
            9:  'F#',
            10: 'G',
            11: 'G#',
            12: 'Am',
            13: 'Bbm',
            14: 'Bm',
            15: 'Cm',
            16: 'C#m',
            17: 'Dm',
            18: 'Ebm',
            19: 'Em',
            20: 'Fm',
            21: 'F#m',
            22: 'Gm',
            23: 'G#m'}

    def predict(self, file_path):
        X = self._get_cqt(file_path)
        if X is None:
            return None

        predicted_probs = self.model.predict(X)

        predicted_idx = np.argmax(predicted_probs)
        predicted_key = self.key_id_to_key_symbol[predicted_idx]

        probs = {self.key_id_to_key_symbol[i]: str(predicted_probs[0, i]) for i in range(24)}

        return probs, predicted_key

    def _get_cqt(
        self,
        audio_file_path,
        sample_rate=22050,
        hop_length=8192 // 2,
        bins_per_semitone=2,
        octaves=7,
        time_steps=70):
        
        signal, _ = librosa.load(audio_file_path, sr=sample_rate)

        bins_per_octave = bins_per_semitone * 12 # 12 notes in octave
        cqt = np.abs(librosa.cqt(
            signal, 
            sr=sample_rate, 
            hop_length=hop_length,
            fmin=librosa.note_to_hz('E1'),
            n_bins=bins_per_octave * octaves,
            bins_per_octave=bins_per_octave))

        X = cqt[np.newaxis, :, :, np.newaxis]
        return X
