import os
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
import random 
from sklearn.utils import shuffle

# key is given in comments, value is tuple: (position ccw from A major, mode)
POS_ON_CIRCLE_OF_FIFTHS = {
    0:  (0, 1),  # A
    1:  (5, 1),  # Bb
    2:  (10, 1), # B
    3:  (3, 1),  # C 
    4:  (8, 1),  # Db
    5:  (1, 1),  # D
    6:  (6, 1),  # Eb
    7:  (11, 1), # E
    8:  (4, 1),  # F
    9:  (9, 1),  # F#
    10: (2, 1),  # G
    11: (7, 1),  # Ab
    12: (3, 0),  # Am
    13: (8, 0),  # Bbm
    14: (1, 0), # Bm
    15: (6, 0),  # Cm
    16: (11, 0),  # C#m
    17: (4, 0),  # Dm
    18: (9, 0),  # Ebm
    19: (2, 0), # Em
    20: (7, 0),  # Fm
    21: (0, 0),  # F#m
    22: (5, 0),  # Gm
    23: (10, 0)   # Abm
}


KEY_SYMBOL_TO_KEY_ID_MAP = {
    'A'   : 0,
    'Bb'  : 1,
    'B'   : 2,
    'C'   : 3,
    'C#'  : 4,
    'Db'  : 4, # duplicate key_id: Db == C#
    'D'   : 5,  
    'Eb'  : 6,
    'E'   : 7,
    'F'   : 8,
    'F#'  : 9,
    'Gb'  : 9,
    'G'   : 10,
    'G#'  : 11,
    'Ab'  : 11, # duplicate key_id: Ab == G#
    'Am'  : 12,
    'Bbm' : 13,
    'Bm'  : 14,
    'Cm'  : 15,
    'C#m' : 16,
    'Dbm' : 16, # duplicate key_id: Db == C#
    'Dm'  : 17,
    'Ebm' : 18,
    'Em'  : 19,
    'Fm'  : 20,
    'F#m' : 21,
    'Gbm' : 21,
    'Gm'  : 22,
    'G#m' : 23,
    'Abm' : 23 # duplicate key_id: Ab == G#
}

# ccw from A
CIRCLE_OF_FIFTHS_MAJOR = ['A' , 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'B',  'E']
CIRCLE_OF_FIFTHS_MINOR = ['Gb', 'B', 'E', 'A', 'D', 'G',   'C',  'F', 'Bb', 'Eb', 'Ab', 'Db']
def encode(y):
    pos_on_circle_of_5ths_relative_to_A_ccw, mode = POS_ON_CIRCLE_OF_FIFTHS[y]
    theta = pos_on_circle_of_5ths_relative_to_A_ccw * 2*np.pi/12
    return np.cos(theta), np.sin(theta), mode

def decode(z1, z2, mode):
    angles = [2*np.pi/12*i for i in range(12)]
    clock_pos = np.argmin([(z1 - np.cos(angle))**2 + (z2 - np.sin(angle)) **2 
        for angle in angles])
    if mode == 1:
        return KEY_SYMBOL_TO_KEY_ID_MAP[CIRCLE_OF_FIFTHS_MAJOR[clock_pos]]
    else:
        return KEY_SYMBOL_TO_KEY_ID_MAP[CIRCLE_OF_FIFTHS_MINOR[clock_pos] +'m']


class KeyDataGenerator(keras.utils.Sequence):
    def __init__(
        self, 
        cqt_file_path_list, 
        key_id_list, 
        batch_size = 32,
        random_key_shift = True, 
        oversample = True,
        short = True):

        self.data_orig = pd.DataFrame({
            'cqt_file_path' : cqt_file_path_list,
            'key_id' : key_id_list})

        self.short = short
        self.batch_size = batch_size
        self.oversample = oversample
        self.random_key_shift = random_key_shift
        self.num_classes = 24
        self.num_channels = 1
        self.bins_per_semitone = 2
        self.octaves = 7
        self.notes_per_octave = 12
        self.num_time_steps = 70 # since ~0.2 secs per time step to get 14 sec clip.
        self.num_freq_bins = self.octaves * self.notes_per_octave * self.bins_per_semitone
        self.dim = (self.num_freq_bins, self.num_time_steps)

        self.on_epoch_end()

    def __len__(self):
        return math.floor(len(self.data)/self.batch_size)

    def __getitem__(self, index):
        batch = self.data.iloc[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        y = []
        for i in range(len(batch)):
            cqt_file_path = batch.iloc[i]['cqt_file_path']
            Xsample = np.load(cqt_file_path)
            ysample = batch.iloc[i]['key_id']
            aug_res = self._augmentation(Xsample, ysample)
            if aug_res is None:
                continue
            else:
                X.append(aug_res[0])
                y.append(aug_res[1])

        X = np.stack(X, axis=0)
        y = np.array(y)
        return X, y

    def on_epoch_end(self):
        # # oversample smaller set.

        if self.oversample:
            major = self.data_orig[self.data_orig.key_id <= 11]
            minor = self.data_orig[self.data_orig.key_id > 11]

            if len(minor) < len(major):
                smaller = minor
                larger = major
            else:
                smaller = major
                larger = minor

            smaller = shuffle(smaller)
            difference = len(larger) - len(smaller)
            whole_multiplier = difference // len(smaller)
            #print(whole_multiplier)
            remaining_rows = difference % len(smaller)
            smaller_new = smaller.copy()
            for _ in range(whole_multiplier):
                smaller_new = pd.concat([smaller_new, smaller])
            smaller_new = pd.concat([smaller_new, smaller.iloc[:remaining_rows]])
            
            self.data = pd.concat([larger, smaller_new])
            self.data = shuffle(self.data)

        else:
            self.data = shuffle(self.data_orig)
    
    def _augmentation(self, X, y):
        # key shift augmentation
        if self.random_key_shift:
            key_shift = random.randint(-4,7) # end points are inclusive
            start_freq_idx = 8 + self.bins_per_semitone*key_shift # centered at note E1
            end_freq_idx = start_freq_idx + self.num_freq_bins
            if y <= 11: # major keys
                y = (y - key_shift) % 12 # keeps y values between 0-11
            else: # minor keys
                y = (y - 12 - key_shift) % 12 + 12 # keeps y value between 12-23
        else:
            start_freq_idx = 8 # note E1
            end_freq_idx = start_freq_idx + self.num_freq_bins

        # crop window in time
        orig_num_time_steps = X.shape[1]
        max_start_time_idx = orig_num_time_steps - self.num_time_steps
        if max_start_time_idx < 0:
            return None
        start_time_idx = random.randint(0, max_start_time_idx)
        end_time_idx = start_time_idx + self.num_time_steps

        # crop in both dimensions and singleton dimension for channel
        if self.short:
            X = X[start_freq_idx:end_freq_idx, start_time_idx:end_time_idx, np.newaxis]
        else:
            X = X[start_freq_idx:end_freq_idx, :, np.newaxis]

        return X, y



def load_val_data(
    cqt_file_path_list, 
    key_id_list, 
    num_freq_bins=168,
    num_time_steps=70):

    cqt_file_path_list = list(cqt_file_path_list)
    X = []
    for i in range(len(cqt_file_path_list)):
        cqt_file_path = cqt_file_path_list[i]
        Xsample = np.load(cqt_file_path)
        
        start_freq_idx = 8 # note E1
        end_freq_idx = start_freq_idx + num_freq_bins

        # crop window in time
        orig_num_time_steps = Xsample.shape[1]
        max_start_time_idx = orig_num_time_steps - num_time_steps
        if max_start_time_idx < 0:
            continue
        start_time_idx = random.randint(0, max_start_time_idx)
        end_time_idx = start_time_idx + num_time_steps

        # crop in both dimensions and singleton dimension for channel
        X.append(Xsample[
            start_freq_idx:end_freq_idx,
            start_time_idx:end_time_idx, 
            np.newaxis])

    
    X = np.stack(X, axis=0)
    y = np.array(key_id_list)

    return X, y

if __name__ == "__main__":
    for y in range(24):
        z1, z2, mode = encode(y)
        print(y, encode(y),  decode(z1, z2, mode))