import os
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
import random 
from sklearn.utils import shuffle


class KeyDataGenerator(keras.utils.Sequence):
    def __init__(
        self, 
        cqt_file_path_list, 
        key_id_list, 
        batch_size = 32,
        random_key_shift = True, 
        oversample = True,
        short = True,
        oversample_fraction = 1.0):

        self.data_orig = pd.DataFrame({
            'cqt_file_path' : cqt_file_path_list,
            'key_id' : key_id_list})

        self.short = short
        self.batch_size = batch_size
        self.oversample = oversample
        self.oversample_fraction = oversample_fraction
        self.random_key_shift = random_key_shift
        self.num_classes = 24
        self.num_channels = 1
        self.bins_per_semitone = 2
        self.octaves = 7
        self.notes_per_octave = 12
        self.num_time_steps = 90 # ~0.2 secs per time step 
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
            full_difference = len(larger) - len(smaller)
            difference = int(self.oversample_fraction * full_difference)
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

        # X = (X - np.mean(X))
        # if (np.std(X) > 0.0001):
        #     X = X/np.std(X)

        return X, y
