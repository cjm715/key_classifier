import numpy as np
import pandas as pd
from music21 import *
from midi2audio import FluidSynth
import IPython.display as ipd
import librosa
import math
import os
import json
import re
from glob import glob
import pandas as pd
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import random
import copy

DATA_PATH = 'data/'
NUM_PROCESSES = 40

LMD_MIDI_DATASET = DATA_PATH + 'lmd_full'

# output 
MODEL_DATA_PATH = DATA_PATH + 'model_data_midi/'
MIDI_SCHEDULER = MODEL_DATA_PATH + 'midi_scheduler.csv'
LOG_DIR = MODEL_DATA_PATH + 'logs/'

if not os.path.exists(MODEL_DATA_PATH):
    os.makedirs(MODEL_DATA_PATH)

# removes all model data
shutil.rmtree(MODEL_DATA_PATH)

# creates folder structure
data_source_list = ['lmd_midi']
subset_list = ['train']
folders_to_create = [
    MODEL_DATA_PATH + f'{data_source}_{subset}'
    for data_source in data_source_list
    for subset in subset_list]
for folder in folders_to_create:
    os.makedirs(folder)
os.makedirs(LOG_DIR)

# create midi dataframe with midi file path and file_id
midi_data = {'midi_fp': []}
for x in os.walk(LMD_MIDI_DATASET):
    for y in glob(os.path.join(x[0], '*.mid')):
        midi_data['midi_fp'].append(y)
midi_df = pd.DataFrame(midi_data)
midi_df.reset_index(drop=True)

num_midi_files_per_process = len(midi_df) // NUM_PROCESSES

for process_id in range(NUM_PROCESSES):
    start_idx = process_id*num_midi_files_per_process

    if process_id == (NUM_PROCESSES - 1):
        end_idx = len(midi_df)
    else:
        end_idx = (process_id + 1)*num_midi_files_per_process

    midi_df.loc[start_idx:end_idx, 'process_id'] = process_id

midi_df.to_csv(MIDI_SCHEDULER, index=False)


for process_id in range(NUM_PROCESSES):
    os.system(f'python midi_transform.py {process_id} > {LOG_DIR}log_p{process_id}.txt 2>&1 &')