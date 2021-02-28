import librosa
import math
import os
import json
import re
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
import shutil

DATA_PATH = 'data/'
SAMPLE_RATE = 22050
WINDOW_LENGTH = 8192
HOP_LENGTH = WINDOW_LENGTH // 2
OCTAVES = 8
BINS_PER_SEMITONE = 2
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
MIN_TIME_STEPS = 125

# inputs
GTZAN_KEY_DATASET = DATA_PATH + 'gtzan_key/gtzan_key/genres/'
GTZAN_AUDIO_DATASET = DATA_PATH + 'GTZAN/genres_original/'

LMD_KEY_TRAIN_DATASET = DATA_PATH + 'keys/lmd_key_train.tsv'
LMD_KEY_VAL_DATASET = DATA_PATH + 'keys/lmd_key_valid.tsv'
LMD_KEY_HOLDOUT_DATASET = DATA_PATH + 'keys/lmd_key_test.tsv'
LMD_AUDIO_DATASET = DATA_PATH + 'lmd_matched_mp3/'

# output 
MODEL_DATA_PATH = DATA_PATH + 'model_data/'
METADATA_FILE = MODEL_DATA_PATH + 'metadata.csv'

# removes all model data
shutil.rmtree(MODEL_DATA_PATH)

# creates folder structure
data_source_list = ['gtzan', 'lmd']
subset_list = ['train', 'val', 'holdout', 'test']
folders_to_create = [
    MODEL_DATA_PATH + f'{data_source}_{subset}'
    for data_source in data_source_list
    for subset in subset_list]
for folder in folders_to_create:
    os.makedirs(folder)

KEY_ID_TO_KEY_SYMBOL_MAP = {
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
    23: 'G#m'
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
    'Gm'  : 22,
    'G#m' : 23,
    'Abm' : 23 # duplicate key_id: Ab == G#
}

def get_cqt(audio_file_path):
    try:
        signal, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE)
    except:
        print(f"WARNING: {audio_file_path} is corrupt or not found")
        return None
    cqt = np.abs(librosa.cqt(
        signal, 
        sr=SAMPLE_RATE, 
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C1'),
        n_bins=BINS_PER_OCTAVE * OCTAVES,
        bins_per_octave=BINS_PER_OCTAVE))
    return cqt


def get_subset_label_gtzan(file_id,
    max_train_file_id=45,
    max_val_file_id=55,
    max_test_file_id=60):
    file_id = int(file_id)

    if file_id <= max_train_file_id:
        return 'train'
    elif (file_id > max_train_file_id) and (file_id <= max_val_file_id):
        return 'val'
    elif (file_id > max_val_file_id) and (file_id <= max_test_file_id):
        return 'test'
    else:
        return 'holdout'

def get_key_gtzan(genre, file_id):
    key_file_path = GTZAN_KEY_DATASET + f'{genre}/{genre}.{file_id}.lerch.txt'
    try: 
        with open(key_file_path, "r") as kfp:
            first_line = kfp.readline()
            key_id = int(first_line)
            if key_id == -1:
                key_id = None
                key = None
            else:
                key = KEY_ID_TO_KEY_SYMBOL_MAP[key_id]
    except:
        key_id = None
        key = None

    return key_id, key

def save_cqt_gtzan():
    data = {
        'audio_file_path' : [],
        'cqt_file_path' : [],
        'key_id' : [],
        'key' : [],
        'subset' : [],
        'genre': []}
    file_num_counters = {
        'train': 0,
        'val': 0,
        'test': 0,
        'holdout': 0
    }
    audio_file_path_list = [ y 
        for x in os.walk(GTZAN_AUDIO_DATASET) 
        for y in glob(os.path.join(x[0], '*.wav'))]

    for audio_file_path in tqdm(audio_file_path_list):
    #for audio_file_path in tqdm(audio_file_path_list[:20]):
        # find key label and meta data
        stem = os.path.splitext(os.path.basename(audio_file_path))[0]
        genre, file_id = stem.split('.')
        key_id, key = get_key_gtzan(genre, file_id)
        if key is None:
            continue
        subset = get_subset_label_gtzan(file_id)

        # calcualte contant-q transform
        cqt = get_cqt(audio_file_path)
        if cqt is None:
            continue
        if cqt.shape[1] < MIN_TIME_STEPS:
            continue
        # save cqt to disk
        file_num = file_num_counters[subset]
        cqt_file_path = MODEL_DATA_PATH + f'gtzan_{subset}/{file_num}_{key_id}.npy'
        file_num_counters[subset] += 1
        np.save(cqt_file_path, cqt)

        print(f"\naudio: {audio_file_path} cqt: {cqt_file_path}, key_id: {key_id}, key: {key}\n")
        # save metadata
        data['audio_file_path'].append(audio_file_path)
        data['cqt_file_path'].append(cqt_file_path)
        data['key_id'].append(key_id)
        data['key'].append(key)
        data['subset'].append(subset)
        data['genre'].append(genre)

    return pd.DataFrame(data)


def get_lmd_file_id(y):
    file_id = os.path.splitext(os.path.basename(y))[0]
    return file_id

def save_cqt_lmd():
    # dictionary to store data
    subsets = [
        ('train', LMD_KEY_TRAIN_DATASET),
        ('val', LMD_KEY_VAL_DATASET),
        ('holdout', LMD_KEY_HOLDOUT_DATASET)]

    data = {
        'audio_file_path' : [],
        'cqt_file_path' : [],
        'key_id' : [],
        'key' : [],
        'subset' : []}

    lmd_audio_file_path_lookup = {
        get_lmd_file_id(y) : y 
        for x in os.walk(LMD_AUDIO_DATASET) 
        for y in glob(os.path.join(x[0], '*.mp3'))}

    for (subset, subset_path) in subsets:
        lmd_df = pd.read_csv(subset_path, sep='\t', header=None)
        lmd_df = lmd_df.iloc[:,[0,2]]
        lmd_df.columns = ['file_id', 'key']

        file_num = 0
        for i in tqdm(range(len(lmd_df))):
        #for i in tqdm(range(10)):
            row = lmd_df.iloc[i, :]
            file_id = row.loc['file_id']
            key = row.loc['key']
            key_id = KEY_SYMBOL_TO_KEY_ID_MAP[key]

            # this step is necessary to deal with duplicate names for the same key
            # this will ensure same key name across datasets.
            key = KEY_ID_TO_KEY_SYMBOL_MAP[key_id]

            audio_file_path = lmd_audio_file_path_lookup[file_id]

            cqt = get_cqt(audio_file_path)

            # calcualte contant-q transform
            cqt = get_cqt(audio_file_path)
            if cqt is None:
                continue
            if cqt.shape[1] < MIN_TIME_STEPS:
                continue

            # save cqt to disk
            cqt_file_path = MODEL_DATA_PATH + f'lmd_{subset}/{file_num}_{key_id}.npy'
            file_num += 1
            np.save(cqt_file_path, cqt)

            data['audio_file_path'].append(audio_file_path)
            data['cqt_file_path'].append(cqt_file_path)
            data['key_id'].append(key_id)
            data['key'].append(key)
            data['subset'].append(subset)

            print(f"\naudio: {audio_file_path} cqt: {cqt_file_path}, key_id: {key_id}, key: {key}\n")

    return pd.DataFrame(data)


df_gtzan = save_cqt_gtzan()
df_gtzan['source'] = 'gtzan'

df_lmd = save_cqt_lmd()
df_lmd['source'] = 'lmd'

df = pd.concat([df_gtzan,df_lmd])
df.to_csv(METADATA_FILE, index=False)