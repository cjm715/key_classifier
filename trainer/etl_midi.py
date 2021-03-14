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
SAMPLE_RATE = 22050
WINDOW_LENGTH = 8192
HOP_LENGTH = WINDOW_LENGTH // 2
OCTAVES = 8
BINS_PER_SEMITONE = 2
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
MIN_TIME_STEPS = 125

# inputs
SOUND_FONT_DIRECTORY = DATA_PATH + 'sound_fonts/' 

LMD_KEY_TRAIN_DATASET = DATA_PATH + 'keys/lmd_key_train.tsv'
LMD_KEY_VAL_DATASET = DATA_PATH + 'keys/lmd_key_valid.tsv'
LMD_KEY_HOLDOUT_DATASET = DATA_PATH + 'keys/lmd_key_test.tsv'
LMD_MIDI_DATASET = DATA_PATH + 'lmd_matched/'

# output 
MODEL_DATA_PATH = DATA_PATH + 'model_data_midi/'
METADATA_FILE = MODEL_DATA_PATH + 'metadata.csv'

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

KEY_ID_TO_TONIC_MODE_TUPLE = {
    0:  ('A', 'major'),
    1:  ('Bb', 'major'),
    2:  ('B', 'major'),
    3:  ('C', 'major'),
    4:  ('C#', 'major'),
    5:  ('D', 'major'),
    6:  ('Eb', 'major'),
    7:  ('E', 'major'),
    8:  ('F', 'major'),
    9:  ('F#', 'major'),
    10: ('G', 'major'),
    11: ('G#', 'major'),
    12: ('A', 'minor'),
    13: ('Bb', 'minor'),
    14: ('B', 'minor'),
    15: ('C', 'minor'),
    16: ('C#', 'minor'),
    17: ('D', 'minor'),
    18: ('Eb', 'minor'),
    19:( 'E', 'minor'),
    20: ('F', 'minor'),
    21: ('F#', 'minor'),
    22: ('G', 'minor'),
    23: ('G#', 'minor')
}

KEY_SYMBOL_TO_KEY_ID_MAP = {
    'A'   : 0,
    'A#'  : 1,
    'Bb'  : 1,
    'B'   : 2,
    'C'   : 3,
    'C#'  : 4,
    'Db'  : 4, # duplicate key_id: Db == C#
    'D'   : 5,
    'D#'  : 6,
    'Eb'  : 6,
    'E'   : 7,
    'F'   : 8,
    'F#'  : 9,
    'Gb'  : 9,
    'G'   : 10,
    'G#'  : 11,
    'Ab'  : 11, # duplicate key_id: Ab == G#
    'Am'  : 12,
    'A#m' : 13,
    'Bbm' : 13,
    'Bm'  : 14,
    'Cm'  : 15,
    'C#m' : 16,
    'Dbm' : 16, # duplicate key_id: Db == C#
    'Dm'  : 17,
    'D#m' : 18,
    'Ebm' : 18,
    'Em'  : 19,
    'Fm'  : 20,
    'F#m' : 21,
    'Gbm' : 21,
    'Gm'  : 22,
    'G#m' : 23,
    'Abm' : 23, # duplicate key_id: Ab == G#
    'A major' : 0,
    'A# major' : 1,
    'Bb major' : 1,
    'B major': 2,
    'C major' : 3,
    'C# major' : 4,
    'Db major' : 4,
    'D major' : 5,
    'D# major' : 6,
    'Eb major' : 6,
    'E major' : 7,
    'F major' : 8,
    'F# major' : 9,
    'Gb major' : 9,
    'G major' : 10,
    'G# major' : 11,
    'Ab major' : 11,
    'A minor' : 12,
    'A# minor' : 13,
    'Bb minor' : 13,
    'B minor': 14,
    'C minor' : 15,
    'C# minor' : 16,
    'Db minor' : 16,
    'D minor' : 17,
    'D# minor' : 18,
    'Eb minor' : 18,
    'E minor' : 19,
    'F minor' : 20,
    'F# minor' : 21,
    'Gb minor' : 21,
    'G minor' : 22,
    'G# minor' : 23,
    'Ab minor' : 23
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

def select_random_sound_font_file(sound_font_directory = '../data/sound_fonts/'):
    sound_font_files = os.listdir(sound_font_directory)
    full_path = sound_font_directory + random.choice(sound_font_files)
    return full_path

def convert_midi_to_wav(
    in_file,                    
    out_file ='out.wav',
    sound_font = '../data/sound_fonts/024_BigSymphny.sf2'):
    fs = FluidSynth(sound_font = sound_font)
    fs.midi_to_audio(in_file, out_file)


def switch_modes(orig, key_mode, key_tonic):
    sc_minor = scale.MinorScale(key_tonic)
    sc_major = scale.MajorScale(key_tonic)

    if key_mode == 'major':
        post_mode = 'minor'
        degree_idx_to_lower = [2, 5, 6]
        notes_to_lower = [sc_major.pitches[p].name for p in  degree_idx_to_lower]
        post = copy.deepcopy(orig)
        for elem in post.recurse().notes:
            if isinstance(elem, note.Note):
                if elem.name in notes_to_lower:
                    elem.transpose(-1, inPlace=True)
            if isinstance(elem, chord.Chord):
                for n in elem:
                    if n.name in notes_to_lower:
                        n.transpose(-1, inPlace=True)    

    elif key_mode == 'minor':
        post_mode = 'major'
        degree_idx_to_raise = [2, 5, 6]
        notes_to_raise = [sc_minor.pitches[p].name for p in  degree_idx_to_raise]
        post = copy.deepcopy(orig)
        for elem in post.recurse().notes:
            if isinstance(elem, note.Note):
                if elem.name in notes_to_raise:
                    elem.transpose(1, inPlace=True)
            if isinstance(elem, chord.Chord):
                for n in elem:
                    if n.name in notes_to_raise:
                        n.transpose(1, inPlace=True)
    else:
        print('mode is neither major nor minor')
        
    return post, post_mode

def get_lmd_file_id(y):
    file_id = os.path.splitext(os.path.basename(y))[0]
    return file_id

def get_tonic_mode(row):
    tonic, mode = KEY_ID_TO_TONIC_MODE_TUPLE[row['key_id']]
    return pd.Series([tonic, mode])

def save_cqt_lmd():
    # dictionary to store data
    subsets = [
        ('train', LMD_KEY_TRAIN_DATASET),
        ('val', LMD_KEY_VAL_DATASET),
        ('holdout', LMD_KEY_HOLDOUT_DATASET)]

    # create midi dataframe with midi file path and file_id
    midi_data = {'midi_fp': []}
    for x in os.walk(LMD_MIDI_DATASET):
        for y in glob(os.path.join(x[0], '*.mid')):
            midi_data['midi_fp'].append(y)
    midi_df = pd.DataFrame(midi_data)
    midi_df['file_id'] = midi_df['midi_fp'].map(lambda y: y.split('/')[-2])

    file_num = 0

    df = None
    for (subset, subset_path) in subsets:
        # create file_id and key table for subset.
        label_df = pd.read_csv(subset_path, sep='\t', header=None)
        label_df = label_df.iloc[:,[0,2]]
        label_df.columns = ['file_id', 'key']
        label_df['key_id'] = label_df['key'].apply(lambda k: int(KEY_SYMBOL_TO_KEY_ID_MAP[k]))
        orig_df = pd.merge(label_df, midi_df, how="outer", on ='file_id').dropna()
        orig_df[['tonic', 'mode']] = orig_df.apply(get_tonic_mode ,axis=1)
        orig_df['version'] = 'orig'
        orig_df['subset'] = subset
        
        print(subset, f'number of midi files: {len(orig_df)}')

        file_num = 0
        for i in tqdm(range(len(orig_df))):
        #for i in tqdm(range(1)):
            # extract info from row
            midi_row = orig_df.iloc[[i]].reset_index(drop=True)
            key_id = int(midi_row.loc[0,'key_id'])
            key = midi_row.loc[0,'key']
            tonic = midi_row.loc[0,'tonic']
            mode = midi_row.loc[0,'mode']
            midi_fp = midi_row.loc[0,'midi_fp']

            # convert orig midi to wav and save file
            sound_font = select_random_sound_font_file(SOUND_FONT_DIRECTORY)

            wav_fp = MODEL_DATA_PATH + f'lmd_midi_{subset}/{file_num}_{key_id}.wav'
            convert_midi_to_wav(midi_fp, wav_fp, sound_font)

            # calculate cqt and save if valid
            cqt = get_cqt(wav_fp)
            cqt_file_path = MODEL_DATA_PATH + f'lmd_midi_{subset}/{file_num}_{key_id}.npy'
            if cqt is None:
                midi_row.loc[0,'cqt_file_path'] = None
            elif cqt.shape[1] < MIN_TIME_STEPS:
                midi_row.loc[0,'cqt_file_path'] = None
            else:
                midi_row.loc[0,'cqt_file_path'] = cqt_file_path
                np.save(cqt_file_path, cqt)
                if df is None:
                    df = midi_row
                else:
                    df = pd.concat([df, midi_row])
            
            file_num += 1
            os.remove(wav_fp)
            print(midi_row)

            par_row = midi_row.copy()
            par_row['version'] = 'parallel'
            # convert orig midi to parallel mode
            orig = converter.parse(midi_fp)
            post, post_mode = switch_modes(orig, mode, tonic)
            
            # calculate key variables and save
            key = tonic + ' ' + post_mode
            key_id = KEY_SYMBOL_TO_KEY_ID_MAP[key]
            key = KEY_ID_TO_KEY_SYMBOL_MAP[key_id]

            par_row.loc[0,'key'] = key
            par_row.loc[0,'key_id'] = int(key_id)
            par_row.loc[0,'tonic'] = tonic
            par_row.loc[0,'mode'] = post_mode


            parallel_midi_fp = MODEL_DATA_PATH + f'lmd_midi_{subset}/{file_num}_{key_id}.mid'
            par_row['midi_fp'] = parallel_midi_fp          
            post.write('midi', parallel_midi_fp)

            sound_font = select_random_sound_font_file(SOUND_FONT_DIRECTORY)

            parallel_wav_fp = MODEL_DATA_PATH + f'lmd_midi_{subset}/{file_num}_{key_id}.wav'
            convert_midi_to_wav(parallel_midi_fp, parallel_wav_fp, sound_font)

            # calculate cqt and save if valid
            cqt = get_cqt(parallel_wav_fp)
            if cqt is None:
                par_row.loc[0,'cqt_file_path'] = None
            elif cqt.shape[1] < MIN_TIME_STEPS:
                par_row.loc[0,'cqt_file_path'] = None
            else:
                cqt_file_path = MODEL_DATA_PATH + f'lmd_midi_{subset}/{file_num}_{key_id}.npy'
                par_row.loc[0, 'cqt_file_path'] = cqt_file_path
                np.save(cqt_file_path, cqt)
                if df is None:
                    df = par_row
                else:
                    df = pd.concat([df, par_row])
            file_num += 1
            os.remove(parallel_wav_fp)
            print(par_row)

            #df = df.dropna()
            df['key_id'] = df['key_id'].astype(int)
            df['source'] = 'lmd_midi'
            df.to_csv(METADATA_FILE, index=False)



# removes all model data
shutil.rmtree(MODEL_DATA_PATH)

# creates folder structure
data_source_list = ['lmd_midi']
subset_list = ['train', 'val', 'holdout', 'test']
folders_to_create = [
    MODEL_DATA_PATH + f'{data_source}_{subset}'
    for data_source in data_source_list
    for subset in subset_list]
for folder in folders_to_create:
    os.makedirs(folder)


save_cqt_lmd()  