import pandas as pd
import numpy as np
import os

METADATA_FP = 'data/model_metadata.csv'
MIDI_FOLDER = "data/model_data_midi"
AUDIO_METADATA_FP = 'data/model_data/metadata.csv'

df = pd.read_csv(AUDIO_METADATA_FP)

for file in os.listdir(MIDI_FOLDER):
    if file.startswith('metafile') and file.endswith(".csv"):
        midi_metadata_fp = os.path.join(MIDI_FOLDER, file)
        print(midi_metadata_fp)
        midi_df = pd.read_csv(midi_metadata_fp)
        df = pd.concat([df, midi_df])

print(df)

print(f'''
length : {len(df)}
number in train set : {sum(df.subset == 'train')}
number in val set : {sum(df.subset == 'val')}
number in test set : {sum(df.subset == 'test')}
number in holdout set : {sum(df.subset == 'holdout')}
number of major keys : {sum(df.key_id <= 11)}
number of minor keys : {sum(df.key_id > 11)}
''')
df.to_csv(METADATA_FP, index=False)