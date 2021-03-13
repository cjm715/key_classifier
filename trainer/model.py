import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import tensorflow as tf
from data_generator import KeyDataGenerator
import random
import tensorflowjs as tfjs


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


def plot_history(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history['accuracy'], label = 'train accuracy')
    axs[0].plot(history.history['val_accuracy'], label = 'val accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Accuracy eval')

    axs[1].plot(history.history['loss'], label='train loss')
    axs[1].plot(history.history['val_loss'], label='val loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='lower right')
    axs[1].set_title('Error eval')
    plt.savefig('error.png')

def build_model(input_shape, complexity = 8):
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(
        filters=complexity, kernel_size=(5, 1), activation='elu', padding='same',
        input_shape = input_shape))
    model.add(keras.layers.Conv2D( 
        filters=complexity, kernel_size=(3, 1), activation='elu', padding='same'))
    model.add(keras.layers.Conv2D(
        filters=complexity, kernel_size=(5, 5), activation='elu', padding='same'))
    model.add(keras.layers.Conv2D(
        filters=complexity, kernel_size=(3, 3), activation='elu', padding = 'same'))
    model.add(keras.layers.MaxPool2D((2,2)))

    model.add(keras.layers.Conv2D(
        filters=2*complexity, kernel_size=(3, 3), activation='elu', padding = 'same'))
    model.add(keras.layers.Conv2D(
        filters=2*complexity, kernel_size=(3, 3), activation='elu', padding = 'same'))
    model.add(keras.layers.MaxPool2D((2,2)))

    model.add(keras.layers.Conv2D(
        filters=4*complexity, kernel_size=(3, 3), activation='elu', padding = 'same'))
    model.add(keras.layers.Conv2D(
        filters=4*complexity, kernel_size=(3, 3), activation='elu', padding = 'same'))
    model.add(keras.layers.MaxPool2D((2,2)))

    model.add(keras.layers.Conv2D(
        filters=8*complexity, kernel_size=(3, 3), activation='elu', padding = 'same'))
    model.add(keras.layers.Conv2D(
        filters=8*complexity, kernel_size=(3, 3), activation='elu', padding = 'same'))

    model.add(keras.layers.Conv2D(
        filters=24, kernel_size=(1, 1), activation='elu', padding = 'same'))
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Activation(keras.activations.softmax))

    return model

if __name__ == "__main__":

    input_shape = (168, None, 1)

    SAVED_MODEL_PATH = 'model_large.h5'
    model = build_model(input_shape, complexity=20)
    #model = keras.models.load_model(SAVED_MODEL_PATH)
    model.save(SAVED_MODEL_PATH)

    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    print(model.summary())
    meta_data_path = 'data/model_data/metadata.csv'

    df = pd.read_csv(meta_data_path)
    df_train = df[df.subset == 'train']
    dg_train = KeyDataGenerator(
        cqt_file_path_list=df_train.cqt_file_path,
        key_id_list=df_train.key_id,
        batch_size=32,
        random_key_shift=True,
        oversample=True,
        short=True)

    df_val = df[df.subset == 'val']
    dg_val = KeyDataGenerator(
        cqt_file_path_list=df_val.cqt_file_path,
        key_id_list=df_val.key_id,
        batch_size=200,
        random_key_shift=False,
        oversample=False,
        short=True)

    dg_val_2 = KeyDataGenerator(
        cqt_file_path_list=df_val.cqt_file_path,
        key_id_list=df_val.key_id,
        batch_size=1,
        random_key_shift=False,
        oversample=False,
        short=False)


    
    print(f"val length: {len(df_val)}")
    best_metric = 0
    for i in range(100):
        y_val = []
        y_val_pred = []
        for j in range(len(df_val)):
            X_val0, y_val0 = dg_val_2[j]
            y_val_pred0 = np.argmax(model.predict(X_val0),axis=1)

            y_val.append(y_val0)
            y_val_pred.append(y_val_pred0)

        dg_val_2.on_epoch_end()
        conf_mat = sklearn.metrics.confusion_matrix(y_val, y_val_pred, labels=list(range(24)))
        print(conf_mat)
        # model.save(SAVED_MODEL_PATH)
        val_acc = sklearn.metrics.accuracy_score(y_val, y_val_pred)

        recall_list = []
        for k in range(24):
            if (sum(conf_mat[k,:]) == 0):
                recall = 0
            else:
                recall = conf_mat[k,k]/sum(conf_mat[k,:])
            key = KEY_ID_TO_KEY_SYMBOL_MAP[k]
            print(f'key: {key} recall: {recall}')
            recall_list.append(recall)

        metric = np.mean(recall_list)
        if metric > best_metric:
            best_metric = metric
            model.save(SAVED_MODEL_PATH)

        print(f'val accuracy: {val_acc}')
        print(f'mean recall : {metric}, best mean recall: {best_metric}')

        model = keras.models.load_model(SAVED_MODEL_PATH)
        optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

        #class_weight = {m : 0.5 + 1 - recall_list[m] for m in range(24)}
        history = model.fit_generator(
            generator = dg_train, 
            validation_data= dg_val,
            epochs=4)


    # print(input_shape)
    # print(model.summary())
    # plot_history(history)
