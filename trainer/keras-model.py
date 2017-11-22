
# coding: utf-8

# In[1]:


import os
import numpy as np
from sklearn.utils import shuffle

# In[2]:


# Se definen los diccionarios de atributos que predeciran cada uno de los modelos que se implementaran

training_specialists_settings = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
]


# In[3]:


# Definimos metodo para cargar los archivos de entrenamiento para cada modelo a ser entrenado
from tensorflow.python.lib.io import file_io
import pickle

def loadTrainFile(cols=None):
    # df = read_csv(os.path.expanduser(train_file))
    # Es un solo campo separado por espacios hay que mostrarlo como array
    # df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    # df.to_pickle('training.pickle')
    file_stream = file_io.FileIO(train_file, mode='r')
    df = pickle.load(file_stream)
    # Filtramos los atributos especificos al modelo a ser entrenado
    if cols:
        df = df[list(cols) + ['Image']]

    # Eliminamos las entradas que no tienen todos los atributos especificados
    df = df.dropna()

    # Escalar los pixeles entre 0 y 1
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    y = df[df.columns[:-1]].values
    y = (y - 48) / 48
    X, y = shuffle(X, y, random_state=42)  # ordenar aleatoriamente la data de entrenamiento
    y = y.astype(np.float32)

    # Escalar en 2 dimensiones, los pixeles son de 96 x 96
    X = X.reshape(-1, 96, 96, 1)
    return X, y


# In[4]:


# El tipo de red neuronal a implementar es LeNet-5  (http://yann.lecun.com/exdb/lenet/)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

def buildModel():

    model = Sequential()

    # Se agregan 3 capas convolucionales
    model.add(Convolution2D(32,(3,3), input_shape=(96, 96, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))


    model.add(Convolution2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Y dos capas densas
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(30))

    return model


# In[63]:


# Definimos metodo que usaremos para data augmentation, mediante el metodo de mirror image

from keras.preprocessing.image import ImageDataGenerator
class FlippedImageDataGenerator(ImageDataGenerator):
    flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9),
                    (6, 10), (7, 11), (12, 16), (13, 17),
                    (14, 18), (15, 19), (22, 24), (23, 25)]

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)
        X_batch[indices] = X_batch[indices, :, :, ::-1]

        if y_batch is not None:
            y_batch[indices, ::2] = y_batch[indices, ::2] * -1

            for a, b in self.flip_indices:
                y_batch[indices, a], y_batch[indices, b] = (
                    y_batch[indices, b], y_batch[indices, a]
                )
        print(X_batch.shape)
        return X_batch, y_batch


# In[73]:


# Se define metodo que entrena los modelos especializados o training specialists

from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.models import model_from_json
from collections import OrderedDict

def normalFit():
    start = 0.03
    stop = 0.001
    nb_epoch = 400
    learning_rate = np.linspace(start, stop, nb_epoch)

    X, y = loadTrainFile()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
    model = model_from_json(net.to_json())
    model.compile(loss='mean_squared_error', optimizer='adam')

    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(patience=100)
    flipGen = FlippedImageDataGenerator()

    model.fit_generator(flipGen.flow(X_train, y_train),
                                steps_per_epoch=X_train.shape[0],
                                epochs=nb_epoch,
                                validation_data=(X_test, y_test),
                                callbacks=[change_lr, early_stop])


def fitTrainingSpecialists():
    train_specialists = OrderedDict()
    start = 0.03
    stop = 0.001
    nb_epoch = 100
    learning_rate = np.linspace(start, stop, nb_epoch)

    for setting in training_specialists_settings:

        # Se extraen las columnas especificas para cada modelo y se divide la data en set de entrenamiento
        # y set de prueba
        train_columns = setting['columns']
        X, y = loadTrainFile(cols=train_columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Se utiliza el modelo definido previamente
        train_specialist = model_from_json(net.to_json())

        train_specialist.layers.pop()
        train_specialist.outputs = [train_specialist.layers[-1].output]
        train_specialist.layers[-1].outbound_nodes = []
        train_specialist.add(Dense(len(train_columns)))

        train_specialist.compile(loss='mean_squared_error', optimizer='adam')
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(patience=100)
        flipGen = FlippedImageDataGenerator()
        flipGen.flip_indices = setting['flip_indices']

        print("Entrenando modelo para las columnas {} por {} epochs".format(train_columns, nb_epoch))

        train_specialist.fit_generator(flipGen.flow(X_train, y_train),
                                steps_per_epoch=X_train.shape[0],
                                epochs=nb_epoch,
                                validation_data=(X_test, y_test),
                                callbacks=[change_lr, early_stop])

        train_specialists[train_columns] = train_specialist

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    train_file = arguments['train_file']
    job_dir = arguments.pop('job_dir')
    net = buildModel()
    fitTrainingSpecialists()
