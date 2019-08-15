import json
import os
import random
import zipfile

from keras import backend as K, Sequential

import numpy as np
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D

from constants import META_KEY, HISTORY_KEY
from utils import coord, DAN_LIST, DAN_MAP, encode_board

DAN_SET = set(DAN_LIST)


def sample_shape():
    if K.image_data_format() == 'channels_last':
        shape = (19, 19, 2)
    else:
        shape = (2, 19, 19)
    return shape


def gen_sample(file_map, folder):
    dan_idx = random.randint(0, len(DAN_LIST) - 1)
    dan = DAN_LIST[dan_idx]
    files = file_map[dan]
    file = random.choices(files)[0]
    data = np.load("/".join([folder, dan, file]))
    board = data["board"]
    move = data["move"]
    if K.image_data_format() == 'channels_last':
        axis = 2
    else:
        axis = 0
    features = np.stack((board, move), axis=axis)
    labels = np.zeros((len(DAN_LIST)))
    labels[dan_idx] = 1.0
    return features, labels


def get_smp(brd, move):
    brd = np.zeros((19, 19), dtype=np.float32)
    move = np.zeros((19, 19), dtype=np.float32)

    rot = random.randint(0, 3)
    if rot > 0:
        brd = np.rot90(brd, k=rot)
        move = np.rot90(move, k=rot)
    if K.image_data_format() == 'channels_last':
        axis = 2
    else:
        axis = 0
    features = np.stack((brd, move), axis=axis)
    return features


def data_gen(zip_name, batch_size):
    smp_shape = sample_shape()
    all_files = [zip_name + "/" + x for x in os.listdir(zip_name)]
    while True:
        batch_features = np.zeros((batch_size, smp_shape[0], smp_shape[1], smp_shape[2]), dtype=np.float32)
        batch_labels = np.zeros((batch_size, len(DAN_LIST)), dtype=np.float32)
        for i in range(batch_size):
            name = random.choice(all_files)
            try:
                game = np.load(name)

                boards = game["boards"]
                moves = game["moves"]
                labels = game["labels"]

                idx = random.randint(0, len(boards) - 1)

                features = get_smp(boards[idx], moves[idx])
                batch_features[i] = features
                batch_labels[i] = labels[idx]
            except BaseException as ex:
                print(name, ex)
                raise ex
        yield batch_features, batch_labels


def keras_model1():
    conv_activation = "sigmoid"

    shape = sample_shape()
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation=conv_activation, input_shape=shape))
    model.add(MaxPool2D())
    model.add(Conv2D(64, kernel_size=3, activation=conv_activation))
    model.add(MaxPool2D())
    # model.add(Conv2D(128, kernel_size=3, activation="relu"))
    # model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(DAN_LIST), activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model
