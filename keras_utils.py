import os
import os
import random

import numpy as np
from keras import backend as K, Sequential, regularizers, Input, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Activation, Dropout, add, \
    GlobalAveragePooling2D

from utils import DAN_LIST

DAN_SET = set(DAN_LIST)


def sample_shape(layers=2, size=19):
    if K.image_data_format() == 'channels_last':
        shape = (size, size, layers)
    else:
        shape = (layers, size, size)
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


def get_smp_alpha(empty, black, white):
    rot = random.randint(0, 3)
    if rot > 0:
        empty = np.rot90(empty, k=rot)
        black = np.rot90(black, k=rot)
        white = np.rot90(white, k=rot)
    if K.image_data_format() == 'channels_last':
        axis = 2
    else:
        axis = 0
    features = np.stack((empty, black, white), axis=axis)
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


def data_gen_alpha(zip_name, batch_size):
    smp_shape = sample_shape(3)
    all_files = [zip_name + "/" + x for x in os.listdir(zip_name)]
    while True:
        batch_features = np.zeros((batch_size, smp_shape[0], smp_shape[1], smp_shape[2]), dtype=np.float32)
        batch_labels = np.zeros((batch_size, len(DAN_LIST)), dtype=np.float32)
        for i in range(batch_size):
            name = random.choice(all_files)
            try:
                game = np.load(name)
                empty_cells = game["empty_cells"]
                black_cells = game["black_cells"]
                white_cells = game["white_cells"]

                idx = random.randint(0, len(empty_cells) - 1)

                batch_features[i] = get_smp_alpha(empty_cells[idx], black_cells[idx], white_cells[idx])
                batch_labels[i] = game["black_dans"]


            except BaseException as ex:
                print(name, ex)
                raise ex

        yield batch_features, batch_labels


def keras_model0():
    conv_activation = "relu"

    shape = sample_shape(3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, activation=conv_activation, input_shape=shape))
    model.add(Conv2D(32, kernel_size=5, activation=conv_activation, input_shape=shape))

    model.add(Dense(len(DAN_LIST), activation="softmax"))
    model.compile(optimizer, loss=loss, metrics=["accuracy"])
    model.summary()

    return model


def res_block(n_output):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(x)

        # second pre-activation
        h = BatchNormalization()(h)
        h = Activation("relu")(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(h)

        # F_l(x) = f(x) + H_l(x):
        h = x = add([x, h])
        h = BatchNormalization()(h)
        h = Activation("relu")(h)
        return h

    return f


def keras_model1(optimizer="adam", loss="categorical_crossentropy"):
    conv_activation = "relu"

    shape = sample_shape(3)
    input_tensor = Input(shape)

    conv_size = 64
    x = Conv2D(conv_size, kernel_size=3, input_shape=shape,
               kernel_regularizer=regularizers.l2(0.01))(input_tensor)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = res_block(conv_size)(x)
    x = res_block(conv_size)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.2)(x)

    x = Dense(len(DAN_LIST), activation="softmax", kernel_regularizer=regularizers.l2(0.01))(x)

    model = Model(inputs=input_tensor, outputs=x)

    model.compile(optimizer, loss=loss, metrics=["accuracy"])
    model.summary()

    return model


class DefaultDataGenerator:
    def __init__(self, input, layers=2) -> None:
        super().__init__()
        self.layers = layers
        self.input = input

    def generator(self, batch_size):
        smp_shape = self.out_shape()
        all_files = [self.input + "/" + x for x in os.listdir(self.input)]
        while True:
            batch_features = np.zeros((batch_size, smp_shape[0], smp_shape[1], smp_shape[2]), dtype=np.float32)
            batch_labels = np.zeros((batch_size, len(DAN_LIST)), dtype=np.float32)
            for i in range(batch_size):
                name = random.choice(all_files)
                try:
                    game = np.load(name)
                    boards = game["boards"]

                    idx = random.randint(0, len(boards) - 1)

                    batch_features[i] = self.generate_features(game, idx)
                    batch_labels[i] = self.generate_labels(game, idx)
                except BaseException as ex:
                    print(name, ex)
                    raise ex
            yield self.post_process(batch_features, batch_labels)

    def post_process(self, batch_features, batch_labels):
        return batch_features, batch_labels

    def generate_features(self, game, move_idx):
        boards = game["boards"]
        moves = game["moves"]
        return self.get_smp(boards[move_idx], moves[move_idx])

    def generate_labels(self, game, move_idx):
        return game["labels"][move_idx]

    def get_smp(self, brd, move):
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

    def out_shape(self):
        return sample_shape(self.layers, 19)
