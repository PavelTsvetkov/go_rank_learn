import random

import keras
import numpy as np
from keras.engine.saving import load_model

from constants import VAE_ENCODER_LAYER
from keras_utils import DefaultDataGenerator, sample_shape
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Activation, Dropout, Conv2DTranspose, \
    MaxPooling2D, UpSampling2D
from keras import backend as K, Sequential, Input, Model

from utils import DAN_LIST


class VaeGenerator(DefaultDataGenerator):

    def __init__(self, input, layers=2, isVAE=True) -> None:
        super().__init__(input, 1)
        self.isVAE = isVAE

    def get_smp(self, brd, move):
        rot = random.randint(0, 3)
        if rot > 0:
            brd = np.rot90(brd, k=rot)
            move = np.rot90(move, k=rot)
        brd = np.pad(brd, pad_width=((9, 10), (9, 10)), mode="constant", constant_values=0)
        if K.image_data_format() == 'channels_last':
            return brd[:, :, np.newaxis]
        else:
            return brd[np.newaxis, :, :]

    def post_process(self, batch_features, batch_labels):
        if self.isVAE:
            return batch_features, batch_features
        else:
            return batch_features, batch_labels

    def out_shape(self):
        return sample_shape(1, 38)


def vae_dnn(shape, optimizer="adam"):
    conv_activation = "relu"

    convs = [16]
    encoder_size = 256

    model = Sequential()

    for i, conv_size in enumerate(convs):
        if i == 0:
            model.add(Conv2D(conv_size, kernel_size=3, activation=conv_activation, input_shape=shape))
        else:
            model.add(Conv2D(conv_size, kernel_size=3, activation=conv_activation))
        model.add(MaxPooling2D())

    # -------------------------------------------------------------------------------------
    model.add(Conv2D(encoder_size, kernel_size=3, activation=conv_activation, name=VAE_ENCODER_LAYER))
    # -------------------------------------------------------------------------------------

    for conv_size in reversed(convs):
        model.add(Conv2DTranspose(conv_size, kernel_size=3, activation=conv_activation))
        model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(1, kernel_size=3, activation="tanh"))

    model.compile(optimizer, loss="mean_squared_error", metrics=["accuracy"])
    model.summary()

    return model


def vae_classifier(vae_model: keras.Model):
    for layer in vae_model.layers:
        layer.trainable = False
    encoded = vae_model.get_layer(name=VAE_ENCODER_LAYER).output
    x = Flatten()(encoded)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dense(len(DAN_LIST))(x)
    x = Activation("softmax")(x)
    model = Model(vae_model.input, x)
    model.compile("adam", loss="mean_squared_error", metrics=["accuracy"])
    model.summary()

    return model
