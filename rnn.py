import os
import random

import numpy as np
from keras.layers import LSTM, Dense, GRU
from keras import Sequential
from keras.layers import Bidirectional

from keras_utils import sample_shape
from utils import DAN_LIST


class RnnDataGenerator:
    def __init__(self, input, seq_len=10) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input = input

    def generator(self, batch_size):
        all_files = [self.input + "/" + x for x in os.listdir(self.input)]
        while True:
            batch_features = np.zeros((batch_size, self.seq_len, 19 * 19), dtype=np.float32)
            batch_labels = np.zeros((batch_size, len(DAN_LIST)), dtype=np.float32)
            for i in range(batch_size):
                name = random.choice(all_files)
                try:
                    game = np.load(name)
                    boards = game["boards"]

                    idx = random.randint(0, len(boards) - 1 - self.seq_len)

                    batch_features[i] = self.generate_features(game, idx)
                    batch_labels[i] = self.generate_labels(game, idx)
                except BaseException as ex:
                    print(name, ex)
                    raise ex
            yield self.post_process(batch_features, batch_labels)

    def post_process(self, batch_features, batch_labels):
        return batch_features, batch_labels

    def generate_features(self, game, move_idx):
        boards = game["boards"][move_idx:move_idx + self.seq_len]
        boards = np.reshape(boards, (self.seq_len, 19 * 19))
        return boards

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
        return (self.seq_len, 19 * 19)


def rnn1(shape, optimizer="adam"):
    model = Sequential()

    model.add(Bidirectional(GRU(20, return_sequences=False), input_shape=shape))
    model.add(Dense(len(DAN_LIST), activation="softmax"))

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model


gen = RnnDataGenerator("C:\\tmp\\kgs_learn_workdir\\processed_black\\train")
data, labels = next(gen.generator(1))
print(data.shape)
print(labels.shape)
