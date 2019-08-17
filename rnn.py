import os
import random

import luigi
import numpy as np
from keras import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense, GRU

from constants import MODELS_DIR
from luigi_tasks import SplitTrainValidation
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


class LearnRNN1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/rnn1.mdl")

    def requires(self):
        return SplitTrainValidation(parallelism=self.parallelism, sample_count=0, black_mode=True,
                                    sub_folder="/processed_black")

    def run(self):
        from rnn import RnnDataGenerator
        from keras.optimizers import Adam
        from rnn import rnn1
        EPOHCH_SIZE = 20000
        batch_size = 100

        VALIDATION_SIZE = 3000

        steps_per_epoch = EPOHCH_SIZE / batch_size
        validation_steps = VALIDATION_SIZE / batch_size

        train_fold = self.input()[0].path
        valid_fold = self.input()[1].path
        test_fold = self.input()[2].path

        train_gen = RnnDataGenerator(train_fold)
        val_gen = RnnDataGenerator(valid_fold)

        mdl = rnn1(train_gen.out_shape(), optimizer=Adam(lr=0.01))

        mdl.fit_generator(train_gen.generator(batch_size),
                          steps_per_epoch=steps_per_epoch,
                          epochs=7,
                          validation_data=val_gen.generator(batch_size),
                          validation_steps=validation_steps, use_multiprocessing=False, workers=1)

        mdl.save(self.output().path)
