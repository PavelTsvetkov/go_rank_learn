import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import luigi
import numpy as np

from constants import WORKDIR, GAMES_MASK, MODELS_DIR, HISTORY_KEY, META_KEY
from goban import load_goban
from utils import game_to_numpy, CLASS_WEIGHTS


class ProcessDir(luigi.Task):
    parallelism = luigi.IntParameter()
    file_mod = luigi.IntParameter()

    ranks = defaultdict(int)
    rank_diff = defaultdict(int)

    def output(self):
        return luigi.LocalTarget(WORKDIR + "/game_stats" + str(self.file_mod) + ".zip")

    def requires(self):
        return None

    def run(self):
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as ztemp:
            with zipfile.ZipFile(ztemp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                files = Path(GAMES_MASK).glob("**/*.sgf")
                files = [x for x in files]
                for idx, f in enumerate(files):
                    if idx % self.parallelism == int(self.file_mod):
                        try:
                            goban, meta_inf = self.process_file(f)
                            if len(goban.history) < 2:
                                raise Exception("History to short")
                            data = {
                                HISTORY_KEY: goban.history,
                                META_KEY: meta_inf
                            }
                            z.writestr(str(idx) + ".json", json.dumps(data))
                        except BaseException as ex:
                            print(f, ":", ex)
                        sys.stdout.write("\r" + str(idx) + " out of " + str(len(files)) + "\r")
                        sys.stdout.flush()
        os.rename(ztemp.name, self.output().path)
        print(self.rank_diff)

    def process_file(self, fil):
        return load_goban(fil)


class SplitTrainValidation(luigi.Task):
    folders = ["/train", "/validation", "/test"]
    probabilities = [0.8, 0.1, 0.1]

    parallelism = luigi.IntParameter()

    sample_count = luigi.IntParameter(default=0)
    black_mode = luigi.BoolParameter(default=False)
    sub_folder = luigi.Parameter(default="/processed_games")

    def output(self):
        return [luigi.LocalTarget(WORKDIR + self.sub_folder + x) for x in
                SplitTrainValidation.folders]

    def requires(self):
        for k in range(int(self.parallelism)):
            yield ProcessDir(parallelism=self.parallelism, file_mod=k)

    def run(self):
        out_zips = [x.path + "_tmp" for x in self.output()]
        for x in out_zips:
            shutil.rmtree(x, ignore_errors=True)
            os.makedirs(x, exist_ok=True)
        try:
            for inp in self.input():
                self.process_input(inp, out_zips)
            for x in self.output():
                os.rename(x.path + "_tmp", x.path)
        except BaseException as ex:
            for x in out_zips:
                shutil.rmtree(x, ignore_errors=True)
            for x in self.output():
                shutil.rmtree(x.path, ignore_errors=True)
            raise ex

    def process_input(self, inp, out_zips):
        with zipfile.ZipFile(inp.path) as z:
            namelist = z.namelist()
            if self.sample_count > 0:
                namelist = random.sample(namelist, k=self.sample_count)
            for idx, name in enumerate(namelist):
                out = random.choices(out_zips, weights=self.probabilities)[0]
                try:
                    game = json.loads(z.read(name))
                    boards, moves, labels = game_to_numpy(game, black_mode=self.black_mode)
                    np.savez_compressed(out + "/" + name, boards=boards, moves=moves, labels=labels)
                except BaseException as ex:
                    print("Failed processing of ", name, " : ", ex)
                print(inp.path, " : ", idx + 1, " of ", len(namelist))


class LearnModel1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/model1.mdl")

    def requires(self):
        return SplitTrainValidation(parallelism=self.parallelism)

    def run(self):
        from keras_utils import keras_model1, data_gen
        from keras.optimizers import Adam
        EPOHCH_SIZE = 20000
        batch_size = 32

        VALIDATION_SIZE = 3000

        steps_per_epoch = EPOHCH_SIZE / batch_size
        validation_steps = VALIDATION_SIZE / batch_size

        mdl = keras_model1(optimizer=Adam(lr=0.01), loss="mean_squared_error")

        train_fold = self.input()[0].path
        valid_fold = self.input()[1].path
        test_fold = self.input()[2].path

        mdl.fit_generator(data_gen(train_fold, batch_size),
                          steps_per_epoch=steps_per_epoch,
                          epochs=7,
                          validation_data=data_gen(valid_fold, batch_size),
                          validation_steps=validation_steps, use_multiprocessing=False, workers=1,
                          class_weight=CLASS_WEIGHTS)

        mdl.save(self.output().path)


class VerifyModel1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/model1_check.csv")

    def requires(self):
        return [LearnModel1(parallelism=self.parallelism), SplitTrainValidation(parallelism=self.parallelism)]

    def run(self):
        from keras_utils import data_gen
        from keras.engine.saving import load_model

        mdl = load_model(self.input()[0].path)
        test_folder = self.input()[1][2].path
        gen = data_gen(test_folder, 1000)
        features, labels = next(gen)
        pred_labels = mdl.predict(features, batch_size=100, verbose=True)

        stacked = np.hstack([labels, pred_labels])
        np.savetxt(self.output().path, stacked, delimiter=";")


class LearnVAE1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/VAE1.mdl")

    def requires(self):
        return SplitTrainValidation(parallelism=self.parallelism)

    def run(self):
        from vae import VaeGenerator, vae_dnn
        from keras.optimizers import Adam
        EPOHCH_SIZE = 20000
        batch_size = 100

        VALIDATION_SIZE = 3000

        steps_per_epoch = EPOHCH_SIZE / batch_size
        validation_steps = VALIDATION_SIZE / batch_size

        train_fold = self.input()[0].path
        valid_fold = self.input()[1].path
        test_fold = self.input()[2].path

        train_gen = VaeGenerator(train_fold)
        val_gen = VaeGenerator(valid_fold)

        mdl = vae_dnn(train_gen.out_shape(), optimizer=Adam(lr=0.01))

        mdl.fit_generator(train_gen.generator(batch_size),
                          steps_per_epoch=steps_per_epoch,
                          epochs=7,
                          validation_data=val_gen.generator(batch_size),
                          validation_steps=validation_steps, use_multiprocessing=False, workers=1)

        mdl.save(self.output().path)


class VerifyVAE1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/vae1_check.csv")

    def requires(self):
        return [LearnVAE1(parallelism=self.parallelism), SplitTrainValidation(parallelism=self.parallelism)]

    def run(self):
        from keras.engine.saving import load_model
        from vae import VaeGenerator

        mdl = load_model(self.input()[0].path)
        test_folder = self.input()[1][2].path

        gen = VaeGenerator(test_folder)
        features, labels = next(gen.generator(10))
        pred_labels = mdl.predict(features, batch_size=100, verbose=True)

        for i in range(len(labels)):
            l = labels[i]
            p = pred_labels[i]
            stacked = np.vstack([l[:, :, 0], p[:, :, 0]])
            np.savetxt(self.output().path + str(i), stacked, delimiter=";")


class LearnVAE_Classifier1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/VAE_classifier1.mdl")

    def requires(self):
        return [LearnVAE1(parallelism=self.parallelism), SplitTrainValidation(parallelism=self.parallelism)]

    def run(self):
        from vae import VaeGenerator
        from keras.engine.saving import load_model
        from vae import vae_classifier

        EPOHCH_SIZE = 20000
        batch_size = 100

        VALIDATION_SIZE = 3000

        steps_per_epoch = EPOHCH_SIZE / batch_size
        validation_steps = VALIDATION_SIZE / batch_size

        mdl = load_model(self.input()[0].path)

        mdl = vae_classifier(mdl)

        train_fold = self.input()[1][0].path
        valid_fold = self.input()[1][1].path
        test_fold = self.input()[1][2].path

        train_gen = VaeGenerator(train_fold, isVAE=False)
        val_gen = VaeGenerator(valid_fold, isVAE=False)

        mdl.fit_generator(train_gen.generator(batch_size),
                          steps_per_epoch=steps_per_epoch,
                          epochs=7,
                          validation_data=val_gen.generator(batch_size),
                          validation_steps=validation_steps, use_multiprocessing=False, workers=1,
                          class_weight=CLASS_WEIGHTS)

        mdl.save(self.output().path)


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


if __name__ == "__main__":
    luigi.run()
