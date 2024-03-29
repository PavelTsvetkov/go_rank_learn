import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import luigi
import numpy as np

from constants import WORKDIR, GAMES_MASK, MODELS_DIR, HISTORY_KEY, META_KEY
from goban import load_goban

from utils import game_to_numpy, CLASS_WEIGHTS, game_to_numpy_alpha


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
    file_mod = luigi.IntParameter()

    sample_count = luigi.IntParameter(default=0)
    black_mode = luigi.BoolParameter(default=False)
    sub_folder = luigi.Parameter(default="/processed_games")

    def output(self):
        return [luigi.LocalTarget(WORKDIR + self.sub_folder + x + str(self.file_mod)) for x in
                SplitTrainValidation.folders]

    def requires(self):
        return ProcessDir(parallelism=self.parallelism, file_mod=self.file_mod)

    def run(self):
        out_zips = [x.path + "_tmp" for x in self.output()]
        for x in out_zips:
            shutil.rmtree(x, ignore_errors=True)
            os.makedirs(x, exist_ok=True)
        try:
            self.process_input(self.input(), out_zips)
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
                    data = self.game_to_data(game)
                    np.savez_compressed(out + "/" + name, **data)
                except BaseException as ex:
                    print("Failed processing of ", name, " : ", ex)
                print(inp.path, " : ", idx + 1, " of ", len(namelist))

    def game_to_data(self, game):
        return game_to_numpy(game, black_mode=self.black_mode)


class SplitTrainValidation_Alpha(SplitTrainValidation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def game_to_data(self, game):
        return game_to_numpy_alpha(game)


class SplitTrainValidation_Alpha_Gather(luigi.Task):
    parallelism = luigi.IntParameter()
    sub_folder = luigi.Parameter(default="/processed_games")

    def output(self):
        return [luigi.LocalTarget(WORKDIR + self.sub_folder + x) for x in
                SplitTrainValidation.folders]

    def requires(self):
        for k in range(self.parallelism):
            yield SplitTrainValidation_Alpha(parallelism=self.parallelism, file_mod=k, sub_folder=self.sub_folder)

    def run(self):
        for inp in self.input():
            for f in inp:
                src_dir = f.path
                target_dir = src_dir[0:-1]
                os.makedirs(target_dir, exist_ok=True)
                for file in os.listdir(src_dir):
                    os.rename(src_dir + "/" + file, target_dir + "/" + file)
                os.rmdir(src_dir)


class LearnModel1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/model1.mdl")

    def requires(self):
        return SplitTrainValidation_Alpha_Gather(parallelism=self.parallelism, sub_folder="/processed_alpha")

    def run(self):
        from keras_utils import data_gen_alpha
        from keras_utils import keras_model1, data_gen
        from keras.optimizers import Adam
        EPOHCH_SIZE = 20000
        batch_size = 32

        VALIDATION_SIZE = 3000

        steps_per_epoch = EPOHCH_SIZE / batch_size
        validation_steps = VALIDATION_SIZE / batch_size

        mdl = keras_model1(optimizer=Adam(lr=0.01), loss="categorical_crossentropy")

        train_fold = self.input()[0].path
        valid_fold = self.input()[1].path
        test_fold = self.input()[2].path

        smp_gen = data_gen_alpha

        train_gen = smp_gen(train_fold, batch_size)
        val_gen = smp_gen(valid_fold, batch_size)

        mdl.fit_generator(train_gen,
                          steps_per_epoch=steps_per_epoch,
                          epochs=7,
                          validation_data=val_gen,
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
        from keras_utils import data_gen_alpha
        from keras.engine.saving import load_model

        mdl = load_model(self.input()[0].path)
        test_folder = self.input()[1][2].path

        smp_gen = data_gen_alpha

        gen = smp_gen(test_folder, 1000)
        features, labels = next(gen)
        pred_labels = mdl.predict(features, batch_size=100, verbose=True)

        stacked = np.hstack([labels, pred_labels])
        np.savetxt(self.output().path, stacked, delimiter=";")


class LearnVAE1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/VAE1.mdl")

    def requires(self):
        return SplitTrainValidation_Alpha_Gather(parallelism=self.parallelism, sub_folder="/processed_alpha")

    def run(self):
        from vae import VaeGenerator, vae_dnn
        from keras.optimizers import Adam
        from keras_utils import data_gen_alpha
        from keras_utils import sample_shape
        from keras.optimizers import Adadelta

        EPOHCH_SIZE = 20000
        batch_size = 100

        VALIDATION_SIZE = 3000

        steps_per_epoch = EPOHCH_SIZE / batch_size
        validation_steps = VALIDATION_SIZE / batch_size

        train_fold = self.input()[0].path
        valid_fold = self.input()[1].path
        test_fold = self.input()[2].path

        train_gen = data_gen_alpha(train_fold, batch_size, vae_mode=True, flatten=True)
        val_gen = data_gen_alpha(valid_fold, batch_size, vae_mode=True, flatten=True)

        # optimizer = Adam(lr=0.001)
        optimizer = Adadelta()

        mdl = vae_dnn(sample_shape(3), optimizer=optimizer)

        mdl.fit_generator(train_gen,
                          steps_per_epoch=steps_per_epoch,
                          epochs=5,
                          validation_data=val_gen,
                          validation_steps=validation_steps, use_multiprocessing=False, workers=1)

        mdl.save(self.output().path)


class VerifyVAE1(luigi.Task):
    parallelism = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(MODELS_DIR + "/vae1_check.csv")

    def requires(self):
        return [LearnVAE1(parallelism=self.parallelism),
                SplitTrainValidation_Alpha_Gather(parallelism=self.parallelism, sub_folder="/processed_alpha")]

    def run(self):
        from keras.engine.saving import load_model
        from vae import VaeGenerator
        from keras_utils import sample_shape
        from keras_utils import data_gen_alpha

        mdl = load_model(self.input()[0].path)
        test_folder = self.input()[1][2].path

        batch_size = 10
        gen = data_gen_alpha(test_folder, batch_size, vae_mode=True)
        features, labels = next(gen)
        pred_labels = mdl.predict(features, batch_size=100, verbose=True)
        pred_labels=np.reshape(pred_labels,(batch_size,19,19,3))

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
        return [LearnVAE1(parallelism=self.parallelism),
                SplitTrainValidation_Alpha_Gather(parallelism=self.parallelism, sub_folder="/processed_alpha")]

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


if __name__ == "__main__":
    luigi.run()
