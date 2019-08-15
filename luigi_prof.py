import json
import os
import time
import zipfile

import numpy as np

from utils import game_to_numpy

ms1 = time.time()
for f in os.listdir("C:\\tmp\\kgs_learn_workdir\\ttt"):
    data = np.load("C:\\tmp\\kgs_learn_workdir\\ttt\\" + f)
    b = data["boards"]
    m = data["moves"]
    l = data["labels"]

    print(f)

print(time.time() - ms1)
