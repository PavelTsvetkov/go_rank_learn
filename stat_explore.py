import os
import numpy as np

folder = "C:\\tmp\\kgs_learn_workdir\\processed_games\\train"

dan_count = [0 for x in range(9)]

for ix,f in enumerate(os.listdir(folder)):
    data = np.load("/".join([folder, f]))
    labels = data["labels"]
    idx = np.argmax(labels[0])
    dan_count[idx] += 1
    idx = np.argmax(labels[1])
    dan_count[idx] += 1
    print(ix)

print(dan_count)
