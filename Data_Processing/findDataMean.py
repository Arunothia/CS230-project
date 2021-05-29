import numpy as np
import os

data_folder = 'data/tempData/trainSet/'
flute_paths = os.listdir(data_folder + 'fluteCqtChunks/')
piano_paths = os.listdir(data_folder + 'pianoCqtChunks/')

flute_array = []
piano_array = []

for file in flute_paths:
    flute_array.append(np.load(data_folder + 'fluteCqtChunks/' + file))

for file in piano_paths:
    piano_array.append(np.load(data_folder + 'pianoCqtChunks/' + file))

flute_mean = np.mean(flute_array)
piano_mean = np.mean(piano_array)

flute_std = np.std(flute_array)
piano_std = np.std(piano_array)

print("Piano ==> mean: " + piano_mean + " std: " + piano_std)
print("Flute ==> mean: " + flute_mean + " std: " + flute_std)