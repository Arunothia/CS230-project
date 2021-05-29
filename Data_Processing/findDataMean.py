import numpy as np
import os

data_folder = '../../dataset/processedData/trainSet/'
flute_paths = os.listdir(data_folder + 'flute/cqtChunks/')
piano_paths = os.listdir(data_folder + 'piano/cqtChunks/')

flute_array = []
piano_array = []

for file in flute_paths:
    x = np.load(data_folder + 'flute/cqtChunks/' + file)
    #m = np.mean(x)
    #if m < -30:
    #    print(file)
    #    print(m)
    flute_array.append(x)

for file in piano_paths:
    piano_array.append(np.load(data_folder + 'piano/cqtChunks/' + file))

flute_mean = flute_array.mean(axis=tuple(range(1, 2)))
piano_mean = piano_array.mean(axis=tuple(range(1, 2)))

flute_std = flute_array.std(axis=tuple(range(1, 2)))
piano_std = piano_array.std(axis=tuple(range(1, 2)))

print("Piano ==> mean: " + str(piano_mean) + " std: " + str(piano_std))
print("Flute ==> mean: " + str(flute_mean) + " std: " + str(flute_std))

# Output
#Piano ==> mean: -6.4774528 std: 5.372044
#Flute ==> mean: -6.9192457 std: 5.299747