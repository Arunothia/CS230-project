import os
import librosa
import librosa.display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from os import path
from pydub import AudioSegment
import numpy as np
from midi2audio import FluidSynth
from scipy.io import wavfile
from scipy.io.wavfile import write
from midi2audio import FluidSynth

rootPath = '/data/sample/'
inputMp3 = f'{rootPath}/BachPianoSolo.mp3'
outputMp3ToWav = f'{rootPath}/BachPianoSolo.wav'
inputMidi = f'{rootPath}/bwv988.mid'
outputMidiToWav = f'{rootPath}/bwv988.wav'

############################
# MIDI to wav conversion
############################

# using the default sound font in 44100 Hz sample rate
fs = FluidSynth()
fs.midi_to_audio(inputMidi, outputMidiToWav)

########################
# MP3 to wav conversion
########################

sound = AudioSegment.from_mp3(inputMp3)
sound.export(outputMp3ToWav, format="wav")


#sr, data = wavfile.read(outputMp3ToWav)
data, sr = librosa.load(outputMp3ToWav, sr=16000)
print ('Sampling rate = ', sr)

print(data.shape)
print ('Shape of wav audio data = ', data.shape)
print ('Max value = ', np.max(data))
print ('Min value = ', np.min(data))
fig, ax = plt.subplots()
ax.plot(data)
ax.set(xlabel = 'time sample', ylabel = 'amplitude', title = 'audio wave')
plt.show()
plt.close(fig)

numSeconds = 10
y = data[0:sr*numSeconds].astype('float32')

# 882 samples @ 44.1KHz = 20ms hop time
# 256 samples @ 16KHz = 16ms hop time
hopLength = 256
numBins = 336
numBinsPerOctave = 48

C = np.abs(librosa.cqt(y, sr=sr, hop_length = hopLength, n_bins = numBins, bins_per_octave = numBinsPerOctave))

# plot the spectogram
mpl.rcParams['figure.figsize'] = [15, 10]
mpl.rcParams['font.size'] = 12
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

print(C.shape)
fig, ax = plt.subplots()
img = ax.imshow(np.log(C))
ax.set_title('log(CQT)')
fig.colorbar(img, ax=ax)

#####################
# STFT
#####################
print (sr)