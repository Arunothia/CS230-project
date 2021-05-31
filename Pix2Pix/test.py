import torch
import librosa
import librosa.display
from utils import load_checkpoint
import torch.optim as optim
import config
from generator_model import Generator
import numpy as np
import soundfile as sf


###########################################################
# CQT specs
###########################################################

# sampling rate
sr = 16000

# 256 samples @ 16KHz = 16ms hop time
hopLength = 256

numBins = 336
numBinsPerOctave = 48
filterScale = 0.8
numSecondsPerChunk = 4
###########################################################

wavFilePath = "data/mist-flute-chill-melody_112bpm_A_minor.wav"


def main():
  gen = Generator().to(config.DEVICE)

  opt_gen = optim.Adam(
    list(gen.parameters()),
    lr = config.LEARNING_RATE,
    betas=(0.5, 0.999),
  )

  load_checkpoint(
    config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
  )

  # load the wav
  data, sr = librosa.load(wavFilePath, sr=16000)

  # perform CQT
  fluteCqt = np.abs(librosa.cqt(data, sr=sr, hop_length = hopLength, n_bins = numBins, bins_per_octave = numBinsPerOctave, filter_scale = filterScale))

  fluteCqtMag = np.abs(fluteCqt)
  fluteCqtPhase = np.angle(fluteCqt)

  x = (np.expand_dims(np.expand_dims(fluteCqtMag, axis=0), axis=0))
  x = x[:, :, :, 0:336]

  print(x.shape)

  fakePianoCQTMag = gen(torch.from_numpy(x)).detach().numpy()
  fakePianoCQTMag = np.squeeze(fakePianoCQTMag)
  fluteCqtPhase = fluteCqtPhase[:, 0:336]
  
  print(fluteCqtPhase.shape)
  print(fakePianoCQTMag.shape)

  # Reconstruct flute wav from flute abs(CQT) + piano phase
  minLength = min(fluteCqtPhase.shape[1], fakePianoCQTMag.shape[1])
  pianoCqtReconstructed = np.abs(fakePianoCQTMag[:, 0:minLength]) * np.exp(1j*np.angle(fluteCqtPhase[:, 0:minLength]))
  data = librosa.icqt(pianoCqtReconstructed, sr=sr, hop_length = hopLength, bins_per_octave = numBinsPerOctave, filter_scale = filterScale)
  
  pianoReconstructedFromFlutePhaseWav = f'data/pianoReconstructedFromFlutePhase.wav'
  sf.write(pianoReconstructedFromFlutePhaseWav, data, sr, subtype='PCM_24')


if __name__ == "__main__":
  main()