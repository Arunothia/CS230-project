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


folder = "../Pix2Pix/data/"
#fluteWavFile = "mist-flute-chill-melody_112bpm_A_minor.wav"
#fluteFile = "mozartFlute.wav"
#pianoFile = "mozartPiano.wav"
fluteFile = "testFlute.wav"
pianoFile = "testPiano.wav"
fluteWavFilePath = folder + fluteFile
pianoWavFilePath = folder + pianoFile


def main():
  gen_piano = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)
  gen_flute = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)

  opt_gen = optim.Adam(
    list(gen_piano.parameters()) + list(gen_flute.parameters()),
    lr = config.LEARNING_RATE,
    betas=(0.5, 0.999),
  )

  load_checkpoint(
    config.CHECKPOINT_GEN_P, gen_piano, opt_gen, config.LEARNING_RATE,
  )
  load_checkpoint(
    config.CHECKPOINT_GEN_F, gen_flute, opt_gen, config.LEARNING_RATE,
  )

  # load the wav
  dataFlute, sr = librosa.load(fluteWavFilePath, sr=16000)
  dataPiano, sr = librosa.load(pianoWavFilePath, sr=16000)

  # perform CQT
  fluteCqt = np.abs(librosa.cqt(dataFlute, sr=sr, hop_length = hopLength, n_bins = numBins, bins_per_octave = numBinsPerOctave, filter_scale = filterScale))
  pianoCqt = np.abs(librosa.cqt(dataPiano, sr=sr, hop_length = hopLength, n_bins = numBins, bins_per_octave = numBinsPerOctave, filter_scale = filterScale))

  fluteCqtMag, fluteCqtPhase = np.abs(fluteCqt), np.angle(fluteCqt)
  pianoCqtMag, pianoCqtPhase = np.abs(pianoCqt), np.angle(pianoCqt)

  x = (np.expand_dims(np.expand_dims(fluteCqtMag, axis=0), axis=0))
  x = x[:, :, :, 0:336]

  print(x.shape)

  fakePianoCQTMag = gen_piano(torch.from_numpy(x)).detach().numpy()
  fakePianoCQTMag = np.squeeze(fakePianoCQTMag)
  fluteCqtPhase = fluteCqtPhase[:, 0:336]
  
  print(fluteCqtPhase.shape)
  print(fakePianoCQTMag.shape)

  x = (np.expand_dims(np.expand_dims(pianoCqtMag, axis=0), axis=0))
  x = x[:, :, :, 0:336]

  print(x.shape)

  fakeFluteCQTMag = gen_flute(torch.from_numpy(x)).detach().numpy()
  fakeFluteCQTMag = np.squeeze(fakeFluteCQTMag)
  pianoCqtPhase = pianoCqtPhase[:, 0:336]
  
  print(pianoCqtPhase.shape)
  print(fakeFluteCQTMag.shape)
  

  # Reconstruct piano wav from piano abs(CQT) + flute phase
  minLength = min(fluteCqtPhase.shape[1], fakePianoCQTMag.shape[1])
  pianoCqtReconstructed = np.abs(fakePianoCQTMag[:, 0:minLength]) * np.exp(1j*np.angle(fluteCqtPhase[:, 0:minLength]))
  genPiano = librosa.icqt(pianoCqtReconstructed, sr=sr, hop_length = hopLength, bins_per_octave = numBinsPerOctave, filter_scale = filterScale)

  # Reconstruct flute wav from flute abs(CQT) + piano phase
  minLength = min(pianoCqtPhase.shape[1], fakeFluteCQTMag.shape[1])
  fluteCqtReconstructed = np.abs(fakeFluteCQTMag[:, 0:minLength]) * np.exp(1j*np.angle(pianoCqtPhase[:, 0:minLength]))
  genFlute = librosa.icqt(fluteCqtReconstructed, sr=sr, hop_length = hopLength, bins_per_octave = numBinsPerOctave, filter_scale = filterScale)
  
  pianoReconstructed = f'data/pianoReconstructed_{fluteFile}.wav'
  sf.write(pianoReconstructed, genPiano, sr, subtype='PCM_24')
  fluteReconstructed = f'data/fluteReconstructed_{pianoFile}.wav'
  sf.write(fluteReconstructed, genFlute, sr, subtype='PCM_24')

if __name__ == "__main__":
  main()