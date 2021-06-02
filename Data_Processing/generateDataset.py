# This file takes in the MIDI files from the piano and flute MIDI datasets and processes it in the following steps:
# 1. Split the dataset for each instrument into train
# For each song (piano or flute MIDI), we split it into 4 second chunks CQT chunks of 4 second durations in the source and transformed domain
# 2. 

import glob
import os
import sys
import subprocess
import librosa
import librosa.display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from os import path
from pydub import AudioSegment
from midi2audio import FluidSynth
from scipy.io import wavfile
from scipy.io.wavfile import write
import soundfile as sf

from multiprocessing import Pool
from multiprocessing import cpu_count
curDir = os.getcwd()
topDir = f'{curDir}/..'
datasetDir = f'{topDir}/dataset'
rawTrainSetDir = f'{datasetDir}/rawData/midiFiles/trainSet'
processedTrainSetDir = f'{datasetDir}/processedData/trainSet'

numCores = cpu_count()

###################
# CQT specs
###################

# sampling rate
sr = 16000

# 256 samples @ 16KHz = 16ms hop time
hopLength = 256

numBins = 336
numBinsPerOctave = 48
filterScale = 0.8
numSecondsPerChunk = 4

###################
# soundfonts
###################
instruments = ['piano', 'flute']
soundFontNum = {'piano': 0, 'flute': 73}


def processMidiFile(midiFilePath, outputDir, soundFontNum, numSecondsPerChunk = 4, generateCqt = True, generateWav = False, verbose = False):
	
	# input - midiFile, outputDir, soundFontNumber of piano or flute timbre, numSeconds per chunk
	# outputs - cqtChunks in piano and flute, wavChunks in piano and flute
   
	if generateCqt or generateWav:

		midiFileName = midiFilePath.split('/')[-1]
		print (f'Processing MIDI File - {midiFileName}')
	
		wavFileName = f'{midiFileName[:-4]}.wav'
		wavFilePath = f'{outputDir}/{wavFileName}'
	
		# generate wav file
		subprocess.run(f"timidity -s {sr} --force-program={soundFontNum}/1 {midiFilePath} -Ow -o {wavFilePath}", shell = True, stdout = subprocess.DEVNULL)
	
		# load the wav file
		data, _ = librosa.load(wavFilePath, sr=sr)
	
		# create CQT
		cqtLogMag = np.log(np.abs(librosa.cqt(data, sr=sr, hop_length = hopLength, n_bins = numBins, bins_per_octave = numBinsPerOctave, filter_scale = filterScale)))
	
		# split C into time chunks
		numCqtSamplesPerChunk = int(sr * numSecondsPerChunk / hopLength)
		idxList = np.arange(start = 0, stop = cqtLogMag.shape[1], step = numCqtSamplesPerChunk)
		idxList = idxList[1:]
		cqtChunkList = np.split(cqtLogMag, idxList, axis = 1)
	
		# check is last 2 matrices have different shape. Its possible!
		if verbose:
			print ('BEFORE rejecting the last sample...')
			print (f'# of CQT chunks = {len(cqtChunkList)}')
			print (f'shape of penultimate matrix = {cqtChunkList[-2].shape}')
			print (f'shape of last matrix = {cqtChunkList[-1].shape}')
	
		# drop the last CQT chunk since the last CQT image may be of a reduced size
		cqtChunkList = cqtChunkList[:-1]
	
		# check again. Last 2 matrices should have same shape now.
		if verbose:
			print ('AFTER rejecting the last sample...')
			print (f'# of CQT chunks = {len(cqtChunkList)}')
			print (f'shape of penultimate matrix = {cqtChunkList[-2].shape}')
			print (f'shape of last matrix = {cqtChunkList[-1].shape}')
	
		# write out the CQT and wav chunks
		cqtChunkDir = f'{outputDir}/cqtChunks'
		wavChunkDir = f'{outputDir}/wavChunks'
	
		for i in range(len(cqtChunkList)):
	
			if generateCqt:

				# write out the CQT chunks
				outputFileName = f'{cqtChunkDir}/{midiFileName[:-4]}_chunk{i}.npy'
				np.save(outputFileName, cqtChunkList[i])
	
			if generateWav:
	
				# write out the wav chunks
				outputFileName = f'{wavChunkDir}/{midiFileName[:-4]}_chunk{i}.wav'
				y = data[sr * numSecondsPerChunk * i : sr * numSecondsPerChunk * (i+1)]
				sf.write(outputFileName, y, sr, subtype = 'PCM_24')

		# delete the wavFile at the end
		subprocess.run(f'rm -f {wavFilePath}', shell = True, stdout = subprocess.DEVNULL)

	return


print (f'Number of cores on system = {numCores}\n')

# get a list of MIDI files in the trainSet
print ('Loading up MIDI files...\n')

midiFileList = glob.glob(f'{rawTrainSetDir}/*.mid')
print (f'# of MIDI files = {len(midiFileList)}\n')

# generate piano and flute datasets
for instrument in instruments:

	print (f'Generating dataset for {instrument}...\n')
	outputDir = f'{processedTrainSetDir}/{instrument}'
	
	for i, midiFile in enumerate(midiFileList):
		print (f'File {i}...')
		processMidiFile(midiFile, outputDir, soundFontNum[instrument], numSecondsPerChunk, generateCqt = True, generateWav = False, verbose = False)



