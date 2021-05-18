import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import time
from IPython.display import clear_output
import random
import CycleGAN.args as args

# Arguments
opt = args.get_setup_args()

AUTOTUNE = tf.data.AUTOTUNE

BUFFER_SIZE = opt.buffer_size
BATCH_SIZE = opt.batch_size

train_piano = np.zeros((opt.train_size, opt.img_width, opt.img_height, opt.output_channels))
for e,filename in enumerate(os.listdir(opt.input_data_piano_path)):
    if filename.endswith('.npy'):
        train_piano[e] = np.expand_dims(np.load(opt.input_data_piano_path+filename), axis=-1)

train_flute = np.zeros((opt.train_size, opt.img_width, opt.img_height, opt.output_channels))
for e,filename in enumerate(os.listdir(opt.input_data_flute_path)):
    if filename.endswith('.npy'):
        train_flute[e] = np.expand_dims(np.load(opt.input_data_flute_path+filename), axis=-1)

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[opt.img_width, opt.img_height, opt.output_channels])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 360 x 270 x 3
  image = tf.image.resize(image, [360, 270],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 336 x 250 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def pad_zeros(image):
  # resizing to 336 x 336 x 3
  return np.pad(image, ((336, 336), (250, 336)), 'constant', (0,0))

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

sample_piano = train_piano[0, :, :, :]
sample_flute = train_flute[0, :, :, :]

plt.subplot(121)
plt.title('piano')
img_piano = sample_piano * 0.5 + 0.5
plt.imshow(img_piano)
plt.savefig(opt.sample_data_path + 'sample_piano.jpg')

plt.subplot(122)
plt.title('piano with random jitter')
img_piano_jitter = random_jitter(sample_piano) * 0.5 + 0.5
plt.imshow(img_piano_jitter)
plt.savefig(opt.sample_data_path + 'sample_piano_jitter.jpg')

plt.subplot(121)
plt.title('flute')
img_flute = sample_flute * 0.5 + 0.5
plt.imshow(img_flute)
plt.savefig(opt.sample_data_path + 'sample_flute.jpg')

plt.subplot(122)
plt.title('flute with random jitter')
img_flute_jitter = random_jitter(sample_flute) * 0.5 + 0.5
plt.imshow(img_flute_jitter)
plt.savefig(opt.sample_data_path + 'sample_flute_jitter.jpg')
    
def generate_images(model, test_input, epoch):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input', 'Predicted']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    img = display_list[i] * 0.5 + 0.5
    plt.imshow(img)
    plt.savefig(opt.output_path + title[i] + str(epoch))
    plt.axis('off')