import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
from IPython.display import clear_output

import CycleGAN.args as args

# Arguments
opt = args.get_setup_args()

AUTOTUNE = tf.data.AUTOTUNE

BUFFER_SIZE = opt.buffer_size
BATCH_SIZE = opt.batch_size
IMG_WIDTH = opt.img_width
IMG_HEIGHT = opt.img_height

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

train_horses = train_horses.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).cache()

train_zebras = train_zebras.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).cache()

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).cache()

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).cache()


sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

plt.subplot(121)
plt.title('Horse')
img_horse = (sample_horse[0] * 0.5 + 0.5)
plt.imshow(img_horse)
plt.show()
#plt.imsave(opt.sample_data_path + 'sample_horse.jpg', img_horse)

plt.subplot(122)
plt.title('Horse with random jitter')
img_horse_jitter = (random_jitter(sample_horse[0]) * 0.5 + 0.5)
plt.imshow(img_horse_jitter)
plt.show()
#plt.imsave(opt.sample_data_path + 'sample_horse_jitter.jpg', img_horse_jitter)

plt.subplot(121)
plt.title('Zebra')
img_zebra = (sample_zebra[0] * 0.5 + 0.5)
plt.imshow(img_zebra)
plt.show()
#plt.imsave(opt.sample_data_path + 'sample_zebra.jpg', img_zebra)

plt.subplot(122)
plt.title('Zebra with random jitter')
img_zebra_jitter = (random_jitter(sample_zebra[0]) * 0.5 + 0.5)
plt.imshow(img_zebra_jitter)
plt.show()
#plt.imsave(opt.sample_data_path + 'sample_zebra.jpg', img_zebra_jitter)
    
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
    plt.imsave(opt.output_path + title[i] + str(epoch), img, cmap=cmap)
    plt.axis('off')