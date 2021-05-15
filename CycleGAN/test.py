# Run the trained model on the test dataset
for inp in test_horses.take(5):
  generate_images(generator_g, inp)