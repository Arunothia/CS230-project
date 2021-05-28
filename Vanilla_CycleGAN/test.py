# Run the trained model on the test dataset
for inp in test_piano.take(5):
  generate_images(generator_g, inp)