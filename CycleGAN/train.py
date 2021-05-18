import CycleGAN.setup as setup
import CycleGAN.loss as loss
import CycleGAN.args as args

# Arguments
opt = args.get_setup_args()

generator_g = setup.pix2pix.unet_generator(opt.output_channels, norm_type=opt.norm_type)
generator_f = setup.pix2pix.unet_generator(opt.output_channels, norm_type=opt.norm_type)

discriminator_x = setup.pix2pix.discriminator(norm_type=opt.norm_type, target=False)
discriminator_y = setup.pix2pix.discriminator(norm_type=opt.norm_type, target=False)

to_flute = generator_g(setup.sample_piano)
to_piano = generator_f(setup.sample_flute)
setup.plt.figure(figsize=(8, 8))
contrast = 8

imgs = [setup.sample_piano, to_flute, setup.sample_flute, to_piano]
title = ['piano', 'To flute', 'flute', 'To piano']

for i in range(len(imgs)):
  setup.plt.subplot(2, 2, i+1)
  setup.plt.title(title[i])
  if i % 2 == 0:
    setup.plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    setup.plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
setup.plt.show()

setup.plt.figure(figsize=(8, 8))

setup.plt.subplot(121)
setup.plt.title('Is a real flute?')
setup.plt.imshow(discriminator_y(setup.sample_flute)[0, ..., -1], cmap='RdBu_r')
plt.savefig(opt.sample_data_path + 'is-real-flute.jpg')

setup.plt.subplot(122)
setup.plt.title('Is a real piano?')
setup.plt.imshow(discriminator_x(setup.sample_piano)[0, ..., -1], cmap='RdBu_r')
plt.savefig(opt.sample_data_path + 'is-real-piano.jpg')
setup.plt.show()

generator_g_optimizer = setup.tf.keras.optimizers.Adam(opt.lr, beta_1=opt.b1)
generator_f_optimizer = setup.tf.keras.optimizers.Adam(opt.lr, beta_1=opt.b1)

discriminator_x_optimizer = setup.tf.keras.optimizers.Adam(opt.lr, beta_1=opt.b1)
discriminator_y_optimizer = setup.tf.keras.optimizers.Adam(opt.lr, beta_1=opt.b1)

checkpoint_path = "./checkpoints/train"

ckpt = setup.tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = setup.tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=opt.checkpoints_keep)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

@setup.tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with setup.tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = loss.generator_loss(disc_fake_y)
    gen_f_loss = loss.generator_loss(disc_fake_x)
    
    total_cycle_loss = loss.calc_cycle_loss(real_x, cycled_x) + loss.calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + loss.identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + loss.identity_loss(real_x, same_x)

    disc_x_loss = loss.discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = loss.discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

for epoch in range(opt.num_epochs):
  start = setup.time.time()

  n = 0
  for image_x, image_y in setup.tf.data.Dataset.zip((setup.train_piano, setup.train_flute)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_piano) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, setup.sample_piano, epoch)

  if (epoch + 1) % opt.checkpoint_epochs == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))