import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

import sys, os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('../../utils')
print(os.getcwd())

import dataset_function
import tf_gan_components
import file_mgmt
import tf_generate_gif
import inferences


tf.__version__

@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = dataset_function.resize_efrosgans(input_image, real_image, 286, 286)
    input_image, real_image = dataset_function.random_crop_efronsgans(input_image, real_image, img_heigth, img_width)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train_pix2pix(image_file):
    input_image, real_image = dataset_function.load_efrosgans(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = dataset_function.normalize_efronsgans(input_image, real_image)

    return input_image, real_image

def load_image_test_pix2pix(image_file):
    input_image, real_image = dataset_function.load_efrosgans(image_file)
    input_image, real_image = dataset_function.resize_efrosgans(input_image, real_image, img_heigth, img_width)
    input_image, real_image = dataset_function.normalize_efronsgans(input_image, real_image)

    return input_image, real_image

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])


    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (Lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
        
def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            inferences.generate_images(generator, example_input, example_target, img_result_training, step)
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

if __name__ == "__main__":
  ds_index = 3
  PATH = dataset_function.get_efrosgans_ds(ds_index)
  list(PATH.parent.iterdir())
  buffer_size = 400
  batch_size = 1
  img_width = 256
  img_heigth = 256
  output_channels = 3
  Lambda = 100
  id_experiment = 'coba_facades'
  training_dir = file_mgmt.create_dir(file_mgmt.info_dir(), id_experiment)
  img_result_training = file_mgmt.create_dir(training_dir, 'saved_image_from_training')
  checkpoint_dir = file_mgmt.create_dir(training_dir, 'training_checkpoints')
  checkpoint_prefix = file_mgmt.create_dir(checkpoint_dir, 'checkpoints')
  log_dir=file_mgmt.create_dir(checkpoint_dir, 'logs')

  train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
  train_dataset = train_dataset.map(load_image_train_pix2pix, num_parallel_calls=tf.data.AUTOTUNE)
  train_dataset = train_dataset.shuffle(buffer_size)
  train_dataset = train_dataset.batch(batch_size)
  try:
      test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
  except tf.errors.InvalidArgumentError:
      test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
  test_dataset = test_dataset.map(load_image_test_pix2pix)
  test_dataset = test_dataset.batch(batch_size)
  generator = Generator()
  tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
  discriminator = Discriminator()
  tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

  summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  # %load_ext tensorboard
  # %tensorboard --logdir {log_dir}
  fit(train_dataset, test_dataset, steps=40000)
  tf_generate_gif.create_gif(id_experiment, img_result_training)