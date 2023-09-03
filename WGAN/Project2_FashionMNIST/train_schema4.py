import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys, os
import time
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('../../utils')
tf.__version__

import tf_dataset_function
import tf_gan_components
import file_mgmt
import tf_generate_gif

@tf.function
def gradient_penalty(real, fake, epsilon):
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        scores = discriminator(interpolated_images)[0]
    gradient = tape.gradient(scores, interpolated_images)[0]
    gradient_norm = tf.norm(gradient)
    gp = tf.math.reduce_mean((gradient_norm - 1)**2)
    return gp

def training_step(images):
    noise = tf.random.normal([batch_size, noise_dimension])
    discriminator_extra_steps = 3
    for i in range(discriminator_extra_steps):
        with tf.GradientTape() as d_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            epsilon = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
            gp = gradient_penalty(images, generated_images, epsilon)
            d_loss = tf_gan_components.wasserstein_discloss(real_output, fake_output, gp)
        discriminator_gradient = d_tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            g_loss = tf_gan_components.wasserstein_genloss(fake_output)
        generator_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

def create_and_save_images(model, epoch, test_input):
    preds = model(test_input, training = False)
    fig = plt.figure(figsize = (4,4))
    for i in range(preds.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(preds[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    fig.savefig(os.path.join(img_result_training, f'image{epoch}.png'), bbox_inches='tight')
    plt.close(fig)       


def train(dataset, epochs, seed):   
    for epoch in range(epochs):
        initial = time.time()
        for img_batch in dataset:
            if len(img_batch) == batch_size:
                training_step(img_batch)
        create_and_save_images(generator, epoch+1, seed)
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print('Time taken to process epoch {} was {} seconds'.format(epoch + 1, time.time() - initial))
    create_and_save_images(generator, epochs, seed)
    generator.save(os.path.join(training_dir, saved_model_name))
    

if __name__ == "__main__":
    batch_size = 256
    gen_lr = 0.0002
    dis_lr = 0.0002
    epochs = 100
    noise_dimension = 100
    number_of_images = 16
    seed = tf.random.normal([number_of_images, noise_dimension])
    id_experiment = 'coba_6'
    saved_model_name = f'{id_experiment}_generator.h5'

    training_dir = file_mgmt.create_dir(file_mgmt.info_dir(), id_experiment)
    img_result_training = file_mgmt.create_dir(training_dir, 'saved_image_from_training')
    checkpoint_dir = file_mgmt.create_dir(training_dir, 'training_checkpoints')
    checkpoint_prefix = file_mgmt.create_dir(checkpoint_dir, 'checkpoints')

    X_train_ds, y_train = tf_dataset_function.prepare_fashion_mnist(batch_size)

    generator = tf_gan_components.build_generator1()
    discriminator = tf_gan_components.build_discriminator1()

    # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
    # discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=dis_lr)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.9)

    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    discriminator = discriminator)
    
    train(X_train_ds, epochs, seed)
    tf_generate_gif.create_gif(id_experiment, img_result_training)