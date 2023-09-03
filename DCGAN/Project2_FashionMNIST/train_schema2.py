import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys, os
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
def train1(images):
    noise = tf.random.normal([batch_size, noise_dimension])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        expected_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = tf_gan_components.generator_loss1(fake_output)
        disc_loss = tf_gan_components.discriminator_loss1(expected_output, fake_output)
    
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

def train_gan1(dataset, epochs, test_images):
    for epoch in range(epochs):
        for image_batch in dataset:
            train1(image_batch)
        print('Epoch: ', epoch+1)
        generated_images = generator(test_images, training=False)
        fig = plt.figure(figsize=(10,10))
        for i in range(generated_images.shape[0]):
            plt.subplot(4,4,i+1)
            plt.imshow(generated_images[i, :, :, 0]*127.5+127.5, cmap='gray')
            plt.axis('off')
        fig.savefig(os.path.join(img_result_training, f'image{epoch}.png'), bbox_inches='tight')
        plt.close(fig)


def proceed_training1(X_train_ds, number_of_images, noise_dimension, epochs):
    tf.config.run_functions_eagerly(True)
    X_train_batch = X_train_ds.as_numpy_iterator().next()
    train1(X_train_batch)
    test_images = tf.random.normal([number_of_images, noise_dimension])
    train_gan1(X_train_ds, epochs, test_images)
    tf_generate_gif.create_gif(id_experiment, img_result_training)

if __name__ == "__main__":
    batch_size = 256
    gen_lr = 0.0001
    dis_lr = 0.0001
    epochs = 100
    noise_dimension = 100
    number_of_images = 16
    id_experiment = 'coba_fashion'
    training_dir = file_mgmt.create_dir(file_mgmt.info_dir(), id_experiment)
    img_result_training = file_mgmt.create_dir(training_dir, 'saved_image_from_training')

    X_train_ds, y_train = tf_dataset_function.prepare_fashion_mnist(batch_size)

    generator = tf_gan_components.build_generator1()
    noise = tf.random.normal([1, 100])
    discriminator = tf_gan_components.build_discriminator1()

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=dis_lr)
    proceed_training1(X_train_ds, number_of_images, noise_dimension, epochs)