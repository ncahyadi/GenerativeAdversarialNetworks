import tensorflow as tf
import numpy as np

# generator functions
def build_generator1():
    network = tf.keras.Sequential()
    network.add(tf.keras.layers.Dense(units=7*7*256, 
                                      use_bias=False, 
                                      input_shape=(100,)))
    network.add(tf.keras.layers.BatchNormalization())
    network.add(tf.keras.layers.LeakyReLU())
    
    network.add(tf.keras.layers.Reshape((7,7,256)))
    
    network.add(tf.keras.layers.Conv2DTranspose(filters=128, 
                                                kernel_size=(5,5), 
                                                padding='same', 
                                                use_bias=False))
    network.add(tf.keras.layers.BatchNormalization())
    network.add(tf.keras.layers.LeakyReLU())
    
    network.add(tf.keras.layers.Conv2DTranspose(filters=64, 
                                                kernel_size=(5,5), 
                                                padding='same', 
                                                strides=(2,2), 
                                                use_bias=False))
    network.add(tf.keras.layers.LeakyReLU())
    
    network.add(tf.keras.layers.Conv2DTranspose(filters=1, 
                                                kernel_size=(5,5), 
                                                padding='same', 
                                                strides=(2,2), 
                                                use_bias=False, 
                                                activation='tanh'))
    
    network.summary()
    
    return network

def generator_loss1(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# discriminator functions
def build_discriminator1():
    network = tf.keras.Sequential()
    
    network.add(tf.keras.layers.Conv2D(filters=64, 
                                       kernel_size=(5,5),
                                       strides=(2,2), 
                                       padding='same', 
                                       input_shape=[28,28,1]))
    network.add(tf.keras.layers.LeakyReLU())
    network.add(tf.keras.layers.Dropout(0.3))
    
    network.add(tf.keras.layers.Conv2D(filters=128, 
                                       kernel_size=(5,5),
                                       strides=(2,2), 
                                       padding='same'))
    network.add(tf.keras.layers.LeakyReLU())
    network.add(tf.keras.layers.Dropout(0.3))
    
    network.add(tf.keras.layers.Flatten())
    network.add(tf.keras.layers.Dense(1))
    
    network.summary()
    
    return network

def discriminator_loss1(expected_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(expected_output), expected_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss