import tensorflow as tf

# mnist dataset from tensorflow
def prepare_mnist_dataset(batch_size):
    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train_normalized = ((X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') - 127.5) / 127.5)
    buffer_size = X_train.shape[0]
    X_train_ds = tf.data.Dataset.from_tensor_slices(X_train_normalized).shuffle(buffer_size).batch(batch_size)
    return X_train_ds, y_train


# fashion mnist dataset from tensorflow
def prepare_fashion_mnist(batch_size):
    (X_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    X_train_normalized = ((X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') - 127.5) / 127.5)
    buffer_size = X_train.shape[0]
    X_train_ds = tf.data.Dataset.from_tensor_slices(X_train_normalized).shuffle(buffer_size).batch(batch_size)
    return X_train_ds, y_train