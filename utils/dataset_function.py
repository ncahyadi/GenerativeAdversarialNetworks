import tensorflow as tf
import pathlib

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

# dataset efrosgans dataset
def get_efrosgans_ds(ds_index):
    dataset_names = ["cityscapes", "edges2handbags", "edges2shoes", "facades", "maps", "night2day"]
    dataset_name = dataset_names[ds_index]
    _URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'
    path_to_zip = tf.keras.utils.get_file(
    fname=f"{dataset_name}.tar.gz",
    origin=_URL,
    extract=True)
    path_to_zip  = pathlib.Path(path_to_zip)
    PATH = path_to_zip.parent/dataset_name
    return PATH

def load_efrosgans(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

## utils in efronsgans
def resize_efrosgans(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop_efronsgans(input_image, real_image, img_heigth, img_width):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, img_heigth, img_width, 3])
    return cropped_image[0], cropped_image[1]

def normalize_efronsgans(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image