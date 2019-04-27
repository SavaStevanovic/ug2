import tensorflow as tf
import os
from keras.applications import vgg16


class DataLoader:
    def __init__(self, path, batch_size):
        self.path = path
        self.images_paths = []
        self.fetch_image_paths(self.path)

        self.path_ds = tf.data.Dataset.from_tensor_slices(self.images_paths)
        self.path_ds = self.path_ds.repeat()
        self.path_ds = self.path_ds.shuffle(buffer_size=len(self.images_paths))
        self.path_ds = self.path_ds.map(self.load_and_preprocess_image)
        self.path_ds = self.path_ds.batch(batch_size)
        self.path_ds = self.path_ds.prefetch(buffer_size=1)

        self.iterator = self.path_ds.make_one_shot_iterator()
        self.handle = self.iterator.string_handle()

    def preprocess_image(self, image):
        image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        return tf.cast(vgg16.preprocess_input(image, mode='tf'), tf.float32)

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def fetch_image_paths(self, path):
        for d, dirs, f in os.walk(path):
            for file in f:
                if any([x in file.lower() for x in ['.jpg', '.png', '.jpeg']]):
                    self.images_paths.append(d+'/'+file)
            for dir in dirs:
                self.fetch_image_paths(d+'/'+dir)
