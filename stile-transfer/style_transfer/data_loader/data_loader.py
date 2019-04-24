import tensorflow as tf
import os


class DataLoader:
    def __init__(self, path, batch_size):
        self.path = path
        self.images_paths = []
        for d, _, f in os.walk(self.path+'/images/'):
            for file in f:
                self.images_paths.append(d+'/images/'+file)

        self.path_ds = tf.data.Dataset.from_tensor_slices(self.images_paths)
        self.path_ds = self.path_ds.map(self.load_and_preprocess_image)
        self.path_ds = self.path_ds.shuffle(buffer_size=len(self.images_paths))
        self.path_ds = self.path_ds.repeat()
        self.path_ds = self.path_ds.batch(batch_size)
        self.path_ds = self.path_ds.prefetch(buffer_size=10)

        self.iterator = self.path_ds.make_one_shot_iterator()
        self.handle = self.iterator.string_handle()

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)
