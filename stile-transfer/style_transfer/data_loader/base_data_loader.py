import tensorflow as tf

class BaseDataLoader:
    def __init__(self, config):
        self.path=config.path
        self.images_paths=[]
        for d,_,f in os.walk(self.path+'/images/'):
            for file in f:
                self.images_paths.append(self.path+'/images/'+file)
        
        self.path_ds = tf.data.Dataset.from_tensor_slices(self.images_paths)
        self.path_ds = self.path_ds.map(load_and_preprocess_image)
        self.path_ds = self.path_ds.shuffle(buffer_size=len(self.images_paths))
        self.path_ds = self.path_ds.repeat()
        self.path_ds = self.path_ds.batch(config.BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches, in the background while the model is training.
        self.path_ds = self.path_ds.prefetch(buffer_size=10)
        self.iterator=dataset.make_one_shot_iterator()
        self.iterator_handle=self.iterator.

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)


