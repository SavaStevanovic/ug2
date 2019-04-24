import tensorflow as tf
from style_transfer.base.base_model_setup import BaseModelSetup
from style_transfer.models.style_transfer_model import StyleTransferModel


class ModelSetup(BaseModelSetup):
    def __init__(self, config):
        super().__init__(config)
        self.build_model()
        self.init_saver()
        self.model = StyleTransferModel(config)

    def setup_model(self):
        self.is_training = tf.placeholder(tf.bool)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, [tf.float32], tf.shape([None, 224, 224, 3]))
        self.x = iterator.get_next()
        self.output = self.model.make_model(self.x)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
