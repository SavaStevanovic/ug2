import tensorflow as tf
from style_transfer.base.base_model import BaseModel


class StyleTransferModel(BaseModel):
    def __init__(self, config):
        self.residual_count = config.residual_count

    def make_model(self, input, is_training):
        with tf.variable_scope('network'):
            net = tf.layers.conv2d(input, 32, 7, padding='same')
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            net = tf.layers.conv2d(net, 64, 3, strides=2, padding='same')
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            net = tf.layers.conv2d(net, 128, 3, strides=2,  padding='same')
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            for _ in range(self.residual_count):
                net = self.residual_block(net, is_training)
            net = tf.keras.layers.UpSampling2D()(net)
            net = tf.layers.conv2d(net, 64, 3, strides=1, padding='same')
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            net = tf.keras.layers.UpSampling2D()(net)
            net = tf.layers.conv2d(net, 32, 3, strides=1, padding='same')
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            net = tf.layers.conv2d(net, 3, 9, activation=tf.nn.sigmoid, padding='same')
            return net

    def residual_block(self, net, is_training):
        orig = net
        net = tf.layers.conv2d(net, 128, 3, strides=1,  padding='same')
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 128, 3, strides=1,  padding='same')
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)
        return orig + net
