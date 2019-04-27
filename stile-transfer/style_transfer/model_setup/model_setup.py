import tensorflow as tf
from style_transfer.base.base_model_setup import BaseModelSetup
from style_transfer.models.style_transfer_model import StyleTransferModel
from keras.applications import vgg16
import keras
from keras import backend as K
import numpy as np


class ModelSetup(BaseModelSetup):
    def __init__(self, config):
        super().__init__(config)
        self.init_saver()
        self.model = StyleTransferModel(config)
        self.setup_model()

    def setup_model(self):
        self.img_nrows = 224
        self.img_ncols = 224
        self.is_training = tf.placeholder(tf.bool)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.style_handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.handle, tf.float32, tf.TensorShape([None, self.img_nrows, self.img_ncols, 3]))
        style_iterator = tf.data.Iterator.from_string_handle(self.style_handle, tf.float32, tf.TensorShape([None, self.img_nrows, self.img_ncols, 3]))
        self.x = iterator.get_next()
        generated_image = self.model.make_model(self.x)
        generated_image_input = keras.layers.Input(tensor=generated_image)
        base_image = keras.layers.Input(tensor=self.x)
        style_reference_image = keras.layers.Input(tensor=style_iterator.get_next())

        # combine the 3 images into a single Keras tensor
        input_tensor = K.concatenate([base_image,
                                      style_reference_image,
                                      generated_image_input], axis=0)

        # build the VGG16 network with our 3 images as input
        # the model will be loaded with pre-trained ImageNet weights
        model = vgg16.VGG16(input_tensor=input_tensor,
                            weights='imagenet', include_top=False)
        # model.summary()
        print('Model loaded.')

        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        self.loss = K.variable(0.0)
        layer_features = outputs_dict['block5_conv2']
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        self.loss += self.config.content_weight * self.content_loss(base_image_features, combination_features)

        feature_layers = ['block1_conv1', 'block2_conv1',
                          'block3_conv1', 'block4_conv1',
                          'block5_conv1']
        for layer_name in feature_layers:
            layer_features = outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self.style_loss(style_reference_features, combination_features)
            self.loss += (self.config.style_weight / len(feature_layers)) * sl
        self.loss += self.config.total_variation_weight * self.total_variation_loss(generated_image_input)

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "network")
        [print(x) for x in train_vars]
        self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, global_step=self.global_step_tensor, var_list=train_vars)

        with tf.variable_scope("train_summaries"):
            tf.summary.image(
                'generated_image',
                generated_image
            )

            tf.summary.image(
                'style_reference_image',
                style_reference_image
            )

            tf.summary.image(
                'base_image',
                base_image
            )

            tf.summary.scalar(
                'loss',
                self.loss
            )

            self.train_summaries = tf.summary.merge_all(scope="train_summaries")

        with tf.variable_scope("validation_summaries"):
            losses = tf.placeholder(tf.float32, shape=[])
            tf.summary.scalar(
                'losses',
                losses
            )

            self.validation_summaries = tf.summary.merge_all(scope="validation_summaries")

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def gram_matrix(self, x):
        assert K.ndim(x) == 3
        if K.image_data_format() == 'channels_first':
            features = K.batch_flatten(x)
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(self, style, combination):
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.img_nrows * self.img_ncols
        return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def content_loss(self, base, combination):
        return K.sum(K.square(combination - base))

    def total_variation_loss(self, x):
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(
                x[:, :, :self.img_nrows - 1, :self.img_ncols - 1] - x[:, :, 1:, :self.img_ncols - 1])
            b = K.square(
                x[:, :, :self.img_nrows - 1, :self.img_ncols - 1] - x[:, :, :self.img_nrows - 1, 1:])
        else:
            a = K.square(
                x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, 1:, :self.img_ncols - 1, :])
            b = K.square(
                x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, :self.img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))
