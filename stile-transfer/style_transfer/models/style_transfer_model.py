import tensorflow as tf

class StyleTransferModel:
    def __init__(self, config):
        self.residual_count= config.residual_count
        

    def make_model(self, input):
        net= tf.layers.conv2d(input, 32, 7,activation=tf.nn.relu,  padding='same')
        net= tf.layers.conv2d(net, 64,3,activation=tf.nn.relu, strides=2, padding='same')
        net= tf.layers.conv2d(net, 128,3,activation=tf.nn.relu, strides=2,  padding='same')
        for _ in range(self.residual_count):
            net= self.residual_block(net)
        net = tf.layers.conv2d_transpose(net , 64 , 3 , activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d_transpose(net , 32 , 3 , activation=tf.nn.relu, padding='same')
        net = tf.layers.conv2d(net , 3 , 9 , activation=tf.nn.relu, padding='same')
        return net
        
    def residual_block(self, net):
        orig=net
        net= tf.layers.conv2d(net, 128,3,activation=tf.nn.relu, strides=2,  padding='same')
        net= tf.layers.conv2d(net, 128,3,activation=tf.nn.relu, strides=2,  padding='same')
        return orig + net