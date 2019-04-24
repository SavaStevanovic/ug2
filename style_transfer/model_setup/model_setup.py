import tensorflow as tf
from style_transfer.models.style_transfer_model StyleTransferModel 

class ModelSetup:
    def __init__(self, config):
        self.learning_rate=config.learning_rate
        self.model=StyleTransferModel(config)
    
    def setup_model(self):
        #add dataset api integration
        self.x= iterator.get_next()
        self.output=self.model.make_model(self.x)
        #add vgg for loss calculation
        self.loss=self.style_loss+ self.content_loss
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        tf.summary.image('output', self.output)
        tf.summary.scalar('style_loss', self.style_loss)
        tf.summary.scalar('content_loss', self.content_loss)
        tf.summary.scalar('loss', self.loss)