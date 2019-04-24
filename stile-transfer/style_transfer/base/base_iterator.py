import tensorflow as tf
from style_transfer.data_loader.data_loader import DataLoader


class BaseIterator:
    def __init__(self, sess, model, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        self.handle = self.sess.run(DataLoader(config.train_data_path, config.batch_size).handle)

    def iterate_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def iterate_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
