from style_transfer.base.base_iterator import BaseIterator
from tqdm import tqdm
import numpy as np


class Validatior(BaseIterator):
    def __init__(self, sess, model, config, logger):
        super().__init__(sess, model, config.validation_data_path, config.validation_style_data_path, config, logger)

    def iterate_epoch(self):
        losses = []
        try:
            while True:
                loss = self.iterate_step(self.handle)
                losses.append(loss)
        except Exception as _:
            pass
        feed_dict = {self.model.losses: sum(losses)/len(losses)}
        loss_summaries = self.sess.run([self.model.validation_summaries], feed_dict=feed_dict)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, loss_summaries)
        self.model.save(self.sess)

    def iterate_step(self, handle):
        feed_dict = {self.model.handle: handle, self.model.is_training: False}
        loss = self.sess.run(self.model.loss, feed_dict=feed_dict)
        return loss
