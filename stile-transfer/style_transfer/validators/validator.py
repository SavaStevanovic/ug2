from style_transfer.base.base_iterator import BaseIterator
from tqdm import tqdm
import numpy as np


class Validator(BaseIterator):
    def __init__(self, sess, model, config, logger):
        super().__init__(sess, model, config.validation_data_path, config, logger)

    def iterate_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for i in loop:
            loss = self.iterate_step(self.handle)
            losses.append(loss)

        feed_dict = {self.model.avg: sum(losses)/len(losses), self.model.is_training: False}
        avg_loss_summ = self.sess.run(self.model.avg_loss_summ, feed_dict=feed_dict)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, avg_loss_summ, 'test')
        self.model.save(self.sess)

    def iterate_step(self, handle):
        feed_dict = {self.model.handle: handle, self.model.is_training: False}
        _, loss = self.sess.run([self.model.loss], feed_dict=feed_dict)
        return loss
