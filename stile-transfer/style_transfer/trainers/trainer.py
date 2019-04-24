from style_transfer.base.base_iterator import BaseIterator
from tqdm import tqdm
import numpy as np


class Trainer(BaseIterator):
    def __init__(self, sess, model, config, logger):
        super().__init__(sess, model, config.train_data_path, config, logger)

    def iterate_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        for i in loop:
            merged_summaries = self.iterate_step(self.handle)
            if i % self.config.logging_period:
                cur_it = self.model.global_step_tensor.eval(self.sess)
                self.logger.summarize(cur_it, merged_summaries)
        self.model.save(self.sess)

    def iterate_step(self, handle):
        feed_dict = {self.model.handle: handle, self.model.is_training: True}
        _, merged_summaries = self.sess.run([self.model.train_step, self.model.merged_summaries],
                                            feed_dict=feed_dict)
        return merged_summaries
