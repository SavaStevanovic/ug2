import tensorflow as tf
import os


class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        summarizer_names = ["train", "test"]
        self.summarizers = {sn: tf.summary.FileWriter(os.path.join(self.config.summary_dir, sn)) for sn in summarizer_names}

    # it can summarize scalars and images.
    def summarize(self, step, summary, summarizer="train", scope=""):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.summarizers[summarizer]
        summary_writer.add_summary(summary, step)
        summary_writer.flush()
