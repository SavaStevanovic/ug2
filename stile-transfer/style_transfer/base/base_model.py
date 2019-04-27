import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config

    def make_model(self, input):
        raise NotImplementedError
