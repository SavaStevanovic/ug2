import tensorflow as tf

from style_transfer.model_setup.model_setup import ModelSetup
from style_transfer.iterators.trainer import Trainer
from style_transfer.iterators.validatior import Validatior
from style_transfer.utils.config import process_config
from style_transfer.utils.dirs import create_dirs
from style_transfer.utils.logger import Logger
from style_transfer.utils.utils import get_args

# tf.enable_eager_execution()


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = process_config("./configs/config.json")

    except Exception as _:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = ModelSetup(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, config, logger)
    # validatior = Validatior(sess, model, config, logger)
    # load model if exists
    model.load(sess)
    for _ in range(config.epohs_count):
        trainer.iterate_epoch()
        # validatior.iterate_epoch()


if __name__ == '__main__':
    main()
