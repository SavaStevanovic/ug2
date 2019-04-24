import tensorflow as tf

from style_transfer.data_loader.data_loader import DataGenerator
from style_transfer.model_setup.model_setup import ModelSetup
from style_transfer.trainers.example_trainer import ExampleTrainer
from style_transfer.utils.config import process_config
from style_transfer.utils.dirs import create_dirs
from style_transfer.utils.logger import Logger
from style_transfer.utils.utils import get_args

# tf.enable_eager_execution()


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = process_config("C:\\Pets\\configs\\config.json")

    except Exception as e:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    model = ModelSetup(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    for _ in range(config.epohs_count):
        trainer.train_epoch()


if __name__ == '__main__':
    main()
