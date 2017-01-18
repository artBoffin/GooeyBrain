__author__ = 'jasrub'

import argparse
import json
from pprint import pformat

import tensorflow as tf

from models.dcgan_model.main import DCGAN
from models.util import log

parser = argparse.ArgumentParser()
parser.add_argument("--parameters_file",type=str, help="path to json file holding parameters")
parser.add_argument("--train",type=bool,
                    help="True if we want to train a model, flase to load a model and generate images")


def main(_):
    args = parser.parse_args()
    with open(args.parameters_file) as data_file:
        data = json.load(data_file)
    params = {}
    for p in data:
        params[p['name']] = p['value']

    log ("params sent: %s"%pformat(params))
    with tf.Session() as sess:
        dcgan = DCGAN(params)
        if (args.train):
            dcgan.train(sess)
        else:
            num_samples = 640
            dcgan.generate(sess, num_samples)

if __name__ == '__main__':
    tf.app.run()
