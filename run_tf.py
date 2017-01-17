import os
import json
import tensorflow as tf
from pprint import pformat
from libs.dcgan_model import DCGAN
import logging
import argparse

logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument("--parameters_file",type=str, help="path to json file holding parameters")

def main(_):
    logging.info("running tensor flow start")
    args = parser.parse_args()
    with open(args.parameters_file) as data_file:
        data = json.load(data_file)
    params = {}
    for p in data:
        params[p['name']] = p['value']

    logging.info(pformat(params))
    with tf.Session() as sess:
        dcgan = DCGAN(sess,params)
        dcgan.train()

if __name__ == '__main__':
    tf.app.run()
