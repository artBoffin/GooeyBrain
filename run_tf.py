import os
import json
import tensorflow as tf
from pprint import pprint
from libs.gan_model import GAN
from libs.dcgan_model import DCGAN

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--parameters_file",type=str, help="path to json file holding parameters")

def main(_):
    args = parser.parse_args()
    with open(args.parameters_file) as data_file:
        data = json.load(data_file)
    params = {}
    for p in data:
        params[p['name']] = p['value']

    pprint(params)
    with tf.Session() as sess:
        dcgan = DCGAN(sess,params)
        dcgan.train()

if __name__ == '__main__':
    tf.app.run()
