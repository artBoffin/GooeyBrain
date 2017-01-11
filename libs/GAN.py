import argparse
import copy
import os
import time
from glob import glob
from types import *

import numpy as np
import tensorflow as tf

from libs.utils import *


class Parameter(object):
    def __init__(self, name, value, p_type=None,
                 p_min=0, p_max=0, step=0, description="",
                 size_change=False, list_type=None, is_path=False):
        self.name = name
        self.value = value
        self.type = p_type if p_type else type(value)
        self.min = p_min
        self.max = p_max
        self.step = step
        self.description = description
        self.size_change = size_change
        self.list_type = list_type
        self.is_path = is_path

    def getJson(self):
        json = {
            'name': self.name,
            'type': self.type.__name__,
            'value': self.value,
            'min': self.min,
            'max': self.max,
            'step': self.step,
            'description': self.description,
            'size_change': self.size_change,
            'is_path': self.is_path
        }
        if self.type==list:
            json['list_type'] = self.list_type.__name__
        return json


class GAN(object):
    parameters = [
        Parameter("files", "/train", str, description="path to files", is_path=True),
        Parameter("learning_rate", 0.0001, p_type=float, p_min=0.00000001, p_max=0.01, step=0.00000001,
                  description="leraning rate"),
        Parameter("batch_size", 64, p_type=int, p_min=2, p_max=500, step=1, description="batch size"),
        Parameter("n_epochs", 25, int, 1, 500, 1, "number of epochs"),
        Parameter("n_examples", 10, int, 1, 200, 1, "number of examples"),
        Parameter("input_shape", [64, 64, 3], p_type=list, description="shape of input image, [h, w, c]",
                  size_change=False, list_type=int),
        Parameter("crop_shape", [64, 64, 3], p_type=list, description="output shape / shape of cropped input",
                  size_change=False, list_type=int),
        Parameter("crop_factor", 0.8, float, 0.01, 1, 0.01, "percentage of image to crop (zoom in)"),
        Parameter("filters",
                  [{'dim': 512, 'size': 3}, {'dim': 256, 'size': 3}, {'dim': 128, 'size': 3}, {'dim': 64, 'size': 3}],
                  p_type=list, description="dimensions of filters in the network", size_change=True, list_type=dict),
        Parameter("n_hidden", 0, int, 0, 200, 1, description="hidden layer"),
        Parameter("n_code", 0, int, 1, 300, 1, "can only be >1 if variational = true"),
        Parameter("convolutional", True, bool, description="is convolutional network (true for DCGAN)"),
        Parameter("variational", True, bool, description="is variational"),
        Parameter("activation", "tf.nn.relo", str, description="activation function"),
        Parameter("save_path", "/tmp", str, description="path to sve model files", is_path=True),
        Parameter("run_name", "gan_%s" % time.strftime("%Y%m%d-%H%M%S"), str,
                  description="name of this run for creating relevant folders")
    ]

    def __init__(self, params):

        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.n_epochs = params['n_epochs']
        self.n_examples = params['n_examples']
        self.input_shape = params['input_shape']
        self.crop_shape = params['crop_shape']
        self.crop_factor = params['crop_factor']
        self.n_filters = [x['dim'] for x in params['filters']]
        self.n_hidden = params['n_hidden'] if params['n_hidden'] > 0 else None
        self.n_code = params['n_code'] if params['n_code'] > 0 else None
        self.convolutional = params['convolutional']
        self.variational = params['variational']
        self.filter_sizes = [x['size'] for x in params['filters']]
        self.activation = eval(params['activation'])
        self.svae_path = params['save_path']
        self.run_name = params['run_name']

        self.TENSORBOARD_DIR = os.path.join(self.save_path, self.run_name, 'logs')
        self.MODEL_DIR = os.path.join(self.save_path, self.run_name, 'model')
        self.CHECKPOINT_PATH =  os.path.join(self.save_path, self.run_name, 'ckpt')

        for path in [self.TENSORBOARD_DIR, self.MODEL_DIR, self.CHECKPOINT_PATH]:
            if not os.path.exists(path):
                os.mkdir(path)

        self._build_model()

    def _build_model(self):
        self.hello = "hi"
        return 'hi!'


def parse_arguments(parameters):
    parser = argparse.ArgumentParser(description='Calling GAN model')
    for p in parameters:
        name = '--%s' % p.name
        nargs = '?'
        if type == list:
            nargs = '+'
            if not p.size_change:
                nargs = str(len(p.value))
        parser.add_argument(name, type=p.type, nargs=nargs, deafult=p.value,
                            help=p.description)
    return parser
