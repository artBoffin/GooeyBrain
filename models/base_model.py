__author__ = 'jasrub'

from __future__ import print_function

import os
import sys
import time

from models.parameter import Parameter
from models.util import log, error

import tensorflow as tf

def time_string():
    return time.strftime("%Y%m%d-%H%M%S")


class BaseModel:
    def __init__(self, sess, params):
        self.log("module ctor")
        self.run_name = params["run_name"] if "run_name" in params else "model%s"%time_string()
        self.checkpoint_dir = params["checkpoint_dir"] if "checkpoint_dir" in params else "./"
        self.report = {}

    def train(self):
        raise ValueError("Model did not override train()")

    def generate(self, event, data_dict):
        raise ValueError("Model did not override generate()")

    def log(self, msg):
        log(self.__class__.__name__ + " :", msg)

    def error(self, msg):
        error(self.__class__.__name__ + " :", msg)

    def save(self, sess, step):
        self.saver.save(sess,
                        os.path.join(self.checkpoint_dir, "%s.model" % self.run_name),
                        global_step=step)

    def load(self, sess):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    @staticmethod
    def parametersJSON(parameters):
        arr = []
        for p in parameters:
            arr.append(p.getJson())
        return arr
