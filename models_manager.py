import pkgutil
import os
import pprint
import models

__author__ = 'jasrub'

models_dict = {}
parameters_dict = {}

def init_models():
    """Initialize the modules and subscription dicts"""
    for loader, name, is_pkg in pkgutil.iter_modules(path=[os.path.dirname(models.__file__)]):
        if not is_pkg:
            continue  # ignore anything that is not a python package

        # import model and put in modules dictionary
        model_package = __import__('models.' + name, fromlist=[''])
        model_getter = model_package.get_model
        models_dict[name] = model_getter
        parameters_dict[name] = model_package.get_parameters()

    # Printing models dict for debug
    # pp = pprint.PrettyPrinter(indent=4)
    # print "models_dict:"
    # pp.pprint(models_dict)
    # print "parameters_dict:"
    # pp.pprint(parameters_dict)
