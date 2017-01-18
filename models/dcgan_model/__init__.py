# This model is from https://github.com/carpedm20/DCGAN-tensorflow"
from models.dcgan_model.main import DCGAN


def get_model(params):
    return DCGAN(params)


def get_parameters():
    return DCGAN.parametersJSON(DCGAN.parameters)