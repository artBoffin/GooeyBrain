# This model is from
from models.dcgan_model.main import DCGAN


def get_model(params):
    return DCGAN(params)


def get_parameters():
    return DCGAN.parametersJSON(DCGAN.parameters)