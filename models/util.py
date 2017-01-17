import logging


logger = logging.getLogger('GooeyBrain')
logger.setLevel(logging.DEBUG)
stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
stream.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream)


def to_str(objs):
    return ' '.join([str(obj) for obj in objs])


def log(*objs):
    logger.info(to_str(objs))


def error(*objs):
    logger.error(to_str(objs))
