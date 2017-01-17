import json


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