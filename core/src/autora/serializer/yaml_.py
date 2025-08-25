import yaml


def dump(data, file):
    yaml.dump(data, file, Dumper=yaml.Dumper)
    return


def load(file):
    data = yaml.load(file, Loader=yaml.Loader)
    return data


def dumps(data):
    string = yaml.dump(data, Dumper=yaml.Dumper)
    return string


def loads(string):
    data = yaml.load(string, Loader=yaml.Loader)
    return data
