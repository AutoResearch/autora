import yaml


def dump(data, file):
    yaml.dump(data, file, Dumper=yaml.Dumper)
    return


def load(file):
    result = yaml.load(file, Loader=yaml.Loader)
    return result
