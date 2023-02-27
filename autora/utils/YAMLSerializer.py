import yaml


def dump(data, file):
    as_string = yaml.dump(data, Dumper=yaml.Dumper)
    file.write(as_string)
    return


def load(file):
    result = yaml.load(file, Loader=yaml.Loader)
    return result
