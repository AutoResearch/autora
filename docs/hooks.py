import os
from urllib.parse import quote


def basename(path):
    return os.path.basename(path)


def on_env(env, config, files, **kwargs):
    env.filters["basename"] = basename
    env.filters["quote"] = quote
    return env
