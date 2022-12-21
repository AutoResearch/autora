from typing import Mapping


class LazyDict(Mapping):
    """Inspired by https://gist.github.com/gyli/9b50bb8537069b4e154fec41a4b5995a"""

    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        func = self._raw_dict.__getitem__(key)
        return func()

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)
