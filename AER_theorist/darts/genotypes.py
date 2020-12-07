from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')


# PRIMITIVES = [
#     'none',
#     'relu',
# ]


PRIMITIVES = [
    'none',
    'add',
    'subtract',
    'linear',
    'sigmoid',
    'exp',
    'relu',
]
