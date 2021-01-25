from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')


# PRIMITIVES = [
#     'none',
#     'relu',
# ]


# PRIMITIVES = [
#     'none',
#     'add',
#     'subtract',
#     'linear',
#     'sigmoid',
#     'exp',
#     'relu',
# ]

PRIMITIVES = [
    'none',
    'add',
    'subtract',
    'linear',
    'lin_sigmoid',
    'mult',
    'lin_relu',
]
