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

# PRIMITIVES = [    # success Weber
#     'none',
#     'add',
#     'subtract',
#     'linear',
#     'lin_sigmoid',
#     'mult',
#     'lin_relu',
# ]

# PRIMITIVES = [  # partial success weber
#     'none',
#     'add',
#     'subtract',
#     'linear',
#     'sigmoid',
#     'lin_sigmoid',
#     'mult',
#     'relu',
#     'lin_relu',
# ]

PRIMITIVES = [  # success LCA
    'none',
    'add',
    'subtract',
    'linear',
    'sigmoid',
    'mult',
    'exp',
    'relu',
    'lin_relu',
    'lin_sigmoid',
]