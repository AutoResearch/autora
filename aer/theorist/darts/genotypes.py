from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat")

# SET OF AVAILABLE PRIMITIVES:
# "none",
# "add",
# "subtract",
# "linear",
# "sigmoid",
# "mult",
# "exp",
# "relu",
# "softplus",
# "softminus",
# 'lin_relu',
# 'lin_sigmoid',


PRIMITIVES = [
    "none",
    "add",
    "subtract",
    "linear",
    "lin_sigmoid",
    "mult",
    "lin_relu",
]
