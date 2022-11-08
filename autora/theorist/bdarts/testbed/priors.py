from bdarts_operations import OPS

# PARAMETERS

# used for initializing guide
a_init_loc = 1.0
b_init_loc = 0.0

# used for coefficient priors
a_loc_prior = 1.0
a_scale_prior = 1.0
b_loc_prior = 0.0
b_scale_prior = 0.1

# used for architecture priors
w_loc_prior = 0.0
w_scale_prior = 1.0

# INITIAL VALUES FOR ARCHITECTURE GUIDE

guide_arch_init = dict()

for op in OPS.keys():
    guide_arch_init["w_" + op] = 0.0
# guide_arch_init["w_linear_tanh"] = -10.

# INITIAL VALUES FOR COEFFICIENT GUIDE

guide_coeff_init = dict()
guide_coeff_init["b"] = 0.0
for op in OPS.keys():
    if "linear" in op:
        guide_coeff_init["a_" + op] = a_init_loc
        guide_coeff_init["b_" + op] = b_init_loc

# PRIORS FOR ARCHITECTURE MODEL

arch_priors = dict()

for primitive in OPS:
    arch_priors["w_" + primitive + "_auto_loc"] = w_loc_prior
    arch_priors["w_" + primitive + "_auto_scale"] = w_scale_prior
# arch_priors["w_none_auto_scale"] = 0.1
# arch_priors["w_none_auto_loc"] = 10.

# PRIORS FOR COEFFICIENT MODEL

coeff_priors = dict()
for primitive in OPS:
    if "linear" in primitive:
        coeff_priors["a_" + primitive + "_auto_loc"] = a_loc_prior
        coeff_priors["a_" + primitive + "_auto_scale"] = a_scale_prior
        coeff_priors["b_" + primitive + "_auto_loc"] = b_loc_prior
        coeff_priors["b_" + primitive + "_auto_scale"] = b_scale_prior
coeff_priors["b_auto_loc"] = 0.0
