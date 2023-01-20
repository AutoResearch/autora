from autora_bsr.utils.funcs import (
    grow,
    prune,
    de_transform,
    transform,
    reassign_op,
    reassign_feat
)


def test_mcmc_grow(**hyper_params):
    grow(**hyper_params)


def test_mcmc_prune(**hyper_params):
    prune(**hyper_params)


def test_mcmc_de_transform(**hyper_params):
    de_transform(**hyper_params)


def test_mcmc_transform(**hyper_params):
    transform(**hyper_params)


def test_mcmc_reassign_op(**hyper_params):
    reassign_op(**hyper_params)


def test_mcmc_reassign_feat(**hyper_params):
    reassign_feat(**hyper_params)
