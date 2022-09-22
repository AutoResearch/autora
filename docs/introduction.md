# Bayesian Machine Scientist

## Introduction

Symbolic regression (SR) refers to a class of algorithms that search for interpretable symbolic expressions which
capture relationships within data. More specifically, SR attempts to find compositions of simple functions
that accurately map independent variables to dependent variables within a given dataset. SR was traditionally tackled
through genetic programming, wherein evolutionary algorithms mutated and crossbred equations billions of 
times in search a match. Problems with genetic programming, however, stem from its inherent search constraints as well
as its reliance upon heuristics and domain knowledge to balance goodness of fit and model complexity. To address these
problems, Guimerà et. al (2020) proposed a Bayesian Machine Scientist, which combines i) a Bayesian approach that
specifies informed priors over expressions and computes their respective posterior probabilities given the data at hand,
and ii) a Markov chain Monte Carlo (MCMC) algorithm that samples from the posterior over expressions to explore the 
space of possible symbolic expressions.
