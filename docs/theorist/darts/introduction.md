# Differentiable Architecture Search

## Introduction

Neural Architecture Search refers to a family of methods for automating the discovery of useful neural network architectures. There are a number of methods to guide this search, such as evolutionary algorithms, reinforcement learning, or Bayesian optimization (for a recent survey of NAS search strategies, see Elsken, Metzen, & Hutter, 2019). However, most of these methods are computationally demanding due to the nature of the optimization problem: The search space of candidate computation graphs is high-dimensional and discrete. To address this problem, Liu et al. (2018) proposed **differentiable architecture search (DARTS)**, which relaxes the search space to become continuous, making architecture search amenable to gradient descent. 

DARTS has been shown to yield useful network architectures for image classification and language modeling that are on par with architectures designed by human researchers. AutoRA provides an adaptation of DARTS for automate the discovery of interpretable quantitative models to explain human information processing (Musslick, 2021).

## References

Liu, H., Simonyan, K., & Yang, Y. (2018). Darts: Differentiable architecture search. In *International Conference on Learning Representations*. arXiv: https://arxiv.org/abs/1806.09055

Elsken, T., Metzen, J. H., Hutter, F., et al. (2019). Neural architecture search: A survey. *JMLR*, 20(55), 1–21

Musslick, S. (2021). Recovering quantitative models of human information processing with differentiable architecture search. In *Proceedings of the 43rd Annual Conference of the Cognitive Science Society* (pp. 348–354). Vienna, AT. arXiv: https://arxiv.org/abs/2103.13939


