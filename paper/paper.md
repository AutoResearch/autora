---
title: 'AutoRA: Automated Research Assistant for Closed-Loop Empirical Research'
tags:
  - Python
  - automated scientific discovery
  - symbolic regression
  - active learning
  - closed-loop behavioral science
authors:
  - name: Sebastian Musslick
    orcid: 0000-0002-8896-639X
    affiliation: "1, 2"
    corresponding: true
  - name: Benjamin Andrew
    affiliation: 1
  - name: Chad C. Williams
    orcid: 0000-0003-2016-5614
    affiliation: 1
  - name: Joshua T. S. Hewson
    affiliation: 1
  - name: Star Li
    orcid: 0009-0009-7849-1923
    affiliation: 3
  - name: Ioana Marinescu
    affiliation: 4
  - name: Marina Dubova
    orcid: 0000-0001-5264-0489
    affiliation: 5
  - name: George T. Dang
    affiliation: 1
  - name: Younes Strittmatter
    orcid: 0000-0002-3414-2838
    equal-contrib: true
    affiliation: 1
  - name: John G. Holland
    orcid: 0000-0001-6845-8657
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: Brown University, USA
    index: 1
  - name: Osnabrück University, Germany
    index: 2
  - name: University of Chicago, USA
    index: 3
  - name: Princeton University, USA
    index: 4
  - name: University of Indiana, USA
    index: 5
date: 22 May 2024
bibliography: paper.bib

---

# Summary

Automated Research Assistant (`autora`) is a Python package for automating and integrating empirical research processes, such as experimental design, data collection, and model discovery. With this package, users can define an empirical research problem and specify the methods they want to employ for solving it. `autora` is designed as a declarative language in that it provides a vocabulary and set of abstractions to describe and execute scientific processes and to integrate them into a closed-loop system for scientific discovery. The package interfaces with computational approaches to scientific discovery, including `scikit-learn` estimators for scientific model discovery, `sweetpea` for automated experimental design, `firebase_admin` for automated behavioral data collection, and `autodoc` for automated documentation of the empirical research process. While initially developed for the behavioral sciences, `autora` is designed as a general framework for closed-loop scientific discovery, with applications in other empirical sciences. Use cases of `autora` include the execution of closed-loop empirical studies [@musslick2024], the benchmarking of scientific discovery algorithms [@hewson_bayesian_2023], and the implementation of metascientific studies [@musslick_evaluation_2023]. 

# Statement of Need
The pace of empirical research is constrained by the rate at which scientists can alternate between the design and execution of experiments, on the one hand, and the derivation of scientific knowledge, on the other hand. However, attempts to increase this rate can compromise scientific rigor, leading to lower quality of formal modeling, insufficient documentation, and non-replicable findings. `autora` aims to surmount these limitations by formalizing the empirical research process and automating the generation, estimation, and empirical testing of scientific models. By providing a declarative language for empirical research, `autora` offers greater transparency and rigor in empirical research while accelerating scientific discovery. While existing scientific computing packages solve individual aspects of empirical research, there is no workflow mechanic for integrating them into a single pipeline, e.g., to enable closed-loop experiments. `autora` offers such a workflow mechanic, integrating Python packages for automating specific aspects of the empirical research process.

![The `autora` framework. (A) `autora` workflow, as applied in a behavioral research study. `autora` implements components (colored boxes; see text) that can be integrated into a closed-loop discovery process. Workflows expressed in `autora` depend on modules for individual scientific tasks, such as designing behavioral experiments, executing those experiments, and analyzing collected data. (B) `autora`’s components acting on the state object. The state object maintains relevant scientific data, such as experimental conditions X, observations Y, and models, and can be modified by `autora` components. Here, the cycle begins with an experimentalist adding experimental conditions $x_1$ to the state. The experiment runner then executes the experiment and collects corresponding observations $y_1$. The cycle concludes with the theorist computing a model that relates $x_1$ to $y_1$.\label{fig:overview}](figure.png)

# Overview and Components
The `autora` framework implements and interfaces with components automating different phases of the empirical research process (\autoref{fig:overview}A). These components include *experimentalists* for automating experimental design, *experiment runners* for automating data collection, and *theorists* for automating scientific model discovery. To illustrate each component, we consider an exemplary behavioral research study (cf. \autoref{fig:overview}) that examines the probability of human participants detecting a visual stimulus as a function of its intensity.

*Experimentalist* components take the role of a research design expert, determining the next iteration of experiments to be conducted. Experimentalists are functions that identify experimental conditions which can be subjected to measurement by experiment runners, such as different levels of stimulus intensity. To determine these conditions, experimentalists may use information about candidate models obtained from theorist components, experimental conditions that have already been probed, or respective observations. The `autora` framework offers various experimentalist packages, each for determining new conditions based on, for example, novelty, prediction uncertainty, or model disagreement [@musslick_evaluation_2023; @dubova_against_2022]. 

*Experiment runner* components correspond to research technicians collecting data from an experiment. They are implemented as functions that accept experimental conditions as input (e.g., a `pandas` dataframe with columns representing different experimental variables) and produce collected observations as output (e.g., a `pandas` dataframe with columns representing different experimental variables along with corresponding measurements). `autora` (4.0.0) provides experiment runners for two types of automated data collection: real-world and synthetic. Real-world experiment runners include interfaces for collecting data in the real world. For example, the `autora` framework offers experiment runners for automating the data collection from web-based experiments for behavioral research studies [@musslick2024]. In the behavioral experiment described above, an experiment runner may set up a web-based experiment that measures the probability of human participants detecting visual stimuli of different intensities. These runners interface with external components including recruitment platforms (e.g, Prolific; @palan_prolific_2018) for coordinating the recruitment of participants, databases (e.g., Google Firestore) for storing collected observations, and web servers for hosting the experiments (e.g., Google Firebase). Synthetic experiment runners specify the data-generating process and collect observations from it. For example, ``autora-synthetic`` implements established models of human information processing (e.g, for perceptual discrimination) and conducts experiments on them. These synthetic experiments serve multiple purposes, such as testing and benchmarking `autora` components before applying them in the real-world [@musslick2024] or conducting computational metascience studies [@musslick_evaluation_2023]. 

*Theorist* components embody the role of a computational scientist, employing modeling techniques to find a model that best characterizes, predicts, and/or explains the study’s observations. Theorists may identify different types of scientific models (e.g., statistical, mathematical, or computational) implemented as `scikit-learn` estimators [@pedregosa2011scikit]. In case of the behavioral research study, a model may correspond to a psychophysical law relating stimulus intensity to the probability of detecting the stimulus. `autora` provides interfaces for various equation discovery methods that are implemented as `scikit-learn` estimators, including deep symbolic regression [@petersen2021deep; @landajuela_unified_2022], `PySR` [@cranmer_discovering_2020], and the Bayesian Machine Scientist [@guimera_bayesian_2020; @hewson_bayesian_2023]. A model is generated by fitting experimental data. Accordingly, theorists take as input a `pandas` dataframe specifying experimental conditions (instances of experimental variables) along with corresponding observations to fit a respective model. The model can then be used to generate predictions, e.g., to inform the design of a subsequent experiment.

# Design Principles and Packaging
`autora` was designed as a general framework aimed at democratizing the automation of empirical research across the scientific community. Key design decisions were: 1) using a functional paradigm for the components and 2) splitting components across Python namespace packages. 

Each component is a function that operates on immutable "state objects" which represent data from an experiment (\autoref{fig:overview}B), such as proposed experimental conditions, corresponding observations (represented as a `pandas` dataframe), and scientific models (represented as a list of `scikit-learn` estimators). Data produced by each component can be seen as additions to the existing data stored in the state. Thus, each component $C$ takes in existing data in a state $S$, adds new data $\Delta S$, and returns an updated state $S'$,

$$
S' = C(S) = S + \Delta S.
$$

Accordingly, the components share their interface – every component loads data from and saves data to state objects, so they can be ordered arbitrarily, and adding a new component is as simple as implementing a new function or `scikit-learn`-compatible estimator and wrapping it with a utility function provided in `autora-core`. State immutability allows for easy parallelism and reproducibility (so long as the components themselves have no hidden state).

The `autora` framework presumes that each component is distributed as a separate package but in a shared namespace, and that `autora-core` – which provides the state – has very few dependencies of its own. For users, separate packages minimize the time and storage required for an install of an `autora` project. For contributors, they reduce incidence of dependency conflicts (a common problem for projects with many dependencies) by reducing the likelihood that the library they need has an existing conflict in `autora`. It also allows contributors to independently develop and maintain modules, fostering ownership of and responsibility for their contributions. External contributors can request to have packages vetted and included as an optional dependency in the `autora` package.

# Acknowledgements
The AutoRA framework is developed and maintained by members of the Autonomous Empirical Research Group. S. M., B. A., C. C. W., J. T. S. H., and Y. S. were supported by the Carney BRAINSTORM program at Brown University. S. M. also received support from Schmidt Science Fellows, in partnership with the Rhodes Trust. The development of auxiliary packages for AutoRA, such as `autodoc`, is supported by the Virtual Institute for Scientific Software. The AutoRA package was developed using computational resources and services at the Center for Computation and Visualization, Brown University.

# References






