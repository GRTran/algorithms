# algorithms
This repository contains sub-folders, containing various algorithmic and numerical solutions using a variety of methods, to example problems. The present subfolders are listed as follows:

- Stochastics

# Stochastics

At present contains a numerical method for solving a stochastic differential equation (SDE) using the Euler-Maruyama method. The stochastic numerical method is applied to an Ornstein-Uhlenbeck process that is also a Markov process. The stochastic diffusion (volatility) term is sampled from a standard normal distribution.

An application of a Monte Carlo numerical solver is also described. The file entitled "mc-beam-walk.py" solves the child beam walk problem using Monte Carlo simulation. The numerical results are compared to analytical results, calculated using a combination of conditional probability for mutually exclusive events as well as the binomial distribution function. 