'''
Author: Gregory Jones
Date: 22-11-2022
Description: This script provides a MC solution to the following problem. A child walks across a beam. 5 steps are required to reach the end of the beam. The probability of the child taking a successful single step is P(S) = 0.9 and P(F) = 0.1 where the sample space for a single step is S = success and F = fail. What is the probability that the child reaches the end of the beam exactly 5 times out of 10 tries.

Classical solution:
The probability that the child reaches the end of the beam is P(S)^5 = 0.9^5. This value can be multiplied by 10 to evaluate the average number of times that a child reaches the end out of 10 attempts. Now, to establish the probability that the child reaches the end exactly 5 out of 10 times we have a binomial distribution where p = 0.9^5 and q = 1-0.9^5
The result for exactly 5 times is therefore P(5S) = n! / k!(n-k)! p^k q^(n-k) = 0.208346

Monte Carlo solution:
As for the Monte Carlo solution, we simply create 5 random numbers between 0 and 1, each number corresponding to a single step. If all of the steps are above 0.1, then the attempt is a success, otherwise it is a failure. The next stage is to perform a set of 10 attempts and tally the number of successes. A large number of sets can be modelled with the number of successes aggregated. We may then simply divide the number of times the child reached the end 5 times divided by the total number of samples. This should equal the above analytical solution.
'''
import matplotlib.pyplot as plt
import random
import math
import statistics
plt.style.use('dark_background')
random.seed(5)

def successful_walk(p: float, steps: int = 5) -> bool:
    '''
    A walk along the beam is evaluated by sampling 'steps'(default 5) points between 0 and 1, the walk is successful if all are below the probability that a single successful step is taken.
    
    p (float) : The probability that a the child successfully takes a single step.
    steps (int) : The number of steps required to reach the end.
    return (bool) : True if the child reached the end of the beam, false otherwise.
    '''
    samples = [random.random() for i in range(steps)]
    if any(map(lambda x: x>p, samples)): return False
    return True

def perform_realisation(n: int, p: int, steps: int = 5) -> int:
    '''
    Performs a set of 'n' realisations and returns the number of successful times the child reached the end of the beam.
    
    k (int) : This is the exact number of successful attempts that we want to know the probability of
    n (int) : The number of total attempts for which the number of successful attempts are to be obtained (k<n).
    p (float) : The probability that a the child successfully takes a single step.
    steps (int) : The number of steps required to reach the end.
    
    return (int) : The number of successful walks across the beam out of n
    '''
    return sum(1 if successful_walk(p) else 0 for i in range(n))

def perform_epoch(k: int, n: int, p: float, n_realisations: int=1000, steps: int = 5) -> dict:
    '''
    This function performs one epoch of the beam-walk problem which consists of a certain number of realisations. Note that each realisation consists of 10 walks across the beam
    
    k (int) : This is the exact number of successful attempts that we want to know the probability of
    n (int) : The number of total attempts for which the number of successful attempts are to be obtained (k<n).
    p (float) : The probability that a the child successfully takes a single step.
    steps (int) : The number of steps required to reach the end.
    
    return (dict[int: int]): A dictionary containing the number of successes for each number of possible attempts. The sample space here is such that the total probability of each outcome is unity.
    '''
    # Create and empty dictionary that contains tallied results
    tally = {i: 0 for i in range(n+1)}
    # Perform epoch and calculate probability
    for i in range(n_realisations):
        tally[perform_realisation(n, p)] +=1
    return tally

def perform_full_mc_simulation(k: int, n: int, p: float, n_realisations: int=1000, n_epochs: int = 100, steps: int = 5) -> dict:
    '''
    This is a full MC simulation for the beam-walk problem. Although the number of successes simulated by each epoch will follow a binomial distribution, the mean of the epochs will flow a normal distribution through the Central Limit Theorem (CLT).
    
    k (int) : This is the exact number of successful attempts that we want to know the probability of
    n (int) : The number of total attempts for which the number of successful attempts are to be obtained (k<n).
    p (float) : The probability that a the child successfully takes a single step.
    steps (int) : The number of steps required to reach the end.
    
    return (dict[int: list[int]]): A dictionary containing the number of successes for each number of possible attempts. The sample space here is such that the total probability of each outcome is unity.
    '''
    # Create an empty dictionary that contains a list of epoch results
    tally = {i: [] for i in range(n+1)}
    # Perform set of MC simulations
    for i in range(n_epochs):
        tmp = perform_epoch(k, n, p, n_realisations)
        for key, value in zip(tmp.keys(), tmp.values()):
            tally[key] += [value/n_realisations]
    return tally

def analytical_solution(k: int, n: int, p: float, steps:int = 5):
    '''
    This is an analytical solution of the beam walk problem using conditional probability coupled with some binomial theorem analysis.
    
    k (int) : This is the exact number of successful attempts that we want to know the probability of
    n (int) : The number of total attempts for which the number of successful attempts are to be obtained (k<n).
    p (float) : The probability that a the child successfully takes a single step.
    steps (int) : The number of steps required to reach the end.
    
    return (float) : The probability that an exact number of successes occur out of some total number of attempts.
    '''
    return math.comb(n, k) * (p**(steps))**k * (1.-p**(steps))**(n-k)
    
if __name__=="__main__":
    # Set parameters
    exact_sucesses = 5; total_attempts = 10; prob_step_success = 0.9; n_realisations = 10000; n_epochs=100
	
    # Run the analysis for a single epoch
    epoch_result = perform_epoch(exact_sucesses, total_attempts, prob_step_success, 1000)
    analytical_result = analytical_solution(exact_sucesses, total_attempts, prob_step_success)

	# Lets also plot the probabilities evaluated using Monte Carlo method and the analytical solution using the binomial pdf
    plt.bar(epoch_result.keys(), list(map(lambda x: x/1000, epoch_result.values())))
	# Create analytical PMF (probability mass function)
    x = [i for i in range(0,total_attempts)]
    pmf = [analytical_solution(val, total_attempts, prob_step_success) for val in x]
    plt.plot(x, pmf, color='red')
    plt.xlabel('Number of successful beam walks')
    plt.ylabel('Probability')
    plt.savefig('distribution_beam_problem.png')
 
    # Now lets run the full analysis for an MC simulation
    results = perform_full_mc_simulation(exact_sucesses, total_attempts, prob_step_success, n_realisations, n_epochs)
    
    # capture the mean and variance of the case where the child has 5 successes out of 10
    mu = statistics.mean(results[5])
    std = statistics.stdev(results[5])
    
    plt.figure()
    # Lets have a look at the distribution of epochs and renormalise the frequencies so that it integrates to unity for fitting to a normal distribution
    plt.hist(results[5], bins=30, density = 1)
    # Now lets have a look at a normal, checking the central limit theorem.
    normal_dist = statistics.NormalDist(mu, std)
    x = [i/1000. for i in range(170,250)]
    vals = [normal_dist.pdf(i) for i in x]
    # print(vals)
    plt.plot(x, vals, color='red')
    plt.xlabel('Probability 5 successful walks out of 10')
    plt.ylabel('Frequency of occurrence')
    # print(normal_dist)
    plt.savefig('full_mc_sol.png')
    
    # Lets print general results
    print('Full MC simulation yields: {}+-{}, and analytical yields: {}'.format(statistics.mean(results[5]), statistics.stdev(results[5]), analytical_result))
    
    # Lets have a look at the convergence
    plt.figure()
    n_realisations = [i*1000 for i in range(1,10)]
    for ep in n_realisations:
        results = perform_full_mc_simulation(exact_sucesses, total_attempts, prob_step_success, ep, n_epochs)
        plt.scatter(ep, (statistics.mean(results[5]) - analytical_result) / analytical_result)
        
    plt.xlabel('Number of epochs')
    plt.ylabel('Percent deviation from analytical')
    plt.savefig('convergence.png')