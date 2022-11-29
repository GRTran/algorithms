'''
Author: Gregory Jones
Date: 22-11-2022
Description: Creating an Euler-Maruyama solver and applying it to the Ornstein-Uhlenbeck process
'''
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def euler_maruyama(drift, volatility, y0, t0, t1, N):
    '''
    Solves a stochastic differential equation using the Euler-Maruyama method of the form dX = a(X,t)dt + b(X,t)dW were a is the drift function and b is the diffusion function.
    
    The solution is given by the simple forwar Euler method but just with the Weiner process term generated according to some appropriate mathematical package.
    Yt+1 = Yt + a(X,t) dt + b(X,t)dW
    '''        
    dt = (t1 - t0) / (N+1)
    y = np.zeros(N)
    y[0] = y0
    for i in range(1, N):
        dW = np.random.normal(loc=0.0, scale = np.sqrt(dt)) # Weiner process sampler
        y[i] = y[i-1] + drift(y[i]) * dt + volatility() * dW # Solve for the next step
    
    return y

# Perform Euler Maruyama simulation for the Ornstein-Uhlenbeck process
sigma = 0.1
mu = 1.5
drift = lambda x: sigma * (mu - x)
volatility = lambda: sigma
n_instances = 10
for i in range(n_instances):
    sol = euler_maruyama(drift, volatility, 0.1, 0, 100, 1000)
    t = np.linspace(0, 100, 1000)
    plt.plot(t, sol)
    
# assert euler_maruyama.nopython_signatures
plt.savefig('sample_solution.png')
        
    