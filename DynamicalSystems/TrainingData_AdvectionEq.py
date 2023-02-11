"""
Solve semi-discretized advection equations to obtain training data
with constant time step tau value
"""

import numpy as np
from scipy.integrate import solve_ivp    
from VolumePreservingODEs import AdvectionEq

# dimension of the problem d
d = 35
# wave speed
c = 1
# domain length [0,L]
L = 1
# grid size
dx = 2*L/d
# grid points; periodic boundary conditions
x = np.linspace(-L, L-dx, d)

# constant time step tau
tau = 0.01

# number of initial conditions; number of training data samples
N = 60
# save training data in time
train_X = np.zeros((N, 1, d))
train_Y = np.zeros((N, 1, d))
train_Tau = tau*np.ones((N, 1, 1))

# number of initial conditions; number of testing data samples
M = 20
# save testing data in time
test_X = np.zeros((M, 1, d))
test_Y = np.zeros((M, 1, d))
test_Tau = tau*np.ones((N, 1, 1))

#==============================================================================
# obtain training data
np.random.seed(0)
for n in range(N):
    # solve semi-discretized advection equations for random initial condition 
    u = np.random.randn(1, d) 
    train_X[n, 0, :] = u
    u0 = []
    for i in range(d):
        u0.append(u[0, i])
    sol = solve_ivp(AdvectionEq, [0, train_Tau[n, 0, 0]], u0, method='RK45',
                    args=(c, dx), rtol = 1e-12, atol = 1e-12)  
    # save data
    train_Y[n, 0, :] = sol.y[:, -1]

#==============================================================================
# obtain testing data
np.random.seed(1)
for m in range(M):
    # solve semi-discretized advection equation for random initial condition 
    u = np.random.randn(1, d) 
    test_X[m, 0, :] = u
    u0 = []
    for i in range(d):
        u0.append(u[0, i])
    sol = solve_ivp(AdvectionEq, [0, test_Tau[n, 0, 0]], u0, method='RK45',
                    args=(c, dx), rtol = 1e-12, atol = 1e-12)  
    # save data
    test_Y[m, 0, :] = sol.y[:, -1]   

#==============================================================================
# save training and testing data
# X - input
# Y - output
# Tau - time steps
np.savez('SavedTrainingData/AdvectionEq/TDAdvectionEqD35N60M20Tau001',
         train_X, train_Y, train_Tau, test_X, test_Y, test_Tau)
