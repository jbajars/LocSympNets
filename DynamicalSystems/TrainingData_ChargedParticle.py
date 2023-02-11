"""
Solve charged particle in an electromagnetic field equations to obtain 
training data for a single trajectory with constant time step tau value
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp    
from VolumePreservingODEs import ChargedParticle

import matplotlib
matplotlib.rc('font', size=22) 
matplotlib.rc('axes', titlesize=22)

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

# dimension of the problem d
d = 4

# constant time step tau
tau = 0.2

# number of time steps; number of training data samples
N = 200
# computational time interval [0,Tn]
Tn = tau*N
# grid points
tn = np.linspace(0, Tn, N+1)
# save training data in time
train_X = np.zeros((N, 1, d))
train_Y = np.zeros((N, 1, d))
train_Tau = tau*np.ones((N, 1, 1))

# number of time steps; number of testing data samples
M = 100
# computational time interval [0,Tm]
Tm = tau*M
# grid points
tm = np.linspace(0, Tm, M+1)
# save testing data in time
test_X = np.zeros((M, 1, d))
test_Y = np.zeros((M, 1, d))
test_Tau = tau*np.ones((N, 1, 1))

# initial conditions
y1 = 0.1
y2 = 1.0
p1 = 1.1
p2 = 0.5

# figure to plot in
fig1, ax = plt.subplots(figsize=(9, 6.5))
# solve charged particle equations with RK45 to obtain 'exact' solution
# on time interval [0, Tn+Tm]
sol = solve_ivp(ChargedParticle, [0, Tn+Tm], [y1, y2, p1, p2],
                method='RK45', rtol = 1e-12, atol = 1e-12)  
# plot (y1, y2)
ax.plot(sol.y[0], sol.y[1], ls='-', color='k', linewidth='1') 
ax.set_xlabel("$y_1$")
ax.set_ylabel("$y_2$")         
ax.set_title("Charged particle equations")  
ax.grid(True)
ax.axis([-1.7, 1.7, -1.7, 1.7])
plt.xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
ax.set_aspect('equal','box')

#==============================================================================
# solve equations with RK45 to obtain training data
sol = solve_ivp(ChargedParticle, [0, Tn], [y1, y2, p1, p2],
                method='RK45', t_eval=tn, rtol = 1e-12, atol = 1e-12)  
train_X[:, 0, :] = sol.y.T[0:N, :] 
train_Y[:, 0, :] = sol.y.T[1:N+1, :]

# plot training data
ax.plot(sol.y[0], sol.y[1], ls='', color='tab:red', 
        marker='o', ms=5, mec='tab:red')

#==============================================================================
# initial conditions to obtain testing data
y1 = sol.y.T[-1, 0]
y2 = sol.y.T[-1, 1]
p1 = sol.y.T[-1, 2] 
p2 = sol.y.T[-1, 3]
 
# solve equations with RK45 to obtain testing data
sol = solve_ivp(ChargedParticle, [0, Tm], [y1, y2, p1, p2],
                method='RK45', t_eval=tm, rtol = 1e-12, atol = 1e-12)  
test_X[:, 0, :] = sol.y.T[0:M, :] 
test_Y[:, 0, :] = sol.y.T[1:M+1, :]

# plot testing data
ax.plot(sol.y[0], sol.y[1], ls='', color='tab:blue', 
        marker='d', ms=5, mec='tab:blue')

# optional: save figure
plt.savefig('TrainingData_Figures/ChargedParticle.png', 
            dpi=300, bbox_inches='tight')

#==============================================================================
# save training and testing data
# X - input
# Y - output
# Tau - time steps
np.savez('SavedTrainingData/ChargedParticle/TDChargedParticleN200M100Tau02', 
         train_X, train_Y, train_Tau, test_X, test_Y, test_Tau)
