"""
Solve rigid body equations to obtain training data for a single trajectory
with constant time step tau value
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp    
from VolumePreservingODEs import RigidBody

import matplotlib
matplotlib.rc('font', size=22)
matplotlib.rc('axes', titlesize=22)

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

# dimension of the problem d
d = 3

# principal components of inertia
I1 = 2
I2 = 1
I3 = 2/3

# system constants
A = (I2-I3)/I2/I3
B = (I3-I1)/I3/I1
C = (I1-I2)/I1/I2

# constant time step tau
tau = 0.1

# number of time steps; number of training data samples
N = 120
# computational time interval [0,Tn]
Tn = tau*N
# grid points
tn = np.linspace(0, Tn, N+1)
# save training data in time
train_X = np.zeros((N, 1, d))
train_Y = np.zeros((N, 1, d))
train_Tau = tau*np.ones((N, 1, 1))

# number of time steps; number of testing data samples
M = 40
# computational time interval [0,Tm]
Tm = tau*M
# grid points
tm = np.linspace(0, Tm, M+1)
# save testing data in time
test_X = np.zeros((M, 1, d))
test_Y = np.zeros((M, 1, d))
test_Tau = tau*np.ones((N, 1, 1))

# figure to plot in training and tetsing data
fig1, ax = plt.subplots(figsize=(9, 6.5))

# initial conditions
y1 = np.cos(1.1)
y2 = 0.0
y3 = np.sin(1.1)

# solve Rigid Body equations with RK45 to obtain 'exact' solution
# on time interval [0, Tn+Tm]
sol = solve_ivp(RigidBody, [0, Tn+Tm], [y1, y2, y3], method='RK45', 
                args=(A, B, C,), rtol = 1e-12, atol = 1e-12)  
# plot solution
ax.plot(sol.t, sol.y[0], ls='-', color='k', linewidth='1.5') 
ax.plot(sol.t, sol.y[1], ls='-', color='k', linewidth='1.5') 
ax.plot(sol.t, sol.y[2], ls='-', color='k', linewidth='1.5') 
ax.set_xlabel("$t$")
ax.set_ylabel("Solution")         
ax.set_title("Rigid Body")  
ax.grid(True)
ax.axis([0, Tn+Tm, -0.75, 1])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
plt.yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

#==============================================================================
# solve Rigid Body equations with RK45 to obtain training data
sol = solve_ivp(RigidBody, [0, Tn], [y1, y2, y3], method='RK45', 
                args=(A, B, C,), t_eval=tn, rtol = 1e-12, atol = 1e-12)  
train_X[:, 0, :] = sol.y.T[0:N, :] 
train_Y[:, 0, :] = sol.y.T[1:N+1, :]

# plot training data
ax.plot(sol.t, sol.y[0], ls='--', color='tab:blue', linewidth='2.5')
ax.plot(sol.t, sol.y[1], ls='--', color='tab:orange', linewidth='2.5')
ax.plot(sol.t, sol.y[2], ls='--', color='tab:green', linewidth='2.5')

#==============================================================================
# initial conditions to obtain testing data
y1 = sol.y.T[-1, 0]
y2 = sol.y.T[-1, 1]
y3 = sol.y.T[-1, 2]  

# solve Rigid Body equations with RK45 to obtain testing data
sol = solve_ivp(RigidBody, [0, Tm], [y1, y2, y3], method='RK45', 
                args=(A, B, C,), t_eval=tm, rtol = 1e-12, atol = 1e-12)  
test_X[:, 0, :] = sol.y.T[0:M, :] 
test_Y[:, 0, :] = sol.y.T[1:M+1, :]

# plot testing data
ax.plot(sol.t+Tn, sol.y[0], ls='-.', color='tab:red', linewidth='2.5')
ax.plot(sol.t+Tn, sol.y[1], ls='-.', color='tab:purple', linewidth='2.5')
ax.plot(sol.t+Tn, sol.y[2], ls='-.', color='tab:brown', linewidth='2.5')

# optional: save figure
plt.savefig('TrainingData_Figures/RigidBody_Single.png', 
            dpi=300, bbox_inches='tight')

#==============================================================================
# saving training and testing data
# X - input
# Y - output
# Tau - time steps
np.savez('SavedTrainingData/RigidBody/Single/TDRigidBodySingleN120M40Tau01',
         train_X, train_Y, train_Tau, test_X, test_Y, test_Tau)
