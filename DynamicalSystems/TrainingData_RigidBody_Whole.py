"""
Solve rigid body equations to obtain training data for learning 
the whole dynamics with constant time step tau value
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

# number of random initial conditions; number of training data samples
N = 300
# save training data in time
train_X = np.zeros((N, 1, d))
train_Y = np.zeros((N, 1, d))
train_Tau = tau*np.ones((N, 1, 1))

# number of random initial conditions; number of testing data samples
M = 100
# save testing data in time
test_X = np.zeros((M, 1, d))
test_Y = np.zeros((M, 1, d))
test_Tau = tau*np.ones((M, 1, 1))

#==============================================================================
# obtain training data
# set random seed
np.random.seed(0)
for n in range(N):
    u = 2*np.pi*np.random.rand(1)
    v = np.pi*np.random.rand(1)
    y1 = np.cos(u)*np.sin(v)
    y2 = np.sin(u)*np.sin(v)
    y3 = np.cos(v) 
    tau = train_Tau[n, 0, 0]
    train_X[n, 0, 0] = y1
    train_X[n, 0, 1] = y2
    train_X[n, 0, 2] = y3
    # solve Rigid Body equations with RK45
    sol = solve_ivp(RigidBody, [0, tau], [y1.item(), y2.item(), y3.item()], 
                    method='RK45', args=(A, B, C,), rtol = 1e-12, atol = 1e-12) 
    # save data
    train_Y[n, 0, :] = sol.y[:, -1]
 
#==============================================================================    
# obtain testing data
# set random seed
np.random.seed(100)
for m in range(M):
    u = 2*np.pi*np.random.rand(1)
    v = np.pi*np.random.rand(1)
    y1 = np.cos(u)*np.sin(v)
    y2 = np.sin(u)*np.sin(v)
    y3 = np.cos(v) 
    tau = test_Tau[m, 0, 0]
    test_X[m, 0, 0] = y1
    test_X[m, 0, 1] = y2
    test_X[m, 0, 2] = y3
    # solve Rigid Body equations with RK45
    sol = solve_ivp(RigidBody, [0, tau], [y1.item(), y2.item(), y3.item()], 
                    method='RK45', args=(A, B, C,), rtol = 1e-12, atol = 1e-12) 
    # save data
    test_Y[m, 0, :] = sol.y[:, -1]

#==============================================================================
# visualize training and testing data
fig1, ax = plt.subplots(figsize=(9, 6.5))
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
y1 = np.outer(np.cos(u), np.sin(v))
y2 = np.outer(np.sin(u), np.sin(v))
y3 = np.outer(np.ones(np.size(u)), np.cos(v))
cmap = plt.contourf(u, v, y3, 10, cmap='Oranges')
cbar = fig1.colorbar(cmap)
cbar.set_label("$z$", y=1, ha='right', rotation=0)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\theta$")
X = train_X[:, 0, 0]
Y = train_X[:, 0, 1]
Z = train_X[:, 0, 2]
Theta = np.arccos(Z)
Phi = np.arctan2(Y, X) + np.pi
ax.plot(Phi, Theta, 'o', ms=5, color='tab:blue', label='training data')
X = test_X[:, 0, 0]
Y = test_X[:, 0, 1]
Z = test_X[:, 0, 2]
Theta = np.arccos(Z)
Phi = np.arctan2(Y, X) + np.pi
ax.plot(Phi, Theta, 'd', ms=5, color='tab:green', label='validation data')
ax.legend(bbox_to_anchor=(-0.075, 1, 1.125, 0.1), 
          ncol=2, shadow=True, mode="expand")

plt.xticks([0, 2, 4, 6])
plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])

# optional: save figure
plt.savefig('TrainingData_Figures/RigidBody_Whole.png', 
            dpi=300, bbox_inches='tight')

#==============================================================================
# save training and testing data
# X - input
# Y - output
# Tau - time steps
np.savez('SavedTrainingData/RigidBody/Whole/TDRigidBodyWholeN300M100Tau01', 
         train_X, train_Y, train_Tau, test_X, test_Y, test_Tau)
