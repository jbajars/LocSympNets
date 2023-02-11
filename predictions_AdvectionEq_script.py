"""
Script file for making predictions 
with linear locally-symplectic neural networks 
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp    
from DynamicalSystems.VolumePreservingODEs import AdvectionEq

# plotting properties
import matplotlib
matplotlib.rc('font', size=22) 
matplotlib.rc('axes', titlesize=22)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

#==============================================================================
# semi-discretized advection equation
#==============================================================================
# dimension of the problem
d = 35 
# wave speed
c = 1
# space domain length
Lx = 2
# grid size
dx = Lx/d
# space grid points
x = np.linspace(-Lx/2, Lx/2, d+1)
# initial condition
u0 = np.exp(-10*x[:-1]**2).tolist()
# make predictions with time step tau
tau = 0.01
# number of time steps
Nsteps = 400
# length of time interval [0, Tend]
Tend = tau*Nsteps
# time grid points
tn = np.linspace(0, Tend, Nsteps+1)
# solve semi-discretized advection equations to obtain exact solution 
sol = solve_ivp(AdvectionEq, [0, Tend], u0, method='RK45', args=(c, dx), 
                t_eval=tn, rtol = 1e-12, atol = 1e-12) 

#==============================================================================
# loop L (even number) for the number of module compositions
# loop m for the number of width value
# loop k for the number of random runs
#==============================================================================
for L in [4]:
    
    for m in [d]: 
        
        # count number of predictions by neural networks
        Nk = 0 
        
        # sum of grid norm errors 
        SumError = 0
        
        for k in range(3):
            
            Nk += 1
            
            #==============================================================
            # make predictions with volume-preserving neural network
            #==============================================================
            # load neural network
            g = "SavedNeuralNets/AdvectionEq/"
            f = str(L) + "L" + str(m) + "m" + str(k) + "k" + ".pth" 
            file_w = g + "AdvectionEqD35N60M20Tau001Epoch20TH" + f
            model, loss, acc, start, end = torch.load(file_w)
            print(f"Runtime of training was {(end - start)/60:.4f} min.")

            # save predictions in matrix U
            U = np.zeros([Nsteps+1, d])
            # initial condition
            Z = torch.tensor(u0).reshape((1,1,d))
            # turn scalar tau into tensor Tau
            Tau = torch.tensor([[[tau]]])
            # perform predictions iteratively without gradient calculation
            with torch.no_grad():
                for j in range(Nsteps+1):
                    U[j, :] = Z[0, 0, :]
                    Z, _ = model(Z, Tau)
                    
            # plot exact and predicted solutions
            fig, ax = plt.subplots(figsize=(9, 6.5))
            # plot exact solution
            ax.plot(x, np.append(sol.y[:, -1], sol.y[0, -1]), ls='-', color='k', 
                    linewidth='1', marker='o', ms=8, label='exact')          
            # plot predicted solution
            ax.plot(x, np.append(U[-1, :], U[-1, 0]), ls='--', color='tab:orange', 
                    linewidth='1', marker='*', ms=6, label='predicted')
            ax.set_xlabel("$x$")
            ax.set_ylabel("solution")         
            ax.grid(True)
            ax.legend(loc=1, shadow=True, prop={'size': 22}, ncol=1) 
            ax.axis([-Lx/2, Lx/2, -0.2, 1])
            plt.xticks([-1, -0.5, 0, 0.5, 1])
            plt.yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])  
            ax.set_title('LocSympNets, Advection Eq: L=' + str(L) +
                         ', m=' + str(m) + ', k=' + str(k))
            # optional: save figure
            plt.savefig('Figures/AdvectionEq/Predictions/Predictions_L' + 
                        str(L) + 'm' + str(m) + 'k' + str(k) + '.png',
                        dpi=300, bbox_inches='tight')
            plt.show()
                            
            #============================================================
            # compute L2 grid norm error
            #============================================================
            error = np.sqrt(dx*np.sum(np.abs(sol.y.T - U)**2, 1))
            SumError += error[-1]
            print(["Grid norm error", error[-1]])
                         
        print(["Averaged grid norm error", SumError/Nk])
