"""
Script file for making predictions 
with nonlinear locally-symplectic neural networks 
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp    
from DynamicalSystems.VolumePreservingODEs import ChargedParticle
from DynamicalSystems.VolumePreservingODEs import ChargedParticle_H

# plotting properties
import matplotlib
matplotlib.rc('font', size=22) 
matplotlib.rc('axes', titlesize=22)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

#==============================================================================
# single quasi-periodic trajectory of charged particle dynamics
#==============================================================================
# dimension of the problem
d = 4
# initial conditions
y1 = 0.1
y2 = 1
p1 = 1.1
p2 = 0.5
# make predictions with time step tau
tau = 0.2
# number of time steps
Nsteps = 1000
# length of time interval [0, Tend]
Tend = tau*Nsteps
# time grid points
tn = np.linspace(0, Tend, Nsteps+1)
# solve charged particle equations with RK45
sol = solve_ivp(ChargedParticle, [0, Tend], [y1, y2, p1, p2], method='RK45',
                t_eval=tn, rtol = 1e-12, atol = 1e-12)
# Hamiltonian value at t=0 
H0 = ChargedParticle_H(np.array([y1, y2, p1, p2]).reshape((1,d)))

#==============================================================================
# loop L (even number) for the number of module compositions
# loop m for the number of width value
# loop k for the number of random runs
#==============================================================================
for L in [4]:
    
    for m in [32, 64]:
        
        # count number of predictions by neural networks
        Nk = 0 
              
        # save errors
        SErr = np.zeros([Nsteps+1, 1])
        HErr = np.zeros([Nsteps+1, 1])
            
        for k in range(2):
                        
            Nk += 1
            
            #==============================================================
            # make predictions with volume-preserving neural network
            #==============================================================
            # load neural network
            g = "SavedNeuralNets/ChargedParticle/"
            f = str(L) + "L" + str(m) + "m" + str(k) + "k" + ".pth"             
            file_w = g + "schChargedParticleN200M100Tau02Epoch1000TH" + f
            model, loss, acc, start, end = torch.load(file_w)
            print(f"Runtime was {(end - start)/60:.4f} min.")
            
            # save predictions in matrix U
            U = np.zeros([Nsteps+1, d])                    
            # initial condition
            Z = torch.tensor([[[np.float32(y1), np.float32(y2), 
                                np.float32(p1), np.float32(p2)]]])
            # turn scalar tau into tensor Tau
            Tau = torch.tensor([[[tau]]])
            # perform predictions iteratively without gradient calculation
            with torch.no_grad():
                for j in range(Nsteps+1):
                    U[j, :] = Z[0, 0, :]
                    Z, _ = model(Z, Tau)
                
            # compute errors
            SErr += np.sqrt(np.sum((U - sol.y.T)**2, 1)).reshape((Nsteps+1,1))
            HErr += np.abs((ChargedParticle_H(U).reshape((Nsteps+1,1)) - H0)/H0)
           
            fig, ax = plt.subplots(figsize=(9, 6.5))
            ax.plot(sol.y[0], sol.y[1], ls='-', color='k', linewidth='1', 
                    marker='', ms=5, label='exact')
            ax.plot(U[:, 0], U[:, 1], ls='--', color='tab:purple', linewidth='1', 
                    marker='', ms=2, label='predicted')
            ax.set_xlabel("$y_1$")
            ax.set_ylabel("$y_2$")
            ax.grid(True)
            ax.axis([-1.7, 1.7, -1.7, 2.1])  
            plt.xticks([-1.5, -0.75, 0, 0.75, 1.5])
            plt.yticks([-1.5, -0.75, 0, 0.75, 1.5])
            ax.set_aspect('equal','box')
            ax.legend(loc=9, shadow=True, prop={'size': 18}, ncol=2) 
            ax.set_title('LocSympNets, Charged Particle: L=' + str(L) + 
                         ', m=' + str(m) + ', k=' + str(k), fontsize=20, loc='right')
            # optional: save figure
            plt.savefig('Figures/ChargedParticle/Predictions/LocSympNets/' +
                        'Predictions_L' + str(L) + 'm' + str(m) + 
                        'k' + str(k) + '.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        #==================================================================
        # plot averaged errors for fixed L and m
        #==================================================================
        fig1, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(tn, SErr/Nk, ls='-', color='k', linewidth='1')
        ax.set_xlabel("$t$")
        ax.set_ylabel("solution absolute error")
        ax.grid(True)
        ax.set_title('LocSympNets, Charged Particle: L=' + str(L) + ', m='+str(m))
        # optional: save figure
        plt.savefig('Figures/ChargedParticle/Predictions/LocSympNets/SErr_L' + 
                    str(L) + 'm' + str(m) + '.png', dpi=300, bbox_inches='tight')
        plt.show()

        fig2, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(tn, HErr/Nk, ls='-', color='k', linewidth='1')
        ax.set_xlabel("$t$")
        ax.set_ylabel("Hamiltonian relative error")
        ax.grid(True)
        ax.set_title('LocSympNets, Charged Particle: L=' + str(L) + ', m='+str(m))
        # optional: save figure
        plt.savefig('Figures/ChargedParticle/Predictions/LocSympNets/HErr_L' + 
                    str(L) + 'm' + str(m) + '.png', dpi=300, bbox_inches='tight')
        plt.show()
