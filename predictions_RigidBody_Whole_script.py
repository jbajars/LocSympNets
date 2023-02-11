"""
Script file for making predictions 
with nonlinear locally-symplectic neural networks 
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp    
from DynamicalSystems.VolumePreservingODEs import RigidBody
from DynamicalSystems.VolumePreservingODEs import RigidBody_H
from DynamicalSystems.VolumePreservingODEs import RigidBody_I

# plotting properties
import matplotlib
matplotlib.rc('font', size=22) 
matplotlib.rc('axes', titlesize=22)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

#==============================================================================
# whole rigid body dynamics
#==============================================================================
# dimension of the problem
d = 3
# principal components of inertia
I1 = 2
I2 = 1
I3 = 2/3
# system constants
A = (I2-I3)/I2/I3
B = (I3-I1)/I3/I1
C = (I1-I2)/I1/I2
# make predictions with time step tau
tau = 0.1
# number of time steps
Nsteps = 1000
# length of time interval [0, Tend]
Tend = tau*Nsteps
# time grid points
tn = np.linspace(0, Tend, Nsteps+1)

# number of initial conditions
J = 12
# solve rigid body equations to obtain exact solutions for J initial conditions
K = 12
h = np.pi/J
u = np.linspace(h/2, 1*np.pi-h/2, J)
h = np.pi/2/J
v = -np.linspace(h/2, np.pi/2-h/2, J)
# store all solutions in the list sol_list
sol_list = []
# store all kinetic energy and invariant values at t=0 in the lists
H0_list = []
I0_list = []
for j in range(J):
    y1 = np.cos(u[j])*np.sin(v[j])
    y2 = np.sin(u[j])*np.sin(v[j])
    y3 = np.cos(v[j])
    sol = solve_ivp(RigidBody, [0, Tend], [y1, y2, y3], method='RK45', 
                    args=(A, B, C,), t_eval=tn, rtol = 1e-12, atol = 1e-12) 
    sol_list.append(sol)
    H0 = RigidBody_H(np.array([y1, y2, y3]).reshape((1,d)),I1,I2,I3)
    H0_list.append(H0)
    I0 = RigidBody_I(np.array([y1, y2, y3]).reshape((1,d)))
    I0_list.append(I0)
              
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
        IErr = np.zeros([Nsteps+1, 1])
            
        for k in range(2):
                        
            Nk += 1
            
            #==============================================================
            # make predictions with volume-preserving neural network
            #==============================================================
            # load neural network
            g = "SavedNeuralNets/RigidBody/Whole/"
            f = str(L) + "L" + str(m) + "m" + str(k) + "k" + ".pth"             
            file_w = g + "schRigidBodyWholeN300M100Tau01Epoch1000TH" + f
            model, loss, acc, start, end = torch.load(file_w)
            print(f"Runtime was {(end - start)/60:.4f} min.")
            
            # plot exact and predicted solutions
            ax = plt.figure(figsize=(6,6)).add_subplot(111, projection='3d')  
            # plot sphere
            u = np.linspace(0, 2 * np.pi, 200)
            v = np.linspace(0, np.pi, 200)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            mycmap = plt.get_cmap('gray')
            ax.plot_surface(x, y, z, color='k', alpha=0.1, cmap=mycmap)
            ax.set_xlim3d(-1.05, 1.05)
            ax.set_ylim3d(-1.05, 1.05)
            ax.set_zlim3d(-1.05, 1.05)
            ax.set_xlabel("$x$", labelpad=10)
            ax.set_ylabel("$y$", labelpad=10)
            ax.set_zlabel("$z$", labelpad=4)
            ax.set_box_aspect((1, 1, 1))
            ax.grid(True)
            ax.set_title('LocSympNets, Rigid Body: L=' + str(L) + 
                        ', m=' + str(m) + ', k=' + str(k), fontsize=20)
            
            # predictions and errors for each initial condition trajectory
            for j in range(J):
                
                # exact solution, kinetic energy and invarinat values
                sol = sol_list[j]
                H0 = H0_list[j]
                I0 = I0_list[j]
                
                # initial conditions
                y1 = sol.y[0, 0]
                y2 = sol.y[1, 0]
                y3 = sol.y[2, 0]
                
                # save predictions in matrix U
                U = np.zeros([Nsteps+1, d])                    
                # initial condition
                Z = torch.tensor([[[np.float32(y1), np.float32(y2), np.float32(y3)]]])
                # turn scalar tau into tensor Tau
                Tau = torch.tensor([[[tau]]])
                # perform predictions iteratively without gradient calculation
                with torch.no_grad():
                    for j in range(Nsteps+1):
                        U[j, :] = Z[0, 0, :]
                        Z, _ = model(Z, Tau)
                
                # compute errors
                SErr += np.sqrt(np.sum((U - sol.y.T)**2, 1)).reshape((Nsteps+1,1))
                HErr += np.abs((RigidBody_H(U,I1,I2,I3).reshape((Nsteps+1,1)) - H0)/H0)
                IErr += np.abs((RigidBody_I(U).reshape((Nsteps+1,1)) - I0)/I0)
                
                # plot exact and predicted solution trajectories
                ax.plot(y1, y2, y3, marker='.', ms=5, color='tab:red')
                ax.plot(sol.y[0], sol.y[1], sol.y[2], ls='-', color='k', linewidth='0.1')
                ax.plot(U[:, 0], U[:, 1], U[:, 2], ls='--', color='tab:red', linewidth='0.5')
                                
            # optional: save figure
            plt.savefig('Figures/RigidBody/Whole/Predictions/LocSympNets/' + 
                        'Predictions_L' + str(L) + 'm' + str(m) +
                        'k' + str(k) + '.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        #==================================================================
        # plot averaged errors for fixed L and m
        #==================================================================
        fig1, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(tn, SErr/Nk/J, ls='-', color='k', linewidth='1')
        ax.set_xlabel("$t$")
        ax.set_ylabel("solution absolute error")
        ax.grid(True)
        ax.set_title('LocSympNets, Rigid Body: L=' + str(L) + ', m='+str(m))
        # optional: save figure
        plt.savefig('Figures/RigidBody/Whole/Predictions/LocSympNets/SErr_L' + 
                    str(L) + 'm' + str(m) + '.png', dpi=300, bbox_inches='tight')
        plt.show()

        fig2, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(tn, HErr/Nk/J, ls='-', color='k', linewidth='1')
        ax.set_xlabel("$t$")
        ax.set_ylabel("kinetic energy relative error")
        ax.grid(True)
        ax.set_title('LocSympNets, Rigid Body: L=' + str(L) + ', m='+str(m))
        # optional: save figure
        plt.savefig('Figures/RigidBody/Whole/Predictions/LocSympNets/HErr_L' + 
                    str(L) + 'm' + str(m) + '.png', dpi=300, bbox_inches='tight')
        plt.show()

        fig3, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(tn, IErr/Nk/J, ls='-', color='k', linewidth='1')
        ax.set_xlabel("$t$")
        ax.set_ylabel("invariant relative error")
        ax.grid(True)
        ax.set_title('LocSympNets, Rigid Body: L=' + str(L) + ', m='+str(m))
        # optional: save figure
        plt.savefig('Figures/RigidBody/Whole/Predictions/LocSympNets/IErr_L' + 
                    str(L) + 'm' + str(m) + '.png', dpi=300, bbox_inches='tight')
        plt.show()
                