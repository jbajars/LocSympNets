"""
Define rhs of a volume-preserving dynamical system, and 
functions of conserved quantities    
"""

import numpy as np

#==============================================================================
# Advection Equation
#==============================================================================
# Semi-discretized advection equation
def AdvectionEq(t, u, c, dx):
    N = len(u)
    f = np.zeros(N)
    for n in range(N):
        if n==0:
            f[n] = u[1] - u[N-1]
        elif n==N-1:
            f[n] = u[0] - u[N-2]
        else:
            f[n] = u[n+1] - u[n-1]
    f  = -c*f/2/dx 
    return f

#==============================================================================
# Rigid Body
#==============================================================================
# Euler equations of the motion of a free rigid body
def RigidBody(t, y, A, B, C):
    y1, y2, y3 = y
    f = np.zeros(3)
    f[0] = A*y2*y3
    f[1] = B*y3*y1  
    f[2] = C*y1*y2 
    return f
# Euler equations of the motion of a free rigid body; kinetic energy
def RigidBody_H(y, I1, I2, I3):
    y1 = y[:, 0]
    y2 = y[:, 1]
    y3 = y[:, 2]
    H = (y1**2/I1 + y2**2/I2 + y3**2/I3)/2 
    return H
# Euler equations of the motion of a free rigid body; quadratic invariant
def RigidBody_I(y):
    y1 = y[:, 0]
    y2 = y[:, 1]
    y3 = y[:, 2]
    I = (y1**2 + y2**2 + y3**2)/2 
    return I

#==============================================================================
# Charged Particle; reduced model
#==============================================================================
# Charged particle motion in an electromagnetic field; m=q=1
def ChargedParticle(t, z):
    y1, y2, p1, p2 = z
    a = np.sqrt(y1**2 + y2**2)
    b = 100*a**3
    f = np.zeros(4)
    f[0] = p1
    f[1] = p2  
    f[2] = y1/b + p2*a
    f[3] = y2/b - p1*a
    return f
# Hamiltonian of the charged particle motion in an electromagnetic field; m=q=1
def ChargedParticle_H(z):
    y1 = z[:, 0]
    y2 = z[:, 1]
    p1 = z[:, 2]
    p2 = z[:, 3]
    H = (p1**2 + p2**2)/2 + 1/np.sqrt(y1**2 + y2**2)/100 
    return H
