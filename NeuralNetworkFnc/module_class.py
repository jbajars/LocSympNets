"""
Volume-preserving neural network module class with PyTorch
"""
import torch
from torch import nn
 
# Locally symplectic linear volume-preserving gradient module: Up 
class LinLocSympModule_Up(nn.Module):
    def __init__(self, d, k1, k2, m, L, sqrt_var):
        super().__init__()
        """
        Weights: W, w and bias vector b=0
        """
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.L = L # L is optional; tau can be scaled with respect to L value
        self.W = torch.nn.Parameter(sqrt_var*torch.randn((self.m, self.d-1)))
        self.w = torch.nn.Parameter(sqrt_var*torch.randn((self.m, 1)))

    def forward(self, X, tau):
        """
        Forward function
        """        
        # y share the same memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k1-1]]
        
        # update k1 component of X
        f = torch.matmul(y, self.W.T)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k1-1):self.k1] += torch.matmul(tau, f)
        
        return X, tau    
    
    def back(self, X, tau):
        """
        Backward function
        """        
        # y share the same memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k1-1]]
        
        # update k1 component of X
        f = torch.matmul(y, self.W.T)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k1-1):self.k1] -= torch.matmul(tau, f)
        
        return X, tau    

# Locally symplectic linear volume-preserving gradient module: Low 
class LinLocSympModule_Low(nn.Module):
    def __init__(self, d, k1, k2, m, L, sqrt_var):
        super().__init__()
        """
        Weights: W, w and bias vector b=0
        """
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.L = L # L is optional; tau can be scaled with respect to L value
        self.W = torch.nn.Parameter(sqrt_var*torch.randn((self.m, self.d-1)))
        self.w = torch.nn.Parameter(sqrt_var*torch.randn((self.m, 1)))

    def forward(self, X, tau):
        """
        Forward function
        """        
        # y share the same memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k2-1]]
        
        # update k2 component of X
        f = torch.matmul(y, self.W.T)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k2-1):self.k2] -= torch.matmul(tau, f)
        
        return X, tau
        
    def back(self, X, tau):
        """
        Backward function
        """        
        # y share the same memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k2-1]]
        
        # update k2 component of X
        f = torch.matmul(y, self.W.T)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k2-1):self.k2] += torch.matmul(tau, f)
               
        return X, tau    

# Locally symplectic nonlinear volume-preserving gradient module: Up 
class LocSympModule_Up(nn.Module):
    def __init__(self, d, k1, k2, m, L, sigma, sqrt_var):
        super().__init__()
        """
        Weights: W, w and bias vector b
        """
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.L = L # L is optional; tau can be scaled with respect to L value
        self.sigma = sigma
        self.W = torch.nn.Parameter(sqrt_var*torch.randn((self.m, self.d-1)))
        self.w = torch.nn.Parameter(sqrt_var*torch.randn((self.m, 1)))
        self.b = torch.nn.Parameter(torch.zeros((1, self.m)))

    def forward(self, X, tau):
        """
        Forward function
        """        
        # y share the memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k1-1]]
        
        # update k1 component of X
        f = self.sigma(torch.matmul(y, self.W.T) + self.b)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k1-1):self.k1] += torch.matmul(tau, f)
        
        return X, tau
    
    def back(self, X, tau):
        """
        Backward function
        """    
        # y share the memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k1-1]]
        
        # update k1 component of X
        f = self.sigma(torch.matmul(y, self.W.T) + self.b)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k1-1):self.k1] -= torch.matmul(tau, f)
        
        return X, tau
    
# Locally symplectic nonlinear volume-preserving gradient module: Low 
class LocSympModule_Low(nn.Module):
    def __init__(self, d, k1, k2, m, L, sigma, sqrt_var):
        super().__init__()
        """
        Weights: W, w and bias vector b
        """
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.L = L # L is optional; tau can be scaled with respect to L value
        self.sigma = sigma
        self.W = torch.nn.Parameter(sqrt_var*torch.randn((self.m, self.d-1)))
        self.w = torch.nn.Parameter(sqrt_var*torch.randn((self.m, 1)))
        self.b = torch.nn.Parameter(torch.zeros((1, self.m)))

    def forward(self, X, tau):
        """
        Forward function
        """               
        # y share the memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k2-1]]
                
        # update k2 component of X
        f = self.sigma(torch.matmul(y, self.W.T) + self.b)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k2-1):self.k2] -= torch.matmul(tau, f)
               
        return X, tau  
    
    def back(self, X, tau):
        """
        Backward function
        """    
        # y share the memory with X 
        y = X[:, :, [i for i in range(self.d) if i!=self.k2-1]]
                
        # update k2 component of X
        f = self.sigma(torch.matmul(y, self.W.T) + self.b)
        f = torch.matmul(f, (self.w*self.W[:, (self.k1-1):self.k1]))
        X[:, :, (self.k2-1):self.k2] += torch.matmul(tau, f)
               
        return X, tau  
