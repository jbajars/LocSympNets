"""
Neural network training functions with PyTorch
"""
import torch
   
# Define training loop for the volume-preserving neural networks
def train_loop(dataloader, model, loss_fn, optimizer, scheduler, sch):
    
    for batch, (X, y, tau) in enumerate(dataloader):
    
        # forward step with time step tau
        pred_y, _ = model(X, tau)
        
        # compute loss
        loss = loss_fn(pred_y, y)  
        
        # return error value          
        error = loss.detach()
            
        # backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # scheduling step if sch=True
        if sch:
            scheduler.step()            

    return error   

# Define training loop for the symmetric volume-preserving neural networks
def train_loop_sym(dataloader, model, loss_fn, optimizer, scheduler, sch):
    
    for batch, (X, y, tau) in enumerate(dataloader):
                
        # forward step with time step tau/2
        pred, _ = model(X, tau/2)
        
        # backward step with time step -tau/2
        pred, _ = model.back(pred, -tau/2)
        
        # compute loss
        loss = loss_fn(pred, y) 
                
        # return error value 
        error = loss.detach()
            
        # backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # scheduling step if sch=True
        if sch:
            scheduler.step()            

    return error

# Define testing loop for the volume-preserving neural networks
def test_loop(dataloader, model, loss_fn):
    
    num_batches = len(dataloader)
    test_loss = 0

    # forward step without gradient calculation
    with torch.no_grad():
        for X, y, tau in dataloader:
            
            # forward step with time step tau
            pred, _ = model(X, tau)
            
            # return testing error 
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    
    return test_loss       

# Define testing loop for the symmetric volume-preserving neural networks
def test_loop_sym(dataloader, model, loss_fn):
    
    num_batches = len(dataloader)
    test_loss = 0
    
    # forward step without gradient calculation
    with torch.no_grad():
        for X, y, tau in dataloader:
    
            # forward step with time step tau/2
            pred, _ = model(X, tau/2)
            
            # backward step with time step -tau/2
            pred, _ = model.back(pred, -tau/2)
            
            # return testing error            
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    
    return test_loss    
