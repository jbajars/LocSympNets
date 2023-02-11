"""
Script file for training locally-symplectic neural networks 
with PyTorch for learning nonlinear phase volume-preserving dynamics 
"""
import numpy as np
import torch
from torch import nn
from NeuralNetworkFnc.custom_dataset import CustomDataset
from NeuralNetworkFnc.mySequential import mySequential
from NeuralNetworkFnc.module_class import LocSympModule_Up
from NeuralNetworkFnc.module_class import LocSympModule_Low
from NeuralNetworkFnc.training_class import train_loop
from NeuralNetworkFnc.training_class import test_loop
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# plotting properties
import matplotlib
matplotlib.rc('font', size=22) 
matplotlib.rc('axes', titlesize=22)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

# find or set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
device = 'cpu'
torch.set_num_threads(1)

#==============================================================================
# load training and testing data and convert to PyTorch data type
#==============================================================================
fl = 'DynamicalSystems/SavedTrainingData/ChargedParticle/'
npz_file = np.load(fl + 'TDChargedParticleN200M100Tau02.npz')
x_train = torch.from_numpy(np.float32(npz_file['arr_0'])).to(device)
y_train = torch.from_numpy(np.float32(npz_file['arr_1'])).to(device)
tau_train = torch.from_numpy(np.float32(npz_file['arr_2'])).to(device)
x_test = torch.from_numpy(np.float32(npz_file['arr_3'])).to(device)
y_test = torch.from_numpy(np.float32(npz_file['arr_4'])).to(device)
tau_test = torch.from_numpy(np.float32(npz_file['arr_5'])).to(device)

# number of training data samples
N = len(x_train)

# dimension of the problem
d = x_train.shape[2]

# custom dataset 
training_data = CustomDataset(x_train, y_train, tau_train)
testing_data = CustomDataset(x_test, y_test, tau_test)

#==============================================================================
# training parameter values
#==============================================================================
# learning rate
learning_rate = 1e-3
# batch size
batch_size = N
# number of epochs
epochs = 1000_000
# scheduling: True or False
sch = True
# gamma value for exponential scheduling
eta1 = 1e-2
eta2 = 1e-6 
gamma = np.exp(np.log(eta2/eta1)/epochs)
# if scheduling True, then set initial learning rate
if sch:
    learning_rate = eta1
# activation function
sigma = nn.Sigmoid()
# square root of variance of the normal distribution 
# for initialization of weight values W and w
sqrt_var = np.sqrt(0.01) 

#==============================================================================
# data loader for PyTorch
#==============================================================================
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)

#==============================================================================
# loop L (even number) for the number of module compositions
# loop m for the number of width value
# loop k for the number of random runs
#==============================================================================
for L in [4]:
    
    for m in [32, 64]:
                  
        for k in range(2):
            
            # optional: set random seed to generate the same initial weight values
            torch.manual_seed(k)
                
            # define neural network of L number of layers  
            layers = []
            for i in range(int(L/2)):
                for j in range(d-1):
                    layers.append(LocSympModule_Up(d, j+1, j+2, m, L, sigma, sqrt_var))
                    layers.append(LocSympModule_Low(d, j+1, j+2, m, L, sigma, sqrt_var))                                           
            model = mySequential(*layers)
            print(model)                

            # send model to device
            model = model.to(device)                

            # initialize the loss function
            loss_fn = nn.MSELoss()

            # optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                
            # start time
            start = time.time()

            # training loop
            loss = np.zeros((epochs, 1))
            acc = np.zeros((epochs, 1))
            for t in range(epochs):
                loss[t] = train_loop(train_dataloader, model, loss_fn,
                                     optimizer, scheduler, sch).cpu()
                acc[t] = test_loop(test_dataloader, model, loss_fn)
                if t % 100 == 0:
                    print('Epoch %d / loss: %.12f / acc: %.12f' % (t+1, loss[t], acc[t]))

            # end time
            end = time.time()
            
            # total time taken
            print(f"Runtime of the program was {(end - start)/60:.4f} min.")

            #==================================================================
            # plot training error vs. test error
            #==================================================================
            fig, ax = plt.subplots(figsize=(9, 6.5))
            v = np.linspace(1, epochs, epochs)
            ax.loglog(v, loss, ls='-', color='tab:red', linewidth='1.5', label='loss')
            ax.loglog(v, acc, ls='--', color='tab:blue', linewidth='1.5', label='accuracy')
            ax.set_xlabel("epochs")
            ax.set_ylabel("MSE")
            ax.grid(True)
            ax.legend(loc=3, shadow=True, prop={'size': 22}, ncol=1) 
            ax.axis([1, epochs, 10**(-8), 10**(-1)])
            plt.xticks([1, 10, 10**2, 10**3, 10**4, 10**5, 10**6])
            plt.yticks([10**(-8), 10**(-7), 10**(-6), 10**(-5), 10**(-4), 
                        10**(-3), 10**(-2), 10**(-1)])
            ax.set_title('LocSympNets, Charged Particle: L=' + str(L) + 
                         ', m=' + str(m) + ', k=' + str(k))
            # optional: save figure
            plt.savefig('Figures/ChargedParticle/LossAcc/LossAcc_L' + str(L) +
                        'm' + str(m) + 'k' + str(k) + '.png',
                        dpi=300, bbox_inches='tight')
            plt.show()
                
            #==================================================================
            # save model in folder SavedNeuralNets
            #==================================================================
            g = "SavedNeuralNets/ChargedParticle/"
            f = str(L) + "L" + str(m) + "m" + str(k) + "k" + ".pth"
            if sch:
                file_w = g + "schChargedParticleN200M100Tau02Epoch1000TH" + f
            else:
                file_w = g + "ChargedParticleN200M100Tau02Epoch1000TH" + f
            model = model.to('cpu')    
            torch.save([model, loss, acc, start, end], file_w)
            
            # delete model
            del model
    