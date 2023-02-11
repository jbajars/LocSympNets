# LocSympNets
PyTorch code for training locally-symplectic neural networks LocSympNets, and symmetric version SymLocSympNets, for learning phase volume-preserving linear and nonlinear dynamics. Using the code, please cite: Jānis Bajārs, *Locally-symplectic neural networks for learning volume-preserving dynamics*, Journal of Computational Physics **476**:111911, 2023 ([DOI: 10.1016/j.jcp.2023.111911](https://www.sciencedirect.com/science/article/pii/S0021999123000062?via%3Dihub)).

File LocSympNets_PureCode_WithoutData.zip contains pure code without precomputed training data, images and pretrained neural networks.

<p float="left">
  <img src="Figures/Fig1.png" width="39%" />
  <img src="Figures/Fig2.png" width="29%" /> 
  <img src="Figures/Fig3.png" width="28%" /> 
</p>

Numerical code has been built on Anaconda open-source Python distribution platform using Spyder IDE. Code contains script files for training adn predicting phase volume-preserving dynamics with LocSympNets and SymLocSympNets, considering three examples: learning linear traveling wave solutions to the semi-discretized advetion equation, periodic trajectories of the Euler equations of the motion of a free rigid body, and quasi-periodic solutions of the charged particle motion in an electromagnetic field.

### Technical instructions
- Volume-preserving dynamics is defined in file <font face="Arial" DynamicalSystems/VolumePreservingODEs.py </font>.
- ddd <font face="Arial">Your text here.</font>
