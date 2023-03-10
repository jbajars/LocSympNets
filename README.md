# LocSympNets
PyTorch code for training locally-symplectic neural networks LocSympNets, and symmetric version SymLocSympNets, for learning phase volume-preserving linear and nonlinear dynamics. Using the code, please cite Jānis Bajārs, *Locally-symplectic neural networks for learning volume-preserving dynamics*, Journal of Computational Physics **476**:111911, 2023 ([DOI: 10.1016/j.jcp.2023.111911](https://www.sciencedirect.com/science/article/pii/S0021999123000062?via%3Dihub)).

File `LocSympNets_PureCode_WithoutData.zip` contains pure code without precomputed training data, images, and examples of pre-trained neural networks.

<p float="left">
  <img src="Figures/Fig1.png" width="39%" />
  <img src="Figures/Fig2.png" width="29%" /> 
  <img src="Figures/Fig3.png" width="28%" /> 
</p>

Numerical code has been built on Anaconda's open-source Python distribution platform using Spyder IDE. Code contains script files for training and predicting phase volume-preserving dynamics with LocSympNets and SymLocSympNets, considering three examples: learning linear traveling wave solutions to the semi-discretized advection equation, periodic trajectories of the Euler equations of the motion of a free rigid body, and quasi-periodic solutions of the charged particle motion in an electromagnetic field. The rigid body example contains two cases: learning a single periodic trajectory and the whole dynamics from randomly sampled training data.

#### Instructions to run the code
- Volume-preserving differential equations are defined in file `DynamicalSystems/VolumePreservingODEs.py`.
- Training and testing data for four example cases are computed in the following files: `DynamicalSystems/TrainingData_AdvectionEq.py`, `DynamicalSystems/TrainingData_RigidBody_Single.py`, `DynamicalSystems/TrainingData_RigidBody_Whole.py`, and `DynamicalSystems/TrainingData_ChargedParticle.py`, respectively.
- All training and testing data is saved in associated problem folders `DynamicalSystems/SavedTrainingData/`.
- All neural network functionas are defined in folder `NeuralNetworkFnc`.
- File `NeuralNetworkFnc/module-class.py` contains LocSympNets and SymLocSympNets modules.
- File `NeuralNetworkFnc/training-class.py` contains neural network training functions.
- Additional neural network supporting functions are defined in files `NeuralNetworkFnc/mySequential.py` and `NeuralNetworkFnc/custom_dataset.py`.
- LocSympNets training script files for four example cases are: `training_AdvectionEq_script.py`, `training_RigidBody_Single_script.py`, `training_RigidBody_Whole_script.py`, and `training_ChargedParticle_script.py`, respectively. 
- SymLocSympNets training script files for four example cases are: `training_AdvectionEq_sym_script.py`, `training_RigidBody_Single_sym_script.py`, `training_RigidBody_Whole_sym_script.py`, and `training_ChargedParticle_sym_script.py`, respectively. 
- All trained neural networks are saved in associated problem folders `SavedNeuralNets/`.
- LocSympNets prediction and testing script files for four example cases are: `predictions_AdvectionEq_script.py`, `predictions_RigidBody_Single_script.py`, `predictions_RigidBody_Whole_script.py`, and `predictions_ChargedParticle_script.py`, respectively. 
- SymLocSympNets training and testing script files for four example cases are: `predictions_AdvectionEq_sym_script.py`, `predictions_RigidBody_Single_sym_script.py`, `predictions_RigidBody_Whole_sym_script.py`, and `predictions_ChargedParticle_sym_script.py`, respectively. 
- Produced images are saved in associated problem folders `Figures/`.
