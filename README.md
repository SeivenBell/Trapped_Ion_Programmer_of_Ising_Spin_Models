## Trapped_Ion_Programmer_of_Ising_Spin_Models

# Project Description and Goal:

This project aims to address a fundamental challenge in quantum simulation using trapped ions, a leading platform for emulating quantum many-body systems. These systems are crucial across various fields of physics but are often intractable by classical computation methods. Our project leverages trapped ions, noted for their versatility in simulating many-body spin systems. These ions, encoded in optical or hyperfine states, can interact through long-range phonon modes, allowing for the engineering of arbitrary spin–spin interaction graphs.

The project's core objective is to efficiently determine the control parameters (like laser frequencies and intensities) required for generating specific spin–spin interaction graphs in trapped ion quantum simulators. This task is complex, often presenting as an inverse problem that traditionally required individual nonlinear optimizations of control parameters. However, our approach introduces a machine learning technique that employs artificial neural networks to find practical, verifiable, and useful solutions to this challenge.

Using a machine learning model, we demonstrate the ability to produce highly accurate spin–spin interaction graphs for various lattice geometries, including square, triangular, kagome, and cubic lattices. This method represents a significant advancement in the field of quantum information processing, as it facilitates the simulation of dynamic interactions and complex problems, such as quenches and transport issues, with greater efficiency and scalability. Our model, executed on a single GPU workstation, suggests potential for expanding these simulations to systems with hundreds of ions, marking a leap forward in practical quantum computing capabilities.

In essence, this project exemplifies the intersection of machine learning and quantum physics, providing a novel approach to overcome the limitations of traditional quantum simulation methods and opening new possibilities for exploring complex quantum systems.


# Contributing:

# Severyn Balaniuk (@SeivenBell) 
# Sebastian Barba Carvajal (@esbarbac) 
# Hassan Subhi (@hassansubhit) 
# Filip Popovic (@ocif) 
# Jessica Bohm (@jessicabohm)


Project Tree:
```
Trapped_Ion_Programmer_of_Ising_Spin_Models
|
├───scripts
│   └───Results
├───TrICal  (ION properties lib)
│   ├───docs
│   │   ├───build     
│   │   └───source
|   |
│   ├───trical (specific classes and functions)
│   │   ├───classes
│   │   ├───control
│   │   ├───data
│   │   └───misc
│   ├───trical.egg-info
│   └───tutorial
└───triprism (Our model and data generator)
```



# Usage

To run the Project, first:

```bash
cd \Trapped_Ion_Programmer_of_Ising_Spin_Models\TrICal into the TriCal folder, then setup the trical library.
```
Then, setup the trical library.

```bash
 python setup.py build
```

then 

```bash
 python setup.py install
``` 


once thats done, 
```bash
cd \Trapped_Ion_Programmer_of_Ising_Spin_Models\scripts
``` 

and run

```bash
 python experiments.py 
```


# Dependencies:

trical
triprism
os 
sys
itertools
numpy
torch
matplotlib
seaborn 



# Files:

TriCal library
Used to generate data and to program experiment

experiment.py
Description: Script for simulating and analyzing quantum systems.

Workflow:
Environment Setup: Includes system path adjustments and matplotlib configuration.
Experiment Tracking: Initialization of MLflow for experiment tracking.
Classes and Functions: Description of utility classes and functions used in experiments.

data.py
Description: Contains functions related to data handling for the project.
Functions:
generate_random_interactions: Generates random spin interactions. Includes arguments such as system size, batch size, and computation device.

model.py
Description: Defines the neural network model for decoding spin interactions.
Classes:
SpinDecoder: A neural network module for decoding spin interactions. Includes attributes like system size and various physical parameters.



## FFNN Model Explanation

# SpinDecoder

The SpinDecoder class is a neural network module designed to decode spin interactions in a quantum system. It operates within the context of trapped ions and quantum simulations, translating intricate physical interactions into a computable format. Key attributes and functionalities include:

Attributes:

N: Represents the size of the spin system.
Physical parameters like hbar, m, w, b, mu, deltak, eta, nu, w_mult_nu are defined, each corresponding to specific aspects of the quantum system, like mass, frequencies, and interaction strengths.
Methods:

vectorize_J: Converts a spin interaction matrix into a vector format by extracting its upper triangular part.
matrixify_J: Reverses the process of vectorize_J, reconstructing a matrix from its vectorized form.
forward: The core function where input tensors representing spin interactions are processed through various tensor operations to simulate the quantum spin interactions.
RabiEncoder
The RabiEncoder class encodes Rabi frequencies for a quantum system, essential for simulating spin-state transitions in trapped ions. Key components include:

Attributes:

N: Size of the spin system.
N_h: Number of hidden units in the linear layers of the network.
activation: Activation function used in the network layers.
layers: A sequential container comprising linear layers and activation functions.
Methods:

forward: Processes the input tensor through the neural network layers to generate a matrix of Rabi frequencies.
forwardOmega: Normalizes the output of the forward method, ensuring that the encoded frequencies are consistent and scaled correctly.
PrISM
The PrISM (Programmable Interactions in Spin Models) class integrates both the encoder (RabiEncoder) and decoder (SpinDecoder) to form a complete system for processing quantum system interactions. It functions as the central model in our project, capable of simulating a variety of interaction graphs in quantum spin systems.

Attributes:

encoder: An instance of RabiEncoder.
decoder: An instance of SpinDecoder.
Methods:

reset_parameters: Initializes the model's parameters using Xavier uniform initialization, ensuring a consistent starting point for training.
forward: Executes a forward pass through the PrISM model, effectively encoding and then decoding the input tensor.
reconstruction_loss: Calculates the reconstruction loss, which is a measure of how well the model's output matches the expected spin interactions.

# - AI Statement:

ChatGPT assisted in designing the architecture, providing explanations on how functions
 and physics works. It also helped us to understand complicated TriCal library and it's functions, 
 clarifying their context and application. 
 Most importantly, AI was instrumental in handling debugging, offering insights into 
 the errors I encountered. This guidance was crucial in reducing unnecessary stress and frustration during the development, 
 training, and evaluation stages of the model.

 


# License
MIT

# Acknowledgements
Paper we replicated and used as a reference: 

Teoh, Y. H., Drygala, M., Melko, R. G., & Islam, R. (2020). Machine learning design of a trapped-ion quantum spin simulator. Quantum Science and Technology, 5(2), 024001. https://doi.org/10.1088/2058-9565/ab657a ​

