"""
Quantum System Evaluation Script

This script is designed for the evaluation and visualization of a neural network model that has been trained for 
quantum systems simulation. It integrates PyTorch for neural network operations, matplotlib and seaborn for data visualization, 
and MLflow for experiment tracking and model retrieval.

Functions and Workflow:

1. Matplotlib Configuration:
    - Sets up parameters for plotting to ensure consistent visualization styles.

2. MLflow Integration:
    - Initializes or retrieves an MLflow experiment for tracking the simulation runs and loads the model from the MLflow tracking server.

3. Utility Class `_RunName`:
    - A class designed to generate unique names for MLflow runs, facilitating the identification of specific model runs or experiments.

4. Model Loading:
    - Searches for and loads a specific neural network model based on parameters set in the script. The model is expected 
    to be related to quantum system simulations.

5. Quantum Simulation Parameters:
    - Defines parameters for quantum simulations and processes interaction matrices for analysis.

6. Model Testing and Visualization:
    - Generates and visualizes quantum interaction matrices using 3D bar plots to evaluate the model's performance.

7. Fidelity Calculation and Plotting:
    - Calculates and plots a fidelity metric between different interaction matrices, assessing the accuracy or quality 
    of the model's predictions.

"""




# Imports
import os
import sys

sys.path.append(os.path.abspath("../"))

########################################################################################

import itertools as itr
import numpy as np

import torch
from torch.optim import lr_scheduler
from torch.optim import Adam

from torchinfo import summary

import mlflow

import matplotlib
from matplotlib import colors, cm, patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

import trical
import triprism

########################################################################################

# Package parameters

matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

experiment_name = "TrIPrISM"
try:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
except:
    experiment_id = mlflow.create_experiment(experiment_name)

########################################################################################


class _RunName:
    def __init__(self, params, repetition=None):
        self.params = params
        self.repetition = repetition if repetition else 0
        pass

    def __add__(self, other):
        self.repetition += other
        return self

    def __radd__(self, other):
        self.repetition += other
        return self

    def __repr__(self):
        s = "-".join(
            [
                "triprism",
                *["{}={}".format(k.lower(), v) for k, v in self.params.items()],
                "rep={}".format(self.repetition),
            ]
        )
        return s


########################################################################################

# Load model from mlflow tracking server

existing_runs = mlflow.search_runs(experiment_id)

run_name = _RunName(
    dict(
        N=,
        encoder="",
        N_h=,
        # rank=7,
        lambda_dnr=0,
    ),
    repetition=0,
)
run_id = existing_runs["run_id"][
    existing_runs["tags.mlflow.runName"] == str(run_name)
].values[0]

model_uri = 
model = 
model.to(device)

########################################################################################

N = model.decoder.N

alpha = 
J = 
J = 
J[..., range(N), range(N)] = 0

x = 
x = 
J = 

Jnn = 
Omega = 

J = J / torch.norm(J, dim=(-1, -2), keepdim=True)
Jnn = Jnn / torch.norm(Jnn, dim=(-1, -2), keepdim=True)

# ########################################################################################

fig = plt.figure()
ax = fig.subplots(1, 3, subplot_kw=dict(projection="3d"))

i = 10

norm = colors.Normalize(
    torch.stack([J[i], Jnn[i]]).min(), torch.stack([J[i], Jnn[i]]).max()
)
norm2 = colors.Normalize(min(Omega[i].min(), 0), max(Omega[i].max(), 0))

XX, YY = np.meshgrid(range(N), range(N))
ax[0].bar3d(
    XX.flatten(),
    YY.flatten(),
    np.zeros(XX.shape).flatten(),
    1,
    1,
    J[i].cpu().detach().numpy().flatten(),
    shade=True,
    color=cm.viridis(norm(J[i].cpu().detach().numpy().flatten())),
)
ax[1].bar3d(
    XX.flatten(),
    YY.flatten(),
    np.zeros(XX.shape).flatten(),
    1,
    1,
    Jnn[i].cpu().detach().numpy().flatten(),
    shade=True,
    color=cm.viridis(norm(Jnn[i].cpu().detach().numpy().flatten())),
)
ax[2].bar3d(
    XX.flatten(),
    YY.flatten(),
    np.zeros(XX.shape).flatten(),
    1,
    1,
    Omega[i].cpu().detach().numpy().flatten(),
    shade=True,
    color=cm.plasma(norm2(Omega[i].cpu().detach().numpy().flatten())),
)

ax[2].set_zlim(
    min(Omega[i].min().cpu().detach().numpy(), 0),
    max(Omega[i].max().cpu().detach().numpy(), 0),
)


########################################################################################

F = (J * Jnn).sum(dim=(-1, -2))

fig = plt.figure()
ax = fig.subplots(1, 1)

ax.plot(alpha.cpu().detach().numpy(), F.cpu().detach().numpy())

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(
    r"$\mathcal{F} \left( J_{ij} (\Omega^{\mathrm{NN}}), 1 / |i-j|^\alpha \right)$"
)
