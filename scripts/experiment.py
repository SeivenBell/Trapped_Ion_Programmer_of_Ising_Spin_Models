"""
This script is designed for the simulation and analysis of quantum systems, particularly focused on trapped ions using machine learning techniques. 


Workflow:
1. Environment Setup:
   - Adjust the system path to include custom modules.
   - Configure matplotlib parameters for consistent plotting styles.

2. Experiment Tracking Setup:
   - Initialize MLflow experiment tracking with an experiment name.

3. Classes and Functions:
   - `_RunName`: A utility class for generating consistent naming conventions for MLflow runs.
   - `append_log10`: A function to append log-transformed parameters for MLflow logging.

4. Quantum System Initialization:
   - Define system parameters like the number of ions, mass, frequencies.
   - Initialize trapped ion system (`ti`) and potential (`tp`) using 'trical'.

5. Neural Network Configuration:
   - Set hyperparameters like the number of hidden units, rank for the encoder.
   - Initialize the encoder, decoder, and model (`PrISM`) for the quantum simulation.

7. Model Summary:
   - Display a summary of the model architecture using `torchinfo`.

8. Training Configuration:
   - Define training parameters like epochs, batch size, learning rate.
   - Configure optimizer and learning rate scheduler.

9. Training and Evaluation Functions:
   - `evaluate_metrics`: Function to compute various metrics during training and validation.

10. Training Loop:
    - Iterative training of the model using generated random interactions.
    - Logging training progress and metrics using MLflow.

11. Model Saving:
    - Save the trained model using MLflow with the specified environment details.

Usage:
- Modify quantum system parameters (e.g., number of ions, mass, frequencies) as needed for specific simulations.
- Adjust neural network hyperparameters (e.g., number of hidden units, learning rate) to experiment with different model configurations.
- Run the script in an environment where MLflow is accessible for experiment tracking.
- The script automatically logs training progress and model details, and saves the final model for later use or analysis.

"""

# Imports

import os
import sys

sys.path.append(os.path.abspath("../"))

########################################################################################

import itertools as itr
import numpy as np

import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW

from torchinfo import summary

#import mlflow

import matplotlib
from matplotlib import colors, cm, patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

import trical
import triprism.old_model


########################################################################################

# Package parameters

matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################

N = 10 # number of ions in configuration
m = trical.misc.constants.convert_m_a(171)  # Mass of Yb+ ion in atomic mass units
l = 1e-6  # Characteristic length scale in meters

# Define trap frequencies for each dimension in Hz
omega = 2 * np.pi * np.array([5, 5.1, 0.41]) * 1e6  # Trap frequencies in Hz

# Initialize the quadratic coefficients for the trapping potential
alpha = np.zeros([3, 3, 3])  # Tensor to hold quadratic coefficients
alpha[tuple(np.eye(3, dtype=int) * 2)] = m * omega**2 / 2  # Populate diagonal elements
tp = trical.classes.PolynomialPotential(alpha)  # Create a polynomial potential object

# Initialize the TrappedIons class and calculate normal modes
ti = trical.classes.TrappedIons(N, tp)
ti.normal_modes(block_sort=True)

########################################################################################

# Define the detuning parameter and calculate modified trap frequencies
omicron = 0.1  # Detuning parameter
wx = ti.w[:N]  # Extract trap frequencies for each ion
_wx = np.concatenate([wx[0:1] + (wx[0:1] - wx[-1:]) / N, wx], axis=0)
mu = _wx[1:] + omicron * (_wx[:-1] - _wx[1:])  # Modify frequencies based on detuning
mu = torch.from_numpy(mu)  # Convert to a torch tensor

#######################################################################################

# Define model architecture parameters
N_h = 1024 # Replace with the number of hidden layers
encoder = triprism.RabiEncoder(N, N_h)  # Initialize the encoder
decoder = triprism.SpinDecoder(ti, mu)  # Initialize the decoder

model = triprism.PrISM(encoder=encoder, decoder=decoder)
model.to(device=device)

########################################################################################

# Print a summary of the model
print(summary(model, input_size=[10, int(N * (N - 1) / 2)], device=device))
print("")

#######################################################################################

N_epochs = 250
batch_size = 1000

lr = 0.001
optimizer_algorithm = AdamW
schedule_params = {"factor": 1}
schedule_algorithm = lr_scheduler.ConstantLR # use for some optimizer algorithms
optimizer = optimizer_algorithm(model.parameters(), lr=lr)
schedule = schedule_algorithm(optimizer, **schedule_params) # use for some optimizer algorithms

########################################################################################

step = 0
train_fidelities = []
val_fidelities = []

train_losses = []
val_losses = []


# Training loop
for epoch in range(N_epochs):
    J_train = triprism.generate_random_interactions(N, batch_size, device)
    J_val = triprism.generate_random_interactions(N, batch_size, device)

    optimizer.zero_grad()

    train_infidelity = model.train().reconstruction_loss(J_train)

    running_loss = train_infidelity
    running_loss.backward()
    optimizer.step()
    schedule.step()
    step += 1

    # Calculate validation infidelity
    val_infidelity = model.eval().reconstruction_loss(J_val)
    val_loss = val_infidelity
    # Update metrics
    metrics = dict(
        epoch=epoch + 1,
        train_loss=running_loss,
        val_loss=val_loss,
        train_infidelity=train_infidelity,
        val_infidelity=train_infidelity,
        train_fidelity=1 - train_infidelity,
        val_fidelity=1 - train_infidelity,
    )

    # save performance metrics after each epoch
    train_fidelities.append(metrics["train_fidelity"].cpu().detach().numpy())
    val_fidelities.append(metrics["val_fidelity"].cpu().detach().numpy())

    train_losses.append(metrics["train_loss"].cpu().detach().numpy())
    val_losses.append(metrics["val_loss"].cpu().detach().numpy())

    ########################################################################################

    print(
        "{:<180}".format(
            "\r"
            + "[{:<60}] ".format(
                "=" * ((np.floor((epoch + 1) / N_epochs * 60)).astype(int) - 1) + ">"
                if epoch + 1 < N_epochs
                else "=" * 60
            )
            + "{:<40}".format(
                "Epoch {}/{}: Fidelity(Train) = {:.5f}, Fidelity(Val) = {:.5f}".format(
                    metrics["epoch"],
                    N_epochs,
                    metrics["train_fidelity"],
                    metrics["val_fidelity"],
                )
            )
        ),
        end="",
    )
    sys.stdout.flush()
    
    #     # Severyn's possible oprimized version
    # progress_bar = "=" * (int(np.floor((epoch + 1) / N_epochs * 60)) - 1) + ">" \
    # if epoch + 1 < N_epochs else "=" * 60

    # print(
    #     "\r[{:<60}] Epoch {}/{}: Fidelity(Train) = {:.5f}, Fidelity(Val) = {:.5f}".format(
    #         progress_bar, epoch + 1, N_epochs, metrics['train_fidelity'], metrics['val_fidelity']
    #     ),
    #     end=""
    # )
    # sys.stdout.flush()

########################################################################################

alpha = torch.linspace(0, 5, 101).to(device)
J = torch.from_numpy(np.indices((N, N))).to(device)
J = 1 / torch.abs(J[0] - J[1])[None, :, :] ** alpha[:, None, None]
J[..., range(N), range(N)] = 0

x = model.decoder.vectorize_J(J)
x = x / torch.norm(x, dim=-1, keepdim=True)
J = model.decoder.matrixify_J(x)

Jnn = model.decoder.matrixify_J(model(x))
Omega = model.encoder.forwardOmega(x)

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

fig.savefig("plot1.png")

########################################################################################

F = (J * Jnn).sum(dim=(-1, -2))

fig = plt.figure()
ax = fig.subplots(1, 1)

ax.plot(alpha.cpu().detach().numpy(), F.cpu().detach().numpy())

ax.set_label(r"$\alpha$")
ax.set_ylabel(
    r"$\mathcal{F} \left( J_{ij} (\Omega^{\mathrm{NN}}), 1 / |i-j|^\alpha \right)$"
)

fig.savefig("plot2.png")


# save loss and fidelity plots
def plot_train_vs_val(train, val, title, save_path):
    epochs = np.arange(len(train))
    plt.cla()
    plt.plot(epochs, train, label="Train")
    plt.plot(epochs, val, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)

plot_train_vs_val(train_losses, val_losses, "Loss", "loss.png")
plot_train_vs_val(train_fidelities, val_fidelities, "Fidelity", "fidelity.png")


