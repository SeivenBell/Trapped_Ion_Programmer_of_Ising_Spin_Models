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

matplotlib.rcParams["figure.figsize"] = (, )
matplotlib.rcParams["font.size"] = 
matplotlib.rcParams["text.usetex"] = 
matplotlib.rcParams["mathtext.fontset"] = ""
matplotlib.rcParams["font.family"] = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################

N = 
m = 
l = 

omega = 
alpha = 
alpha[tuple(np.eye(3, dtype=int) * 2)] = m * omega**2 / 2
tp = 

ti = 
ti.normal_modes(block_sort=True)

########################################################################################

omicron = 
wx = 
_wx = 
mu = 
mu = 

#######################################################################################

N_h = 
encoder = 
decoder = 

model = 
model.to(device=device)

########################################################################################

print(summary(model, input_size=[10, int(N * (N - 1) / 2)], device=device))
print("")

#######################################################################################

N_epochs = 
batch_size = 

lr =
optimizer_algorithm = 
schedule_params = {"": }
schedule_algorithm = 
optimizer = optimizer_algorithm()
schedule = schedule_algorithm()

########################################################################################

step = 0
for epoch in range(N_epochs):
    
    
    
    
    

    metrics = dict(
        epoch=epoch + 1,
        train_loss=running_loss,
        val_loss=val_loss,
        train_infidelity=train_infidelity,
        val_infidelity=train_infidelity,
        train_fidelity=1 - train_infidelity,
        val_fidelity=1 - train_infidelity,
    )

    ########################################################################################

    print(
       
                )
    
    
    sys.stdout.flush()

########################################################################################

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

i = 

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
