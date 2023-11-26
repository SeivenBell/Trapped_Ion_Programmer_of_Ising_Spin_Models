
"""
Training Script for PrISM

Workflow Steps:

1. Library Imports:
    - Import `trical` and `triprism` for quantum mechanics and simulation-specific functionalities.

2. MLflow Experiment Setup:
    - Initialize or create an MLflow experiment for tracking and logging the simulation runs.

3. Utility Class and Function:
    - `_RunName`: A utility class to generate unique run names for MLflow tracking.
    - `append_log10`: A function to log the base-10 transformation of specified parameters.

4. Quantum System Initialization:
    - Define system-specific parameters like the number of ions, mass, and frequencies.
    - Utilize `trical` to set up the trapped ion system (`ti`) and polynomial potential (`tp`).

5. Neural Network Setup:
    - Define neural network hyperparameters like hidden units and rank.
    - Initialize the k-rank Rabi encoder, spin decoder, and the composite model (`PrISM`) for simulation.

6. Training and Evaluation Logic:
    - Implement `evaluate_metrics` function to compute training and evaluation metrics.
    - Define training and evaluation loops for the model.

7. Model Saving

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

matplotlib.rcParams["figure.figsize"] = 
matplotlib.rcParams["font.size"] = 
matplotlib.rcParams["text.usetex"] = 
matplotlib.rcParams["mathtext.fontset"] = ""
matplotlib.rcParams["font.family"] = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_name = ""
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
                *[
                    "{}={:.5g}".format(k.lower(), v)
                    if isinstance(v, float)
                    else "{}={}".format(k.lower(), v)
                    for k, v in self.params.items()
                ],
                "rep={}".format(self.repetition),
            ]
        )
        return s


def append_log10(d, exclude=[]):
    _d = {}

    for k, v in d.items():
        try:
            v = float(v)
        except:
            continue

        if k not in exclude and not k.startswith("log10_"):
            _d["log10_" + k] = np.log10(v)

    d.update(_d)
    pass


########################################################################################

N = 
m = trical.misc.constants.convert_m_a(171)
l = 

omega = 2 * np.pi * np.array([5, 5.1, 1.05]) * 1e6
alpha = np.zeros([3, 3, 3])
alpha[tuple(np.eye(3, dtype=int) * 2)] = m * omega**2 / 2
tp = trical.classes.PolynomialPotential(alpha)

ti = trical.classes.TrappedIons(N, tp)
ti.normal_modes(block_sort=True)

########################################################################################

omicron = 0.1
wx = ti.w[:N]
_wx = np.concatenate([wx[0:1] + (wx[0:1] - wx[-1:]) / N, wx], axis=0)
mu = _wx[1:] + omicron * (_wx[:-1] - _wx[1:])
mu = torch.from_numpy(mu)

#######################################################################################

N_h = 
rank = 

encoder = triprism.kRankRabiEncoder(1, N, N_h)
decoder = triprism.SpinDecoder(ti, mu)

model = triprism.PrISM(encoder=encoder, decoder=decoder)
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

smoothing_factor = 
dynamic_range_regularization_factor = 

########################################################################################


def evaluate_metrics(model, x, mode="train"):
    if mode == "train":
        
    elif mode == "eval":
       
    else:
        raise 

    metrics = {}
    metrics["smoothed_infidelity"] = model.smoothed_reconstruction_loss(
        x, smoothing_factor
    )
    metrics["infidelity"] = model.reconstruction_loss(x)
    metrics["fidelity"] = 1 - metrics["infidelity"]
    metrics["dynamic_range_penalty"] = model.dynamic_range_penalty(x)
    metrics["loss"] = (
        metrics["smoothed_infidelity"]
        + dynamic_range_regularization_factor * metrics["dynamic_range_penalty"]
    )
    return metrics


########################################################################################

for _, rank in itr.product(
    range(5),
    set([int(i) for i in np.linspace(1, N, N)]),
):
    
    model.to(device=device)

    ########################################################################################

    optimizer = optimizer_algorithm(model.parameters(), lr=lr)
    schedule = schedule_algorithm(optimizer, **schedule_params)

    ########################################################################################

    existing_runs = mlflow.search_runs(experiment_id)
    if not existing_runs.empty:
        existing_runs = existing_runs["tags.mlflow.runName"]

    run_name = _RunName(
        dict(
            N=N,
            encoder=model.encoder.__class__.__name__,
            rank=rank,
            N_h=N_h,
            lambda_dnr=dynamic_range_regularization_factor,
        )
    )
    if not existing_runs.empty:
        while existing_runs.str.contains(str(run_name)).any():
            run_name += 1

    ########################################################################################

    log_every = 
    with mlflow.start_run(experiment_id=experiment_id, run_name=str(run_name)) as run:
        script_path = os.path.realpath(__file__)
        mlflow.log_artifact(script_path, "scripts")

        ########################################################################################

        params = dict(
            N=N,
            N_h=N_h,
            rank=rank,
            model=model.__class__.__name__,
            encoder=model.encoder.__class__.__name__,
            decoder=model.decoder.__class__.__name__,
            N_trainable_params=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            N_params=sum(p.numel() for p in model.parameters()),
            initial_lr=lr,
            lr_scheduler=schedule.__class__.__name__,
            **{"lr_" + k: v for k, v in schedule_params.items()},
            optimizer=optimizer.__class__.__name__,
            epochs=N_epochs,
            batch_size=batch_size,
            repetition=run_name.repetition,
            smoothing_factor=smoothing_factor,
            dynamic_range_regularization_factor=dynamic_range_regularization_factor,
        )

        append_log10(params, exclude=["repetition"])
        mlflow.log_params(params)

        ########################################################################################

        step = 0
        for epoch in range(N_epochs):
            J_train = triprism.generate_random_interactions(N, batch_size, device)

            

            ########################################################################################

            if epoch % log_every == 0 or epoch == N_epochs - 1:
                

                ########################################################################################

                metrics = dict(
                    epoch=epoch + 1,
                )
                for k, v in train_metrics.items():
                    metrics["train_" + k] = v
                for k, v in val_metrics.items():
                    metrics["val_" + k] = v
                append_log10(metrics)

                ########################################################################################

                mlflow.log_metrics(
                    metrics,
                    step=step,
                )

                ########################################################################################

                print(
                    
                    
                    
                    
                    
                )
                sys.stdout.flush()

        ########################################################################################

        conda_env = {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10.11",
                "pip<=23.1.2",
                {"pip": ["mlflow", "torch==2.0.1cu117", "cloudpickle==2.2.1"]},
            ],
        }

        mlflow.pytorch.log_model(model, "checkpoint", conda_env=conda_env)

        ########################################################################################
