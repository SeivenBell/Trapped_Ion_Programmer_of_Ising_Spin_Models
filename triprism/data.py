import torch
from .model import SpinDecoder

########################################################################################


def generate_feasible_interactions(N: int, batch_size: int, spin_decoder: SpinDecoder, device: torch.device):
    """
    Generates feasible interaction matrices using the provided spin decoder.

    This function creates random Omega matrices and then uses the spin decoder to transform these into interaction matrices (J). This is typically used for generating synthetic data for training or testing in quantum systems simulations.

    Args:
        N (int): The size of the interaction matrix, typically representing the number of ions or spins in the quantum system.
        batch_size (int): The number of interaction matrices to generate.
        spin_decoder (SpinDecoder): An instance of the SpinDecoder class used for transforming Omega matrices into interaction matrices.
        device (torch.device): The computational device (CPU or GPU) where the tensor operations will be performed.

    Returns:
        torch.Tensor: A tensor of interaction matrices (J) of shape (batch_size, N, N).
    """
    return J


def generate_random_interactions(N: int, batch_size: int, device: torch.device):
    """
    Generates random interaction matrices (J) for a given size and batch.

    This function is used to create a batch of random interaction matrices, typically for the purpose of testing or training
    neural network models in quantum system simulations. The interaction matrices are normalized to have a unit norm.

    Args:
        N (int): The size of the interaction matrix, typically representing the number of ions or spins in the quantum system.
        batch_size (int): The number of interaction matrices to generate.
        device (torch.device): The computational device (CPU or GPU) where the tensor operations will be performed.

    Returns:
        torch.Tensor: A tensor of normalized random interaction matrices of shape (batch_size, N * (N - 1) / 2).
    """

    return J
