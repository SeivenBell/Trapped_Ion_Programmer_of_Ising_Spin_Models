import torch
from .model import SpinDecoder

########################################################################################
def generate_random_interactions(N: int, batch_size: int, device: torch.device):
    """
    Generate random spin interactions.

    Args:
    - N (int): The size of the spin system.
    - batch_size (int): The number of interaction sets to generate.
    - device (torch.device): The device on which the computations will be executed.

    Returns:
    - torch.Tensor: Tensor representing the random spin interactions.
    """

    # Calculate the number of unique interactions in a system of size N
    num_interactions = int(N * (N - 1) / 2)

    # Generate a random tensor for interactions
    J = torch.randn((batch_size, num_interactions), device=device)

    # Normalize J for each set of interactions in the batch
    J_normalized = J / torch.norm(J, dim=1, keepdim=True)

    return J_normalized
