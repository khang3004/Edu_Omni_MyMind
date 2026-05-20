import random

from loguru import logger
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across random, numpy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Selects the optimal hardware device (CUDA, MPS, CPU) available.

    Returns:
        torch.device: The selected PyTorch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using NVIDIA CUDA GPU acceleration.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon Metal Performance Shaders (MPS) acceleration.")
    else:
        device = torch.device("cpu")
        logger.info("Using standard CPU execution.")

    return device
