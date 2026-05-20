from collections.abc import Callable

from loguru import logger
from torch import nn

# Central model registry
_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """
    Decorator to register a model class or factory function.
    """

    def decorator(cls: Callable[..., nn.Module]):
        _MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model architecture: '{name}'")
        return cls

    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Instantiates a registered model with given hyperparameters.

    Args:
        name (str): Registered identifier of the model.
        **kwargs: Arguments passed to model's constructor.

    Returns:
        nn.Module: The instantiated model.
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        logger.error(f"Model '{name}' not found. Available architectures: {available}")
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    logger.info(f"Instantiating model: '{name}'")
    return _MODEL_REGISTRY[name](**kwargs)
