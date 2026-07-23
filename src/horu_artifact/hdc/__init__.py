"""Hyperdimensional encoding and prototype operations."""

from .encoder import NonlinearEncoder, make_projection
from .prototype import PrototypeMemory

__all__ = ["NonlinearEncoder", "PrototypeMemory", "make_projection"]
