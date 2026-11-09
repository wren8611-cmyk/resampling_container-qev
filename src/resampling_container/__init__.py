"""
resampling_container
--------------------
A lightweight Python library for jackknife and bootstrap resampling.

Author: Wren Yamada (2026)
"""

from .container import Container
from .jackknife import Jackknife
from .bootstrap import Bootstrap

__author__ = "Wren Yamada"
__version__ = "0.0.0"

__all__ = ["Container", "Jackknife", "Bootstrap"]