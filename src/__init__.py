"""
MultiFuse: Multimodal Weighted Consensus for Action Recognition in Videos

A PyTorch-based framework for multimodal video action recognition using
weighted consensus fusion of multiple streams (RGB, optical flow, etc.).
"""

__version__ = "0.1.0"
__author__ = "Nico Grassetto"

# Import existing modules
from .segmentation import *

# Import new organized modules
try:
    from .models import *
    from .training import *
    from .data import *
except ImportError:
    # Modules might not be fully implemented yet
    pass