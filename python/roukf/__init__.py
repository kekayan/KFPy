import importlib.util
import os

# Get the path to the compiled module
spec = importlib.util.find_spec("roukf.roukf_py")
if spec is None:
    raise ImportError("Could not find roukf_py module")

# Import the module
roukf_py = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roukf_py)

# Import specific classes and enums
from .roukf_py import ROUKF, AbstractROUKF, SigmaDistribution

__all__ = ['ROUKF', 'AbstractROUKF', 'SigmaDistribution']
