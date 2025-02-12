__version__ = '0.1.0'

from lsdo_rotor.core.BEM.BEM_caddee import BEM, BEMParameters, evaluate_multiple_BEM_models
from lsdo_rotor.core.pitt_peters.pitt_peters_m3l import PittPeters, PittPetersParameters, evaluate_multiple_pitt_peters_models
from lsdo_rotor.utils.helper_classes import AcStates, RotorMeshes
from lsdo_rotor.utils.atmosphere_model import AtmosphericProperties
from lsdo_rotor.core.rotor_model import RotorAnalysis
from lsdo_rotor.utils.atmosphere_model import get_atmosphere
from lsdo_rotor.utils.print_output import print_output
from pathlib import Path


_REPO_ROOT_FOLDER = Path(__file__).parents[0]
CUSTOM_ML_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'airfoil' / 'ml_trained_models'