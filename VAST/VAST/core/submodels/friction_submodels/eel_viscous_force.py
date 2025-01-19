from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class EelViscousModel(Model):
    """
    Compute the viscous force of an eel geometry with a simple surrogate model.

    parameters
    ----------
    v_x : csdl variable [1,]
        array defining the x velocity of the eel

    Returns
    -------
    1. visous_force_coeff : csdl array [1,]
    """
    def initialize(self):
        pass

    def define(self):
        v_x = self.declare_variable(name='v_x')
        C_F = 3.66/1000*(v_x)**(-0.5) * 0.2944404050399099 /0.13826040386294708 * 6

        self.register_output('C_F', C_F)