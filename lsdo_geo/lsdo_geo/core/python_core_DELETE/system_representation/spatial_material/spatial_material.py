import numpy as np

from utils.caddee_base import CADDEEBase
from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
from lsdo_geo.caddee_core.system_representation.component.component import Component
from lsdo_geo.caddee_core.system_representation.utils.material import Material
from lsdo_geo.primitives.primitive import Primitive
from lsdo_geo.primitives.b_splines import b_spline_surface, b_spline_functions
import scipy.sparse as sps
import vedo
import array_mapper as am


class SpatialMaterial:
    '''
    Definition of a single material over some parametric space
    
    Parameters
    ----------

    
    '''
    def __init__(self, name:str, index:int, level_set_primitive:Primitive=None, h=1e-4, properties:dict={}):
        self.name = name
        self.ls_primitive = level_set_primitive
        self.h = h
        self.properties=properties
        properties['name'] = name
        properties['index'] = index


    def import_material(self, material:Material):
        pass

    def fit_levelset(self, data:np.ndarray):
        pass

    def _heaviside(self, x):
        if x < -self.h:
            return 0
        elif x > self.h:
            return 1
        else: 
            return -1/4*(x/self.h)**3+3/4*(x/self.h)+1/2
    
    def _heaviside_derivative(self, x):
        if x<-self.h or x > self.h:
            return 0
        return -3/4*(x/self.h)**2/self.h+3/4/self.h

    def fit_property(self, parametric_coordinates, function_value):
        '''
        Fits new primitive to input data
        '''
        pass

    def evaluate_points(self, u_vec, v_vec):
        '''
        Evaluates the level set function at the parametric coordinates.
        '''
        points=self.primitive.evaluate_points(u_vec, v_vec)

        for i in range(0,points.shape[0]):
            points[i] = self._heaviside(points[i])

        return points

    def evaluate_derivative(self, u_vec, v_vec):
        '''
        Evaluates the derivative of the level set function at the parametric coordinates.
        '''
        # num_coefficients = self.shape[0] * self.shape[1]
        
        # basis1 = self.compute_derivative_evaluation_map(u_vec, v_vec)
        # derivs1 = basis1.dot(self.coefficients.reshape((num_coefficients, 3)))

        # return derivs1 
        pass

    def evaluate_second_derivative(self, u_vec, v_vec):
        '''
        Evaluates the second derivative of the level set function at the parametric coordinates.
        '''
        # num_coefficients = self.shape[0] * self.shape[1]
        
        # basis2 = self.compute_second_derivative_evaluation_map(u_vec, v_vec)
        # derivs2 = basis2.dot(self.coefficients.reshape((num_coefficients, 3)))

        # return derivs2
        pass