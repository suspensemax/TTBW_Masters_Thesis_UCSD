import array_mapper as am
import numpy as np
from caddee.core.caddee_core.system_representation.component.component import Component


class PrescribedActuation:
    '''
    Defines a "solver" for defining actuation within a component. This is the simplest solver
    which is just prescribing the value of the actuation. This is just a parent class and does
    not provide functionality.
    '''

    def __init__(self, component:Component, axis_origin:am.MappedArray, axis_vector:am.MappedArray, value:np.ndarray=None) -> None:
        self.component = component
        self.axis_origin = axis_origin
        self.axis_vector = axis_vector
        self.value = value


class PrescribedRotation(PrescribedActuation):
    '''
    Defines a "solver" for defining rotational actuation within a component. This is the simplest solver
    which is just prescribing the value of the rotation.
    '''

    def __init__(self, component: Component, axis_origin: am.MappedArray, axis_vector:am.MappedArray, value:np.ndarray=None) -> None:
        super().__init__(component, axis_origin, axis_vector, value)

        if self.value is None:
            self.value = 0.
        self.units = 'radians'

    def set_rotation(self, name:str, value:np.ndarray, units:str='radians'):
        self.name = name
        self.value = value
        self.units = units

    def assemble_csdl(self):
        '''
        Assembles the CSDL model to perform this operation.
        '''
        from caddee.core.csdl_core.system_representation_csdl.prescribed_rotation_csdl import PrescribedRotationCSDL
        return PrescribedRotationCSDL(prescribed_rotation = self)


class PrescribedTranslation(PrescribedActuation):
    '''
    Defines a "solver" for defining translational actuation within a component. This is the simplest solver
    which is just prescribing the value of the translation.
    '''
    
    def assemble_csdl(self):
        '''
        Assembles the CSDL model to perform this operation.
        '''
        from caddee.core.csdl_core.system_representation_csdl.system_representation_csdl import PrescribedTranslationCSDL
        return PrescribedTranslationCSDL(prescribed_translation = self)
