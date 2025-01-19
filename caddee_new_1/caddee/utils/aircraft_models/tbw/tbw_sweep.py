import csdl
from typing import Union, Tuple
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from dataclasses import dataclass, field
import numpy as np
import caddee.api as cd 
import lsdo_geo as lg
import m3l

@dataclass
class geometryOutputs: 
    wing_sweep : m3l.Variable = m3l.Variable(name='wing_sweep_coefficients', shape = (3,), value= [-10.,0.,-10.])
    # wing_sweep : m3l.Variable

class tbwSweep(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        # self.parameters.declare('component', default=None, types=None)
        self.num_nodes = 3

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self):
        C_x_new = tbwSweepModelCSDL(
            tbwSweepModel=self,
        )
        return C_x_new

    def evaluate(self, wing_sweep_angle: Union[m3l.Variable, None], wing_root_chord_dv: Union[m3l.Variable, None],
                 wing_tip_chord_left_dv: Union[m3l.Variable, None],):
        
        self.name = f"tbw_sweep_model"
        # self.parameters.declare('name', types=str)
        # self.name = self.parameters['name']
        self.arguments = {}
        self.arguments['wing_sweep_angle'] = wing_sweep_angle
        self.arguments['wing_root_chord_dv'] = wing_root_chord_dv
        self.arguments['wing_tip_chord_left_dv'] = wing_tip_chord_left_dv

        wing_sweep = m3l.Variable(name='wing_sweep', shape=(self.num_nodes,), operation=self)


        outputs = geometryOutputs(
            wing_sweep = wing_sweep,
        )
        return outputs


class tbwSweepModelCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='sweep')
        # self.parameters.declare('tbwSweepModel', types=tbwSweep)
        self.parameters.declare('num_nodes_1', default=1)
        # return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes_1']

        wing_sweep_angle = self.declare_variable('wing_sweep_angle',
                                          shape=(num_nodes, 1))
        wing_root_chord_dv = self.declare_variable('wing_root_chord_dv',
                                          shape=(num_nodes, 1))
        wing_tip_chord_left_dv = self.declare_variable('wing_tip_chord_left_dv',
                                          shape=(num_nodes, 1))

        angle_BAC_rad = wing_sweep_angle

        le_root_new_x = 47.231 + (0.25 * wing_root_chord_dv)
        le_root_new_point = np.pad(le_root_new_x, (0, 2), 'constant')

        le_root_new_point[1] = 0.
        le_root_new_point[2] = 6.937

        # wing_le_left_new = np.array([le_root_new_x.value, 85.291, 4.704])
        wing_le_left_new = np.array([le_root_new_point[0], 85.291, 4.704])
        # le_root_new_point_ambigu = np.array([le_root_new_x.value, 0., 6.937])
        le_root_new_point_ambigu = np.array([le_root_new_point[0], 0., 6.937])
        AB = wing_le_left_new - le_root_new_point_ambigu    
        magnitude_AB = np.linalg.norm(AB)

        # wing_le_left_new = system_model.create_input(name = 'wing_le_left_new', shape = (3,), val = np.array([le_root_new_point[0], 85.291, 4.704]))
        # le_root_new_point_ambigu = system_model.create_input(name = 'le_root_new_point_ambigu', shape = (3,), val = np.array([le_root_new_point[0], 0., 6.937]))
        # AB_a = wing_le_left_new - le_root_new_point_ambigu    
        # AB = csdl.sum(AB_a, axes=1)
        # magnitude_AB = csdl.pnorm(AB, axis = 1)
        unit_AB = AB / magnitude_AB

        # Get the direction cosines of AC
        l_AC = unit_AB[0]
        m_AC = unit_AB[1]
        n_AC = unit_AB[2]

        # Calculate the direction ratios of AB
        cos_angle = np.cos(angle_BAC_rad)
        # sweep_cos_angle = system_model.create_input(name = 'sweep_cos_angle', shape = (1,), val = cos_angle)
        # system_model.print_var(sweep_cos_angle)
        sin_angle = np.sin(angle_BAC_rad)

        # cos_angle = csdl.cos(angle_BAC_rad)
        # sin_angle = csdl.sin(angle_BAC_rad)

        C_y, C_z = 85.291, 4.704
        # Since C_y and C_z are known, let's calculate direction ratios of AC assuming it's in the plane
        C_y_diff = C_y - le_root_new_point_ambigu[1]
        C_z_diff = C_z - le_root_new_point_ambigu[2]

        # Calculate the remaining direction ratio for AC (C_x - A[0])
        C_x_diff = (C_y_diff * m_AC + C_z_diff * n_AC) / (l_AC - cos_angle * (m_AC ** 2 + n_AC ** 2) / sin_angle)
        print(C_x_diff)
        # Calculate the x-coordinate of C
        C_x = le_root_new_point_ambigu[0] - C_x_diff 
        print(C_x)
        C_x_new = int(68.035 - C_x)
        print(C_x_new)
        wing_sweep = self.create_output(name='wing_sweep', shape=(num_nodes * 3,), val = 0)
        wing_sweep[:,0] = C_x_new
        wing_sweep[:,1] = 0.
        wing_sweep[:,0] = C_x_new

        return 
