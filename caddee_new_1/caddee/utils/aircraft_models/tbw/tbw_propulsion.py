
# from caddee.core.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import \
#     MechanicsModel
import csdl
import numpy as np
import m3l
# from caddee.utils.helper_functions.tbw_propulsion_helper import TbwpropulsionProperties
from lsdo_rotor.utils.helper_classes import RotorMeshes, AcStates, BEMOutputs
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from typing import Union, Tuple
from dataclasses import dataclass
import m3l
import random


@dataclass
class TbwpropulsionProperties:
    forces : m3l.Variable
    moments: m3l.Variable
    thrust : m3l.Variable

class tbwPropulsionModel(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        # self.parameters.declare('component', default=None, types=None)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        # throttle = self.parameters['throttle']
        tbw_prop = tbwPropulsionModelCSDL(
            tbwPropulsionModel=self,
            thrust_vector=np.array([1, 0, 0]),  # Cruise and Climb
            #thrust_vector=np.array([0, 0, -1]),  # Hover
        )
        return tbw_prop

    def evaluate(self, throttle: Union[m3l.Variable, None])-> TbwpropulsionProperties:
        
        self.name = f"{self.counter}_tbw_prop_model"
        # self.parameters.declare('name', types=str)
        # self.name = self.parameters['name']
        self.arguments = {}
        self.arguments['throttle'] = throttle

        #tbw_prop_operation = m3l.CSDLOperation(name='tbw_prop_model', arguments=arguments, operation_csdl=operation_csdl)
        forces = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=self)
        # forces = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(self.num_nodes, 3), operation=self)
        thrust = m3l.Variable(name='thrust', shape=(self.num_nodes, 1), operation=self)

        outputs = TbwpropulsionProperties(
            forces=forces,
            moments=moments,
            thrust = thrust,
        )
        return outputs


class tbwPropulsionModelCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('tbwPropulsionModel', types=tbwPropulsionModel)
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('thrust_vector', default=np.array([1, 0, 0]), types=np.ndarray)
        # return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        thrust_vector = self.parameters['thrust_vector']

        ref_pt_input = np.array([0., 0., 2.8])
        thrust_origin = self.declare_variable('thrust_origin', shape=(3,), units='m', val=np.array([0., 0., 6.256]))
        ref_pt = self.declare_variable('ref_pt', shape=(3,), units='m', val=ref_pt_input)
        throttle = self.declare_variable('throttle',shape=(num_nodes,1))

        u = self.declare_variable(name='u',
                                  shape=(num_nodes, 1), units='rad', val=1)
        v = self.declare_variable(name='v',
                                  shape=(num_nodes, 1), units='rad', val=0)
        w = self.declare_variable(name='w',
                                  shape=(num_nodes, 1), units='rad', val=0)

        p = self.declare_variable(name='p',
                                  shape=(num_nodes, 1), units='rad', val=0)
        q = self.declare_variable(name='q',
                                  shape=(num_nodes, 1), units='rad', val=0)
        r = self.declare_variable(name='r',
                                  shape=(num_nodes, 1), units='rad', val=0)

        phi = self.declare_variable(name='phi',
                                    shape=(num_nodes, 1), units='rad', val=0)
        theta = self.declare_variable(name='theta',
                                      shape=(num_nodes, 1), units='rad', val=0)
        psi = self.declare_variable(name='psi',
                                    shape=(num_nodes, 1), units='rad', val=0)

        gamma = self.declare_variable(name='gamma',
                                      shape=(num_nodes, 1), units='rad', val=0)

        x = self.declare_variable(name='x',
                                  shape=(num_nodes, 1), units='rad', val=0)
        y = self.declare_variable(name='y',
                                  shape=(num_nodes, 1), units='rad', val=0)
        z = self.declare_variable(name='z',
                                  shape=(num_nodes, 1), units='rad', val=0)

        # constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=2 * 19467)
        # constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=2 * 30780.5945) # cruise - 3145 lbf, SFC - 0.455 wrong formula
        # constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=2 * 34041.158) # cruise - 3181 lbf, SFC - 0.416 wrong formula
        constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=102313.06) #23,000 lbf
        # constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=74470.65312) #16716 lbf max thrust
        # constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=74470.65312) #16716 lbf max thrust
        # constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=2 * 74470.65312) #2 * 16716 lbf max thrust
        # constant = self.declare_variable(name='constant', shape=(num_nodes, 1), val=2 * 120000.) #2 * 16716 lbf max thrust

        # T = (constant * throttle) + x*y*z*gamma*psi*theta*u*v*w*p*q*r*phi*0
        T2 = (constant * throttle) 
        T1 = x*y*z*gamma*psi*theta*u*v*w*p*q*r*phi*0
        T = T1 + T2

        self.register_output(name='T', var=T)

        F = self.create_output(name='F', shape=(num_nodes, 3), val=0)
        thrust = self.create_output(name='thrust', shape=(num_nodes, 1), val=0)
        thrust[:,0] = T
        for i in range(3):
            if thrust_vector[i] == 1 or thrust_vector[i] == -1:
                F[:, i] = T * thrust_vector[i]
            elif thrust_vector[i] == 0:
                F[:, i] = T * 0
            else:
                raise ValueError
        
        offset = ref_pt - thrust_origin #Moment direction is important.  position vector pointing from the point of interest (the rotation center) to a point along the force’s line-of-action
        M = self.create_output(name='M', shape=(num_nodes, 3))
        M[:, 0] = T * 0
        for ii in range(num_nodes):
            M[ii, 1] = F[ii, 0] * csdl.reshape(offset[2], (1, 1)) + F[ii, 2] * csdl.reshape(offset[0], (1, 1))
        M[:, 2] = T * 0
        return


# if __name__ == '__main__':
#     prop_model = tbwPropulsionModelCSDL()
#     system_model = m3l.Model()
#     import python_csdl_backend
#     throttle = system_model.create_input('throttle', val = 1., dv_flag=True, lower=0., upper=1.)
#     sim = python_csdl_backend.Simulator(prop_model)
#     sim['thrust_origin'] = np.array([61.009, 42.646, 6.256])
#     sim['ref_pt'] = np.array([0., 0., -2.8])
#     sim.run()
#     print('Thrust: ', sim['T'])
#     print('Force: ', sim['F'])
#     print('Moment:', sim['M'])

#     throttle = system_model.create_input('throttle_a', val = 0.5, dv_flag=True, lower=0., upper=1.)
#     tbw_left_prop_model = tbwPropulsionModel(
#         name='TBW_Propulsion',
#     )
#     # ref_pt = np.array([0., 0., 2.8])
#     # thrust_origin = system_model.create_input('thrust_origin', val=np.array([0., 0., 6.256]))
#     # ref_pt_input = system_model.create_input('ref_pt', val=ref_pt)

#     # tbw_left_propulsion_outputs = tbw_left_prop_model.evaluate(ac_states=ac_states_2_point_5g, thrust_origin=thrust_origin,
#     #                                                            ref_pt_input=ref_pt_input, throttle=throttle)
#     tbw_left_propulsion_outputs = tbw_left_prop_model.evaluate(throttle = throttle)
#     system_model.register_output(tbw_left_propulsion_outputs)
#     caddee_csdl_model = system_model.assemble_csdl()
#     from python_csdl_backend import Simulator
#     import caddee.api as cd
#     sim = Simulator(caddee_csdl_model,analytics=True)
#     sim.run()
#     cd.print_caddee_outputs(system_model, sim, compact_print=True)