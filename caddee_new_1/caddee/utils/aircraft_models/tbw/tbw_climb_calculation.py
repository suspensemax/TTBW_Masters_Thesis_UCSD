
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl
import numpy as np
import lsdo_geo as lg
from caddee.core.caddee_core.system_model.design_scenario.design_condition.design_condition import AcStates, AtmosphericProperties
# from VAST.core.vast_solver import VLMOutputs
# from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbwviscousdrag
from caddee.utils.aircraft_models.tbw.Tbw_L_over_D_final import Tbw_L_over_D 
from caddee.utils.helper_classes import MassProperties
from caddee.utils.aircraft_models.tbw.tbw_propulsion import TbwpropulsionProperties
import m3l
from typing import Union, Tuple
from dataclasses import dataclass
import numpy as np 

@dataclass
class Tbw_climb:
    climb_fuel: m3l.Variable
    climb_time : m3l.Variable

class Tbw_climb_Model(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        # parametersMassProperties
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        return Tbw_climb_Model_CSDL(
            module=self,
        )

    # def evaluate(self, L_over_D: Tbw_L_over_D, total_mass_props: MassProperties, thrust: TbwpropulsionProperties,)->Tbw_climb:
    def evaluate(self, thrust: m3l.Variable,ac_states:AcStates, climb_flight_path_angle: m3l.Variable, climb_hf: m3l.Variable, climb_hi:m3l.Variable,)->Tbw_climb:
        self.name = f"{self.counter}_tbw_climb_model"
        self.arguments = {}
        self.arguments['thrust'] = thrust
        self.arguments['u'] = ac_states.u
        self.arguments['v'] = ac_states.v
        self.arguments['w'] = ac_states.w
        self.arguments['climb_flight_path_angle'] = climb_flight_path_angle
        self.arguments['climb_hf'] = climb_hf
        self.arguments['climb_hi'] = climb_hi

        climb_cruise_m3l = m3l.Variable(name='climb_fuel_burn_m3l', shape=(self.num_nodes,1), operation=self)
        climb_delta_t_m3l = m3l.Variable(name='climb_delta_t_m3l', shape=(self.num_nodes,1), operation=self)
       
        outputs = Tbw_climb(
            climb_fuel=climb_cruise_m3l,
            climb_time = climb_delta_t_m3l
        )
        return outputs


class Tbw_climb_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='climb_m3l_model_csdl')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
      
        thrust = self.declare_variable('thrust', shape=(num_nodes, 1))
        climb_hi = self.declare_variable('climb_hi', shape=(num_nodes, 1))
        climb_hf = self.declare_variable('climb_hf', shape=(num_nodes, 1))
        climb_flight_path_angle = self.declare_variable('climb_flight_path_angle', shape=(num_nodes, 1))

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
                                    shape=(num_nodes, 1), units='rad', val = 0)

        gamma = self.declare_variable(name='gamma',
                                      shape=(num_nodes, 1), units='rad', val=0)

        x = self.declare_variable(name='x',
                                  shape=(num_nodes, 1), units='rad', val=0)
        y = self.declare_variable(name='y',
                                  shape=(num_nodes, 1), units='rad', val=0)
        z = self.declare_variable(name='z',
                                  shape=(num_nodes, 1), units='rad', val=0)

        u = self.declare_variable('u', shape=(num_nodes, 1))      
        v = self.declare_variable('v', shape=(num_nodes, 1))      
        w = self.declare_variable('w', shape=(num_nodes, 1))      

        V_inf = (u**2 + v**2 + w**2)**(0.5)
        altitude_gain = (climb_hf - climb_hi)
        V_gamma = V_inf * csdl.sin(climb_flight_path_angle)

        SFC = self.declare_variable(name='SFC', shape=(num_nodes,1), val=0.5*(10**(-1)))
        delta_t = (altitude_gain/V_gamma)/3600
        climb_delta_t_m3l = self.create_output(name='climb_delta_t_m3l', shape=(num_nodes,1), val=0)
        climb_delta_t_m3l[:,0] = delta_t        
        total_fuel_burned = (thrust * SFC * delta_t)+ p*q*r*phi*theta*psi*gamma*x*y*z*0
        # total_fuel_burned = (thrust * c_1g * delta_t) * p*q*r*phi*theta*psi*gamma*x*y*z*0 + constant_aahahaha_1111
        self.print_var(total_fuel_burned)
        climb_fuel_burn_m3l = self.create_output(name='climb_fuel_burn_m3l', shape=(num_nodes,1), val=0)
        climb_fuel_burn_m3l[:,0] = total_fuel_burned

        return
