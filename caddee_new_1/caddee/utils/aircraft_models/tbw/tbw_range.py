
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
class Tbw_range:
    range: m3l.Variable

class Tbw_range_Model(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        # parametersMassProperties
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        return Tbw_range_Model_CSDL(
            module=self,
        )

    # def evaluate(self, L_over_D: Tbw_L_over_D, total_mass_props: MassProperties, thrust: TbwpropulsionProperties,)->Tbw_range:
    def evaluate(self, L_over_D: Tbw_L_over_D, total_mass_props: Union[m3l.Variable], thrust: Union[m3l.Variable],)->Tbw_range:
        self.name = f"{self.counter}_tbw_range_model"
        self.arguments = {}
        self.arguments['L_over_D'] = L_over_D.L_over_D
        self.arguments['total_mass_props'] = total_mass_props
        self.arguments['thrust'] = thrust

        # self.arguments['total_mass_props'] = total_mass_props.mass
        # self.arguments['thrust'] = thrust.thrust

        range_cruise_m3l = m3l.Variable(name='range_cruise_m3l', shape=(self.num_nodes,1), operation=self)
       
        outputs = Tbw_range(
            range=range_cruise_m3l,
        )
        return outputs


class Tbw_range_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='range_m3l_model_csdl')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']


        L_over_D = self.declare_variable('L_over_D',
                                          shape=(num_nodes, 1))
        total_mass_props = self.declare_variable('total_mass_props',
                                          shape=(num_nodes, 1))
        # thrust_a = self.declare_variable('thrust',
        #                                   shape=(num_nodes, 1))
        # constant_a_1 = self.declare_variable('constant_a_1', shape=(num_nodes, 1), val= 1/4.44822)
        # thrust = thrust_a * constant_a_1
      
        thrust = self.declare_variable('thrust', shape=(num_nodes, 1))

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

      
        # c_11g = self.declare_variable('c_11g', shape=(num_nodes, 1), val = 0.46850 * 3.5 * 10**(-5)) #TSFC
        c_11g = self.declare_variable('c_11g', shape=(num_nodes, 1), val = 0.46850 * 4.5 * 10**(-5)) #TSFC
        c_1g = self.declare_variable('c1g', shape=(num_nodes, 1), val = 0.46850 * 11 * 10**(-2)) #TSFC
        delta_t = self.declare_variable('delta_t', shape=(num_nodes, 1), val = 9.0) # in hours
        # constant_aahahaha = self.declare_variable('constant_aahahaha', shape=(num_nodes, 1), val = 1/1000)
        # constant_aahahaha_1111 = self.declare_variable('constant_aahahaha_1111', shape=(num_nodes, 1), val = 5603.801641)
        # total_fuel_burned = (thrust * constant_aahahaha * c_1g * delta_t) + p*q*r*phi*theta*psi*gamma*x*y*z*0
        total_fuel_burned = (thrust * c_1g * delta_t) + p*q*r*phi*theta*psi*gamma*x*y*z*0
        # total_fuel_burned = (thrust * c_1g * delta_t) * p*q*r*phi*theta*psi*gamma*x*y*z*0 + constant_aahahaha_1111
        self.print_var(total_fuel_burned)
        self.register_output(name = 'total_fuel_burned', var = total_fuel_burned)
        final_weight = total_mass_props - total_fuel_burned
        self.register_output(name = 'final_weight', var = final_weight)
        weight_ratio = total_mass_props / final_weight
        # self.print_var(weight_ratio)
        log_weight_ratio = csdl.log(weight_ratio)
        self.register_output(name = 'log_weight_ratio', var = log_weight_ratio)
        self.register_output(name = 'weight_ratio', var = weight_ratio)

        velocity_1g = self.declare_variable('velocity_1g', shape=(num_nodes, 1), val = 207.02897322)
        constant_aahahaha_1 = self.declare_variable('constant_aahahaha_1', shape=(num_nodes, 1), val = 9.78)
        range_a = (velocity_1g / (c_11g * constant_aahahaha_1)) * (L_over_D * log_weight_ratio)
        self.register_output(name = 'range_aaa', var = range_a)

        # range_a = (velocity_1g / c_1g) * (L_over_D * weight_ratio)

        range = self.create_output(name='range_cruise_m3l', shape=(num_nodes,1), val=0)
        range[:,0] = range_a

        return
