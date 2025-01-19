
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
class Tbw_fuel_burn_cruise:
    fuel_burn_cruise: m3l.Variable

class Tbw_fuel_burn_cruise_Model(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        # parametersMassProperties
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        return Tbw_fuel_burn_cruise_Model_CSDL(
            module=self,
        )

    # def evaluate(self, L_over_D: Tbw_L_over_D, total_mass_props: MassProperties, thrust: TbwpropulsionProperties,)->Tbw_fuel_burn_cruise:
    def evaluate(self, L_over_D: Tbw_L_over_D, wing_beam_mass_props: m3l.Variable, tbw_mass_properties:m3l.Variable ,thrust: m3l.Variable, ac_states:AcStates)->Tbw_fuel_burn_cruise:
        self.name = f"{self.counter}_tbw_fuel_burn_cruise_model"
        self.arguments = {}
        self.arguments['L_over_D'] = L_over_D.L_over_D
        self.arguments['wing_beam_mass_props'] = wing_beam_mass_props
        self.arguments['tbw_mass_properties'] = tbw_mass_properties
        self.arguments['thrust'] = thrust
        self.arguments['u'] = ac_states.u
        self.arguments['v'] = ac_states.v
        self.arguments['w'] = ac_states.w

        fuel_burn_cruise_m3l = m3l.Variable(name='fuel_burn_cruise_m3l', shape=(self.num_nodes,1), operation=self)
       
        outputs = Tbw_fuel_burn_cruise(
            fuel_burn_cruise=fuel_burn_cruise_m3l,
        )
        return outputs


class Tbw_fuel_burn_cruise_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='fuel_burn_cruise_m3l_model_csdl')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']


        L_over_D = self.declare_variable('L_over_D',
                                          shape=(num_nodes, 1))
        wing_beam_mass_props = self.declare_variable('wing_beam_mass_props',
                                          shape=(num_nodes, 1))
        tbw_mass_properties = self.declare_variable('tbw_mass_properties',
                                          shape=(num_nodes, 1))
        W_i = wing_beam_mass_props + tbw_mass_properties
        self.register_output(name = 'Weight_i', var = W_i)

        thrust = self.declare_variable('thrust', shape=(num_nodes, 1))

        u = self.declare_variable('u', shape=(num_nodes, 1))
        v = self.declare_variable('v', shape=(num_nodes, 1))
        w = self.declare_variable('w', shape=(num_nodes, 1))

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

        V_inf = (u**2 + v**2 + w**2)**(0.5)
        c = self.declare_variable('c_11g', shape=(num_nodes, 1), val = 1.3 * 10**(-4)) #TSFC
        range = self.declare_variable(name='range', shape=(num_nodes, 1), units='m', val=6000000)
        ln_weight_ratio =  (range * c)/(V_inf * L_over_D) + (p*q*r*phi*theta*psi*gamma*x*y*z*thrust*0)
        a = csdl.exp(ln_weight_ratio)
        fuel_burn = W_i-(W_i/a)

        fuel_burn_cruise_m3l = self.create_output(name='fuel_burn_cruise_m3l', shape=(num_nodes,1), val=0)
        fuel_burn_cruise_m3l[:,0] = fuel_burn

        return
