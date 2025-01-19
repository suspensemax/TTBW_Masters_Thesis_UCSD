
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl
import numpy as np
import lsdo_geo as lg
from caddee.utils.aircraft_models.tbw.tbw_fuel_cruise import Tbw_fuel_burn_cruise
from caddee.utils.aircraft_models.tbw.tbw_climb_calculation import Tbw_climb
from caddee.utils.aircraft_models.tbw.tbw_descent_calculation import Tbw_descent
import m3l
from typing import Union, Tuple
from dataclasses import dataclass
import numpy as np 

@dataclass
class Tbw_combined:
    combined_fuel: m3l.Variable

class Tbw_combined_Model(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        # parametersMassProperties
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        return Tbw_combined_Model_CSDL(
            module=self,
        )

    # def evaluate(self, L_over_D: Tbw_L_over_D, total_mass_props: MassProperties, thrust: TbwpropulsionProperties,)->Tbw_combined:
    def evaluate(self, cruise_fuel: Tbw_fuel_burn_cruise, climb_fuel: Tbw_climb, descent_fuel: Tbw_descent)->Tbw_combined:
        self.name = f"{self.counter}_tbw_combined_model"
        self.arguments = {}
        self.arguments['cruise_fuel'] = cruise_fuel.fuel_burn_cruise
        self.arguments['climb_fuel'] = climb_fuel.climb_fuel
        self.arguments['descent_fuel'] = descent_fuel.descent_fuel

        combined_cruise_m3l = m3l.Variable(name='combined_fuel_burn_m3l', shape=(self.num_nodes,1), operation=self)
       
        outputs = Tbw_combined(
            combined_fuel=combined_cruise_m3l,
        )
        return outputs


class Tbw_combined_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='combined_m3l_model_csdl')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
      
        cruise_fuel = self.declare_variable('cruise_fuel', shape=(num_nodes, 1))
        climb_fuel = self.declare_variable('climb_fuel', shape=(num_nodes, 1))
        descent_fuel = self.declare_variable('descent_fuel', shape=(num_nodes, 1))

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

      
        total_fuel_burned = (cruise_fuel + climb_fuel + descent_fuel)+ p*q*r*phi*theta*psi*gamma*x*y*z*0
        self.print_var(total_fuel_burned)
        combined_fuel_burn_m3l = self.create_output(name='combined_fuel_burn_m3l', shape=(num_nodes,1), val=0)
        combined_fuel_burn_m3l[:,0] = total_fuel_burned

        return
