
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl
import numpy as np
import lsdo_geo as lg
from caddee.core.caddee_core.system_model.design_scenario.design_condition.design_condition import AcStates, AtmosphericProperties
from VAST.core.vast_solver import VLMOutputs
from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbwviscousdrag
import m3l
from typing import Union, Tuple
from dataclasses import dataclass

@dataclass
class Tbw_L_over_D:
    L_over_D: m3l.Variable

class Tbw_L_over_D_Model(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        # parameters
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        return Tbw_L_over_D_Model_CSDL(
            module=self,
        )

    def evaluate(self, vlm_outputs: VLMOutputs, tbw_viscous_drag: Tbwviscousdrag)->Tbw_L_over_D:
            # def evaluate(self, Total_lift_value_vlm: m3l.Variable, Total_drag_value_vlm: m3l.Variable, viscous_drag_forces : m3l.Variable)->Tbw_L_over_D:
        self.name = f"{self.counter}_tbw_L_over_D_model"
        self.arguments = {}
        self.arguments['Total_lift_value_vlm'] = vlm_outputs.Total_lift
        self.arguments['Total_drag_value_vlm'] = vlm_outputs.Total_drag
        self.arguments['viscous_drag_forces'] = tbw_viscous_drag.D

        L_over_D = m3l.Variable(name='L_over_D', shape=(self.num_nodes,1), operation=self)
       
        outputs = Tbw_L_over_D(
            L_over_D=L_over_D,
        )
        return outputs


class Tbw_L_over_D_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='L_over_D')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']


        Total_lift_value_vlm = self.declare_variable('Total_lift_value_vlm',
                                          shape=(num_nodes, 1))
        Total_drag_value_vlm = self.declare_variable('Total_drag_value_vlm',
                                          shape=(num_nodes, 1))
        viscous_drag_forces = self.declare_variable('viscous_drag_forces',
                                          shape=(num_nodes, 1))

        
        # Total_lift_value_vlm = self.declare_variable('total_lift',
        #                                   shape=(num_nodes, 1))
        # Total_drag_value_vlm = self.declare_variable('total_drag',
        #                                   shape=(num_nodes, 1))
        # viscous_drag_forces = self.declare_variable('viscous_drag_forces',
        #                                   shape=(num_nodes, 1))
        
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

        abc_Drag = Total_drag_value_vlm + (viscous_drag_forces) + p*q*r*phi*theta*psi*gamma*x*y*z*0
        # abc_Drag = Total_drag_value_vlm + p*q*r*phi*theta*psi*gamma*x*y*z*0
        abc_Lift_over_Drag = Total_lift_value_vlm / abc_Drag
        self.register_output(name = 'Lift_over_Drag', var = abc_Lift_over_Drag)
        L_over_D = self.create_output(name='L_over_D', shape=(num_nodes,1), val=0)
        L_over_D[:,0] =  abc_Lift_over_Drag
        
        return
