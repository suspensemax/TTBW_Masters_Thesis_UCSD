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
    wing_AR : m3l.Variable
    strut_area : m3l.Variable 
    wing_area : m3l.Variable
    # wing_span : m3l.Variable

class tbwArea(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('counter', types = str)
        # self.parameters.declare('component', default=None, types=None)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        wing_area_value = tbwAreaModelCSDL(
            tbwAreaModel=self,
        )
        return wing_area_value

    def evaluate(self, wing_span_dv: Union[m3l.Variable, None],
                 wing_root_chord_dv: Union[m3l.Variable, None], wing_mid_chord_left_dv: Union[m3l.Variable, None],
                  wing_tip_chord_left_dv: Union[m3l.Variable, None], area_wing:Union[m3l.Variable, None], 
                  AR_wing:Union[m3l.Variable, None], strut_area:Union[m3l.Variable, None], )-> geometryOutputs:
        
        self.name = f"{self.counter}_tbw_area_model"
        # self.parameters.declare('name', types=str)
        # self.name = self.parameters['name']
        self.arguments = {}
        self.arguments['wing_span_dv'] = wing_span_dv
        self.arguments['wing_mid_chord_left_dv'] = wing_mid_chord_left_dv
        self.arguments['wing_root_chord_dv'] = wing_root_chord_dv
        self.arguments['wing_tip_chord_left_dv'] = wing_tip_chord_left_dv
        self.arguments['area_wing'] = area_wing
        self.arguments['strut_area'] = strut_area
        self.arguments['AR_wing'] = AR_wing
        

        wing_area = m3l.Variable(name='wing_area_value',shape= (1,1),operation = self)
        wing_AR = m3l.Variable(name='wing_AR',shape= (1,1),operation=self)
        strut_area_value = m3l.Variable(name='strut_area_value',shape= (1,),operation=self)

        outputs = geometryOutputs(
            wing_area = wing_area,
            strut_area = strut_area_value,
            wing_AR = wing_AR,
        )
        return outputs


class tbwAreaModelCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='area')
        self.parameters.declare('tbwAreaModel', types=tbwArea)
        self.parameters.declare('num_nodes', default=1)
        # return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']

        wing_span_dv = self.declare_variable('wing_span_dv',
                                          shape=(num_nodes, 1))
        wing_mid_chord_left_dv = self.declare_variable('wing_mid_chord_left_dv',
                                          shape=(num_nodes, 1))
        wing_root_chord_dv = self.declare_variable('wing_root_chord_dv',
                                          shape=(num_nodes, 1))
        wing_tip_chord_left_dv = self.declare_variable('wing_tip_chord_left_dv',
                                          shape=(num_nodes, 1))
        area_wing = self.declare_variable('area_wing',
                                          shape=(num_nodes, 1))
        AR_wing = self.declare_variable('AR_wing',
                                          shape=(num_nodes, 1))
        strut_area = self.declare_variable('strut_area',
                                          shape=(num_nodes, 1))


        wing_area = self.create_output(name='wing_area_value', shape=(num_nodes,1), val = 0)
        wing_area_value_1 = (wing_span_dv/6) * (wing_root_chord_dv + (4 * wing_mid_chord_left_dv) + wing_tip_chord_left_dv) # simpsons rule 
        wing_area_value = area_wing + wing_area_value_1 - wing_area_value_1
        # wing_area_value = area_wing + wing_area_value_1 - area_wing
        wing_area[:,0] = wing_area_value

        wing_AR_normal_value = self.create_output(name='wing_AR', shape=(num_nodes,1), val = 0)
        # wing_AR_normal_value[:,0] = ((wing_span_dv)**2 )/ wing_area
        wing_AR_normal_value[:,0] = AR_wing + (((wing_span_dv)**2 )/ wing_area) - (((wing_span_dv)**2 )/ wing_area)
        # wing_AR_normal_value[:,0] = (((wing_span_dv)**2 )/ wing_area) + AR_wing - AR_wing

        strut_area_a = self.create_output(name='strut_area_value', shape=(num_nodes,1), val = 0)
        strut_area_1 = strut_area + AR_wing - AR_wing
        strut_area_a[:,0] = strut_area_1
        return
