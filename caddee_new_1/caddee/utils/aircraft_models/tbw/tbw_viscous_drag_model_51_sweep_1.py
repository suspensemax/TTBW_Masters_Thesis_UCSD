
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl
import numpy as np
import lsdo_geo as lg
from caddee.core.caddee_core.system_model.design_scenario.design_condition.design_condition import AcStates, AtmosphericProperties
import m3l
from typing import Union, Tuple, List
from dataclasses import dataclass
from VAST.core.vast_solver import VLMOutputs
# from examples.advanced_examples.ex_tbw_geometry_trial_2 import geometryOutputs
from caddee.utils.aircraft_models.tbw.tbw_area_ffd import tbwArea, geometryOutputs


@dataclass
class Tbwviscousdrag:
    forces:m3l.Variable
    moments:m3l.Variable
    D:m3l.Variable = None
    cf_wave_1:m3l.Variable= None

class Tbw_Viscous_Drag_Model(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        # parameters
        self.parameters.declare('name', types=str)
        self.parameters.declare('geometry_units', default='m', values=['m', 'ft'])
        self.parameters.declare('counter', types = str)
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.geometry_units = self.parameters['geometry_units']
        self.counter = self.parameters['counter']

    def compute(self) -> csdl.Model:
        return Tbw_Viscous_Drag_Model_CSDL(
            module=self,
            geometry_units=self.parameters['geometry_units'],
        )

    def evaluate(self, atmos: AtmosphericProperties, ac_states : AcStates, 
                 tbw_area_outputs_plus_1_point_0g: geometryOutputs, chord: Union[m3l.Variable, None], vlm_outputs: VLMOutputs, 
                 h_tail_area:Union[m3l.Variable, None],strut_area:Union[m3l.Variable, None], sweep_angle:Union[m3l.Variable, None], 
                 mach_number:Union[m3l.Variable, None],)->Tbwviscousdrag:
    # def evaluate(self, atmos: AtmosphericProperties, ac_states : AcStates, 
    #              tbw_area_outputs_plus_1_point_0g: geometryOutputs, chord: Union[m3l.Variable, None], vlm_outputs: VLMOutputs, 
    #              mach_number: Union[m3l.Variable, None])->Tbwviscousdrag:
    # sweep_angle:Union[m3l.Variable, None],
        self.name = f"{self.counter}_tbw_viscous_drag_model"
        self.arguments = {}
        self.arguments['density'] = atmos.density
        # self.arguments['dynamic_viscosity'] = atmos.dynamic_viscosity
        # self.arguments['a'] = atmos.speed_of_sound
        self.arguments['u'] = ac_states.u
        self.arguments['v'] = ac_states.v
        self.arguments['w'] = ac_states.w
        # self.arguments['theta'] = ac_states.theta
        # self.arguments['psi'] = ac_states.psi       
        self.arguments['chord'] = chord
        # self.arguments['area'] = area
        self.arguments['area'] = tbw_area_outputs_plus_1_point_0g.wing_area
        self.arguments['induced_drag_vlm'] = vlm_outputs.cd_induced_drag
        self.arguments['cl_vlm'] = vlm_outputs.total_CL
        self.arguments['mach_number'] = mach_number
        self.arguments['h_tail_area'] = h_tail_area
        self.arguments['strut_area'] = strut_area
        self.arguments['sweep_angle'] = sweep_angle
        # self.arguments['angle_beta'] = angle_beta

        forces = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(self.num_nodes, 3), operation=self)
        D = m3l.Variable(name='D', shape=(self.num_nodes, 3), operation=self)
        cf_wave_1 = m3l.Variable(name='cf_wave_1', shape=(self.num_nodes,), operation=self)
        
        outputs = Tbwviscousdrag(
            forces=forces,
            moments=moments,
            D = D,
            cf_wave_1 = cf_wave_1,
        )
        return outputs


class Tbw_Viscous_Drag_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='viscous_drag')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('geometry_units', default='m')
        self.parameters.declare(name='reference_area', default=137.3107)  # 1478 ft^2 = 137.3107 m^2
        self.parameters.declare(name='htail_reference_area', default=27.52)  # 296.23538 ft^2 = 27.52 m^2
        # self.parameters.declare(name='strut_reference_area', default=20.49)  # 220.58 ft^2 = 20.49 m^2 23.76
        self.parameters.declare(name='strut_reference_area', default=23.76)  # 220.58 ft^2 = 20.49 m^2 
        # self.parameters.declare(name='wing_viscous_cf', default=0.00185)
        self.parameters.declare(name='wing_viscous_cf', default=0.0135) # 135 drag counts
        self.parameters.declare(name='wing_interference_cf', default=0.0015) # 15 drag counts
        self.parameters.declare(name='wing_wave_cf', default=0.0005) # 5 drag counts 
        self.parameters.declare(name='wing_induced_cf', default=0.0075) # 75 drag counts
        # self.parameters.declare(name='wing_induced_cf', default=-0.0024453) # 15 drag counts
        # self.parameters.declare(name='wing_viscous_cf', default=0.05)
        # self.parameters.declare(name='wing_viscous_cf', default=0.0075)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        geometry_units = self.parameters['geometry_units']
        reference_area_m2 = self.parameters['reference_area']
        htail_reference_area_m2 = self.parameters['htail_reference_area']
        strut_reference_area_m2 = self.parameters['strut_reference_area']

        # rho = 1.225  # kg/m^3
        # kinematic_viscosity = 1.4207E-5  # m^2/s

        ft2_2_m2 = 0.092903
        ft2m = 0.3048

        area = self.declare_variable('area',
                                          shape=(num_nodes, 1))
        chord = self.declare_variable('chord',
                                          shape=(num_nodes, 1))
        induced_drag_vlm = self.declare_variable('induced_drag_vlm',
                                          shape=(num_nodes, 1))
        # angle_beta = self.declare_variable('angle_beta',
        #                                   shape=(num_nodes, 1))
        sweep_angle = self.declare_variable('sweep_angle',
                                          shape=(1,))
        cl_vlm = self.declare_variable('cl_vlm',
                                          shape=(num_nodes, 1))
        mach_number = self.declare_variable('mach_number',
                                          shape=(num_nodes, 1))
        h_tail_area = self.declare_variable('h_tail_area',
                                          shape=(num_nodes, 1))
        strut_area = self.declare_variable('strut_area',
                                          shape=(num_nodes, 1))
        k_A = self.declare_variable(name = 'k_A', shape = (num_nodes,1), 
                                                     val = 0.95)
        t_by_c = self.declare_variable(name = 't_by_c', shape = (num_nodes,1), 
                                                     val = 0.154)
        constant = self.declare_variable(name = 'constant', shape = (num_nodes,1), 
                                                     val = 20)
        constant_1 = self.declare_variable(name = 'constant_1', shape = (num_nodes,1), 
                                                     val = 10)
        constant_2 = self.declare_variable(name = 'constant_2', shape = (num_nodes,1), 
                                                     val = (0.1/80)**(1/3))
        sweep_angle_expand = csdl.expand(sweep_angle, shape = (num_nodes,1))
        sweep_angle_cos = csdl.cos(sweep_angle_expand)
        sweep_angle = sweep_angle_cos
        cd_wave = constant * (mach_number - (k_A/sweep_angle) + (t_by_c/(sweep_angle**2)) + (cl_vlm/(constant_1*(sweep_angle**3))) + constant_2)**4
        # constant_3 = self.declare_variable(name = 'constant_3', shape = (num_nodes,1), val = (((8.825+3.22)/2)*ft2m)**2)
        # t_by_c = self.declare_variable(name = 't_by_c', shape = (num_nodes,1), val = 0.154)

        rho = self.declare_variable('density', shape=(num_nodes, 1))
        # rho = self.declare_variable('density', shape=(num_nodes, 1), val=1.11)
        mu = self.declare_variable('dynamic_viscosity', shape=(num_nodes, 1))
        a = self.declare_variable('a', shape=(num_nodes, 1))

        kinematic_viscosity = mu/rho

        u = self.declare_variable('u', shape=(num_nodes, 1))
        v = self.declare_variable('v', shape=(num_nodes, 1))
        w = self.declare_variable('w', shape=(num_nodes, 1))

        if geometry_units == 'ft':
            area_m2 = area * ft2_2_m2
            chord_m = chord * ft2m
            htail_area_m2 = h_tail_area * ft2_2_m2
            strut_area_m2 = strut_area * ft2_2_m2

        elif geometry_units == 'm':
            area_m2 = area * 1.
            chord_m = chord * 1.
            htail_area_m2 = h_tail_area * 1
            strut_area_m2 = strut_area * 1

        else:
            raise IOError
        
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

        V_inf = (u**2 + v**2 + w**2)**0.5

        VTAS = V_inf + (p+q+r+theta+psi+phi+gamma+x+y+z) * 0
        qBar = 0.5 * rho * VTAS**2 #dynamic pressure 

        Re = VTAS*chord_m/kinematic_viscosity

        # area_fraction = area_m2 / reference_area_m2
        # area_fraction = (area_m2 + strut_area_m2) / (reference_area_m2 + strut_reference_area_m2)
        area_fraction = (area_m2 + htail_area_m2 + strut_area_m2) / (reference_area_m2 + htail_reference_area_m2 + strut_reference_area_m2)
        self.register_output('area_fraction', var=area_fraction)
        self.register_output('area_m2', var=area_m2)
        # self.print_var(var=area_fraction)s

        # Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + sweep_angle)*area_fraction #drag coefficients from tbw report M70-GE
        # Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + cd_wave)*area_fraction #drag coefficients from tbw report M70-GE
        Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] )*area_fraction + cd_wave #drag coefficients from tbw report M70-GE
        # Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'])*area_fraction + cd_induced + cd_wave #drag coefficients from tbw report M70-GE

        Cf_wo_area_fraction = (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + cd_wave + (induced_drag_vlm - induced_drag_vlm))
        Cf_viscous = self.parameters['wing_viscous_cf'] * area_fraction
        Cf_interference = self.parameters['wing_interference_cf'] * area_fraction
        # Cf_wave = self.parameters['wing_wave_cf'] * area_fraction
        # Cf_wave = sweep_angle * area_fraction
        # Cf_wave = cd_wave * area_fraction
        # Cf_wave = cd_wave
        # Cf_induced = (self.parameters['wing_induced_cf'] - induced_drag_vlm) * area_fraction
        Cf_induced = (induced_drag_vlm) * area_fraction
        # self.print_var(cd_wave)
        self.register_output(name='Cf', var=Cf)
        self.register_output(name='Cf_viscous', var=Cf_viscous)
        self.register_output(name='Cf_interference', var=Cf_interference)
        self.register_output(name='Cf_wave', var=cd_wave)
        self.register_output(name='Cf_induced', var=Cf_induced)
        self.register_output(name='Cf_wo_area_fraction', var=Cf_wo_area_fraction)

        D = qBar * Cf * area_m2 #drag force
        # D = qBar * Cf * (area_m2 + strut_area_m2) #drag force
        # D = qBar * Cf * (area_m2 + htail_area_m2 + strut_area_m2) #drag force
        # a = qBar * area_m2
        a = qBar * (area_m2 + htail_area_m2 + strut_area_m2)
        
        # self.register_output(name='D', var=D)
        self.register_output(name='qBar', var=qBar)
        self.register_output(name='qBar_area', var=a)

        F = self.create_output(name='F', shape=(num_nodes, 3), val=0)
        # F[:, 0] = (D * -1.) + 140000
        F[:, 0] = D * -1. 
        # F[:, 0] = D * -1. * csdl.cos(theta[i, 0]) * csdl.cos(psi[i, 0]) 
        D_1 = self.create_output(name='D', shape=(num_nodes, 1), val=0)
        D_1[:,0] = D
        M = self.create_output(name='M', shape=(num_nodes, 3), val=0)
        M[:, 0] = D * 0
        cf_wave_1 = self.create_output(name='cf_wave_1', shape=(num_nodes, 1), val=0)
        cf_wave_1[:,0] = cd_wave * 1
        return
