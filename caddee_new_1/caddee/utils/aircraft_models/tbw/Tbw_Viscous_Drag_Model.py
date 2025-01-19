
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
import math 

@dataclass
class Tbwviscousdrag:
    forces:m3l.Variable
    moments:m3l.Variable
    D:m3l.Variable = None

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

    # def evaluate(self, atmos: AtmosphericProperties, ac_states : AcStates, 
    #              tbw_area_outputs_plus_1_point_0g: geometryOutputs, chord: Union[m3l.Variable, None], vlm_outputs: VLMOutputs, 
    #              )->Tbwviscousdrag:
    def evaluate(self, atmos: AtmosphericProperties, ac_states : AcStates, 
                 tbw_area_outputs_plus_1_point_0g: geometryOutputs, chord: Union[m3l.Variable, None], vlm_outputs: VLMOutputs, 
                 mach_number: Union[m3l.Variable, None], sweep_angle:Union[m3l.Variable, None])->Tbwviscousdrag:
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
        self.arguments['mach_number'] = mach_number
        self.arguments['sweep_angle'] = sweep_angle

        forces = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(self.num_nodes, 3), operation=self)
        D = m3l.Variable(name='D', shape=(self.num_nodes, 3), operation=self)
        
        outputs = Tbwviscousdrag(
            forces=forces,
            moments=moments,
            D = D,
        )
        return outputs


class Tbw_Viscous_Drag_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='viscous_drag')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('geometry_units', default='m')
        self.parameters.declare(name='reference_area', default=137.3107)  # 1478 ft^2 = 137.3107 m^2
        # self.parameters.declare(name='wing_viscous_cf', default=0.00185)
        self.parameters.declare(name='wing_viscous_cf', default=0.0135) # 75 drag counts
        self.parameters.declare(name='wing_interference_cf', default=0.0015) # 135 drag counts
        self.parameters.declare(name='wing_wave_cf', default=0.0005) # 5 drag counts 
        self.parameters.declare(name='wing_induced_cf', default=0.0075) # 15 drag counts
        # self.parameters.declare(name='wing_induced_cf', default=-0.0024453) # 15 drag counts
        # self.parameters.declare(name='wing_viscous_cf', default=0.05)
        # self.parameters.declare(name='wing_viscous_cf', default=0.0075)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        geometry_units = self.parameters['geometry_units']
        reference_area_m2 = self.parameters['reference_area']

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
        mach_number = self.declare_variable('mach_number',
                                          shape=(num_nodes, 1))
        critical_mach_number = self.declare_variable(name = 'critical_mach_number', shape = (num_nodes,1), 
                                                     val = 0.72)
        sweep_angle = self.declare_variable('sweep_angle',
                                          shape=(num_nodes, 1))
        
        # constants = self.declare_variable(name = 'constants', shape = (num_nodes,1), val = 0.000502)
        # constants = self.declare_variable(name = 'constants', shape = (num_nodes,1), val = 0.000368)
        constants = self.declare_variable(name = 'constants', shape = (num_nodes,1), val = 0)
        
        # Cd_wave_after_sweep = constants * ((mach_number*csdl.cos(sweep_angle) * critical_mach_number)/(1 - critical_mach_number)) * ((mach_number*csdl.cos(sweep_angle) * critical_mach_number)/(1 - critical_mach_number))
        # Cd_who_knows = constants * ((mach_number*csdl.cos(sweep_angle) * critical_mach_number)/(1 - critical_mach_number)) * ((mach_number*csdl.cos(sweep_angle) * critical_mach_number)/(1 - critical_mach_number))
        # Cd_wave_after_sweep = sweep_angle + Cd_who_knows
        # Cd_wave_after_sweep = (1/csdl.cos(sweep_angle))^2 *(mach_number^2/sqrt(mach_number^2-1)) # formula darshan said - whitcomb area rule or whitcomb transonic drag rise
        # sweep = csdl.cos(sweep_angle)
        # self.print_var(sweep)
        # # wing_wave_cf = self.parameters['wing_wave_cf']
        # first_cd = (0.02699476221/sweep)**2
        # self.print_var(first_cd)
        # # print(first_cd.val)
        # mach_number_square = mach_number**2
        # self.print_var(mach_number_square)
        # mach_number_sqrt = (1-mach_number_square)**(1/2)
        # self.print_var(mach_number_sqrt)
        # # exit()
        # # mach_number_sqrt = csdl.exp(mach_number_square-1)
        # Cd_wave_after_sweep = first_cd *(mach_number_square/mach_number_sqrt) 
        Cd_wave_after_sweep = sweep_angle + mach_number - mach_number
        # self.add_constraint(Cd_wave_after_sweep, scaler = 1)
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
        elif geometry_units == 'm':
            area_m2 = area * 1.
            chord_m = chord * 1.
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

        area_fraction = area_m2 / reference_area_m2
        self.register_output('area_fraction', var=area_fraction)
        self.register_output('area_m2', var=area_m2)
        # self.print_var(var=area_fraction)

        # Cf = ( 0.664 / (Re)**0.5 ) * 0 + self.parameters['wing_viscous_cf']*area_fraction #drag coefficient - Blasius Equation - laminar flow over a flat plate. 
        # Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + self.parameters['wing_wave_cf'] + (induced_drag_vlm)
        #                                   )*area_fraction #drag coefficients from tbw report M70-GE
        # Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + self.parameters['wing_wave_cf'] + (induced_drag_vlm - self.parameters['wing_induced_cf'])
        #                                   )*area_fraction #drag coefficients from tbw report M70-GE
        # Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + self.parameters['wing_wave_cf'])*area_fraction #drag coefficients from tbw report M70-GE
        Cf = ( 0.664 / (Re)**0.5 ) * 0 + (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + Cd_wave_after_sweep)*area_fraction #drag coefficients from tbw report M70-GE
        # Cf_wo_area_fraction = (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + self.parameters['wing_wave_cf'] + (self.parameters['wing_induced_cf'] - induced_drag_vlm))
        # Cf_wo_area_fraction = (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + self.parameters['wing_wave_cf'] + (induced_drag_vlm - induced_drag_vlm))
        Cf_wo_area_fraction = (self.parameters['wing_viscous_cf'] + self.parameters['wing_interference_cf'] + Cd_wave_after_sweep + (induced_drag_vlm - induced_drag_vlm))
        Cf_viscous = self.parameters['wing_viscous_cf'] * area_fraction
        Cf_interference = self.parameters['wing_interference_cf'] * area_fraction
        Cf_wave = Cd_wave_after_sweep * area_fraction
        # Cf_wave = self.parameters['wing_wave_cf'] * area_fraction
        # Cf_induced = (self.parameters['wing_induced_cf'] - induced_drag_vlm) * area_fraction

        self.register_output(name='Cf', var=Cf)
        self.register_output(name='Cf_viscous', var=Cf_viscous)
        self.register_output(name='Cf_interference', var=Cf_interference)
        self.register_output(name='Cf_wave', var=Cf_wave)
        self.register_output(name='Cf_wo_area_fraction', var=Cf_wo_area_fraction)

        D = qBar * Cf * area_m2 #drag force
        a = qBar * area_m2
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
        return

# # Module imports
# import numpy as np
# import caddee.api as cd
# import m3l
# from python_csdl_backend import Simulator

# if __name__ == '__main__':
#     system_model = m3l.Model()
#     sizing_1_point_0g_condition = cd.CruiseCondition(
#     name='plus_1_point_0g_sizing',
#     stability_flag=False,
#     )

#     h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106.4)
#     M_1_point_0g = system_model.create_input('mach_1_point_0g', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
#     r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000)
#     theta_1_point_0g = system_model.create_input('pitch_angle_1_point_0g', val=np.deg2rad(6), dv_flag=True, lower=np.deg2rad(-10), upper=np.deg2rad(10), scaler=1e1)

#     # ac sates + atmos
#     ac_states_1_point_0g, atmos_1_point_0g = sizing_1_point_0g_condition.evaluate(mach_number=M_1_point_0g, pitch_angle=theta_1_point_0g, cruise_range=r_1_point_0g, altitude=h_1_point_0g)
#     system_model.register_output(ac_states_1_point_0g)
#     system_model.register_output(atmos_1_point_0g)

#     wing_ref_area = 1477.109999845  # 1477.109999845 ft^2 = 137.2280094 m^2   
#     wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 m
#     tbw_viscous_drag_model = Tbw_Viscous_Drag_Model(
#             name = 'tbw_viscous_drag',
#             geometry_units='ft',
#     )
#     area_value = system_model.create_input('area', val=wing_ref_area)
#     chord_value = system_model.create_input('chord', val=wing_ref_chord)
#     tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(atmos=atmos_1_point_0g, ac_states=ac_states_1_point_0g, area = area_value, chord = chord_value)
#     system_model.register_output(tbw_viscous_drag_outputs)
#     caddee_csdl_model = system_model.assemble_csdl()
#     sim = Simulator(caddee_csdl_model,analytics=True)
#     sim.run()
#     cd.print_caddee_outputs(system_model, sim, compact_print=True)
#     print('Area', sim['tbw_viscous_drag_model.area'])
