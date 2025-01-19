import numpy as np
from typing import Union, Tuple, List
from dataclasses import dataclass

import m3l 
import csdl
from csdl import Model
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL

from caddee.utils.helper_classes import MassProperties
from caddee.core.caddee_core.system_model.design_scenario.design_condition.design_condition import AcStates, AtmosphericProperties

import lsdo_geo as lg
from VAST.core.vast_solver import VLMOutputs
# from pycycle_caddee import tbwpycycle
# from Raj.caddee.tbw_area_ffd import geometryOutputs
from caddee.utils.aircraft_models.tbw.tbw_area_ffd import tbwArea, geometryOutputs
from caddee.utils.aircraft_models.tbw.tbw_propulsion import TbwpropulsionProperties


class TBW_Mass_Properties(m3l.ExplicitOperation):
    def initialize(self, kwargs): 
        self.parameters.declare('name', types=str)
        # self.parameters.declare('exclude_wing', types=bool, default=False)
        # self.parameters.declare('full_wing', types=bool, default=False)
        self.parameters.declare('geometry_units', default='m', values=['m', 'ft'])
        self.num_nodes = 1
        
    def assign_attributes(self):
        self.name = self.parameters['name']
        self.geometry_units = self.parameters['geometry_units']

    def compute(self) -> csdl.Model:
        return TBW_Mass_Properties_CSDL(
            module=self,
            geometry_units=self.parameters['geometry_units'],
            )
    
    def evaluate(self, area: Union[m3l.Variable, None], 
                 fuselage_mass: Union[m3l.Variable, None], tail_mass: Union[m3l.Variable, None], 
                 payload_mass: Union[m3l.Variable, None], 
                 tbwarea: geometryOutputs,  
                 wing_span_dv: Union[m3l.Variable, None], 
                 ):

        self.arguments = {}

        self.arguments['area'] = area
        self.arguments['fuselage_mass'] = fuselage_mass
        self.arguments['tail_mass'] = tail_mass
        self.arguments['payload_mass'] = payload_mass
        self.arguments['span'] = wing_span_dv
        self.arguments['area'] = tbwarea.wing_area

        mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable(name='cg_vector', shape=(3,), operation=self)
        inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3, 3), operation=self)

        outputs = MassProperties(
            mass=mass,
            cg_vector=cg_vector,
            inertia_tensor=inertia_tensor,
        )
        return outputs

class TBW_Mass_Properties_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('name', default='TBW_minus_wing_sizing', types=str)
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare(name='reference_area', default=137.3107)  # 1478 ft^2 = 137.3107 m^2
        self.parameters.declare('geometry_units', default='ft')

    def define(self):
        shape = (1,)
        num_nodes = self.parameters['num_nodes']
        geometry_units = self.parameters['geometry_units']
        reference_area_m2 = self.parameters['reference_area']

        area = self.declare_variable('area', shape=(num_nodes, ))

        # Unit conversions
        ft2_2_m2 = 1 / 10.764
        ft2m = 0.3048
        ktas_to_m_s = 1 / 1.944
        lbs_sq_ft_to_kg_sq_m = 0.0421401101
        # lbs_sq_ft_to_kg_sq_m = 1.

        if geometry_units == 'ft':
            area_m2 = area * ft2_2_m2
        elif geometry_units == 'm':
            area_m2 = area * 1.
        else:
            raise IOError
        
        area_fraction = area_m2 / reference_area_m2
        self.register_output('area_fraction_weights', var=area_fraction)
        # self.print_var(var=area_fraction)

        self.register_output('area_m2_weights', var=area_m2)
 
        lbs_to_kg =  1 / 2.205

        # wing_mass = self.declare_variable('wing_mass', shape=shape, val=16740.*lbs_to_kg)
        wing_strut_mass = self.declare_variable('wing_strut_mass', shape=shape, val=3680.*lbs_to_kg)
        fueltank_mass = self.declare_variable('fueltank_mass', shape=shape, val=3000.*lbs_to_kg)
        # random variables
        fuselage_mass = self.declare_variable('fuselage_mass', shape=shape, val=120000.*lbs_to_kg)
        tail_mass = self.declare_variable('tail_mass', shape=shape, val=3000.*lbs_to_kg)
        payload_mass = self.declare_variable('payload_mass', shape=shape, val=3580.*lbs_to_kg)
        total_mass_a = self.declare_variable('total_mass', shape=shape, val=150000.*lbs_to_kg)
        dummy = self.declare_variable('dummy_variable', shape=shape, val=0.)

        # # old mass -----------------------
        # total_mass = (wing_mass + wing_strut_mass + fueltank_mass + 
        #               fuselage_mass + tail_mass + payload_mass) * area_fraction 
        # total_mass = total_mass_a * area_fraction + (wing_mass + wing_strut_mass + 
        #                                              fueltank_mass + fuselage_mass + tail_mass + payload_mass) * 0
        # smallp = 0.75
        # total_mass = (wing_mass + wing_strut_mass + tail_mass + fueltank_mass) * smallp * area_fraction + (fuselage_mass + payload_mass) * smallp
        # total_mass = total_mass_a * 1. + area_fraction * 0. + (wing_mass + wing_strut_mass + fueltank_mass + 
        #                                                        fuselage_mass + tail_mass + payload_mass) * 0

        # new weight estimaton equation -----------------------
        span = self.declare_variable('span', shape=shape, val=1.)
        S = self.declare_variable('area', shape=shape, val=1.)

        A = span**2. / S

        Wdg = 150000. # flight design gross weight
        n = 5. # limit load factor
        Nz =  1.5 * n # ultimate load factor
        Sw = 1270. # trapezoidal wing area
        t_c_root = 0.106 # thickness-to-chord ratio (not constant, use average of portion of wing inboard of C-bar)
        f_lambda = 0. # taper ratio (wing or tail)
        LAMBDA = 0. # wing sweep at 25% MAC
        Scsw = 10. # control surface area (wing-mounted, includes flaps)
        wing_mass_Eng = 0.0051 * (Wdg * Nz)**0.557 * Sw**0.649 * A**0.5 * t_c_root**(-0.4) * (1+f_lambda)**0.1 * (np.cos(LAMBDA))**(-0.1) * Scsw**0.1

        other_mass = 133260.
        wing_mass_gross = 16740.
        wing_fraction = wing_mass_Eng / wing_mass_gross

        wing_mass = wing_mass_Eng * lbs_to_kg

        total_mass_Eng = wing_mass_Eng + other_mass * wing_fraction + (wing_strut_mass + tail_mass + fueltank_mass + fuselage_mass + payload_mass) * 0.
        total_mass = total_mass_Eng * lbs_to_kg
        self.register_output(name = 'mass', var = total_mass)

        # region inertia_tensor & cg_vector
        # Ixx = 412459.845 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_xx_a = self.declare_variable('moment_of_inertia_xx_a', shape=shape, val=1488519.55240438)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xx_without_wing = moment_of_inertia_xx_a + dummy 
        self.register_output(name = 'Ixx', var = moment_of_inertia_xx_without_wing)
        
        # Iyy = 13379933.707 * lbs_sq_ft_to_kg_sq_m
        moment_of_inertia_yy_a = self.declare_variable('moment_of_inertia_yy', shape=shape, val=3104778.44820183)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yy_without_wing = moment_of_inertia_yy_a + dummy
        self.register_output(name = 'Iyy', var = moment_of_inertia_yy_without_wing)

        moment_of_inertia_zz_a = self.declare_variable('moment_of_inertia_zz', shape=shape, val=4540505.81552173)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_zz_without_wing = moment_of_inertia_zz_a + dummy
        self.register_output(name = 'Izz', var = moment_of_inertia_zz_without_wing)

        moment_of_inertia_xy_a = self.declare_variable('moment_of_inertia_xy', shape=shape, val=0)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xy_without_wing = moment_of_inertia_xy_a + dummy
        self.register_output(name = 'Ixy', var = moment_of_inertia_xy_without_wing)

        moment_of_inertia_yz_a = self.declare_variable('moment_of_inertia_yz', shape=shape, val=0)
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_yz_without_wing = moment_of_inertia_yz_a + dummy
        self.register_output(name = 'Iyz', var = moment_of_inertia_yz_without_wing)

        moment_of_inertia_xz_a = self.declare_variable('moment_of_inertia_xz', shape=shape, val=259151.12468558 )
        # lbs_sq_ft_to_kg_sq_m = self.declare_variable('lbs_sq_ft_to_kg_sq_m', shape=shape, val=0.0421401101)
        moment_of_inertia_xz_without_wing = moment_of_inertia_xz_a + dummy
        self.register_output(name = 'Ixz', var = moment_of_inertia_xz_without_wing)

        inertia_tensor = self.create_output('inertia_tensor', shape=(3, 3), val=0)
        inertia_tensor[0, 0] = csdl.reshape(moment_of_inertia_xx_without_wing, (1, 1)) 
        inertia_tensor[1, 1] = csdl.reshape(moment_of_inertia_yy_without_wing, (1, 1)) 
        inertia_tensor[2, 2] = csdl.reshape(moment_of_inertia_zz_without_wing, (1, 1)) 
        inertia_tensor[0, 2] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[2, 0] = csdl.reshape(moment_of_inertia_xz_without_wing, (1, 1)) 
        inertia_tensor[1, 2] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[2, 1] = csdl.reshape(moment_of_inertia_yz_without_wing, (1, 1)) 
        inertia_tensor[0, 1] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 
        inertia_tensor[1, 0] = csdl.reshape(moment_of_inertia_xy_without_wing, (1, 1)) 

        cg_x_a = self.declare_variable('center_of_gravity_x', shape=shape, val=58.595*ft2m)
        # cg_x = 57.505*ft2m
        cg_x = cg_x_a + dummy
        # cg_y = 0.
        cg_y_a = self.declare_variable('center_of_gravity_y', shape=shape, val=0.*ft2m)
        cg_y = cg_y_a + dummy
        # cg_z = 0.253*ft2m
        cg_z_a = self.declare_variable('center_of_gravity_z', shape=shape, val=5.715*ft2m)
        cg_z = cg_z_a + dummy

        self.register_output(
            name='cgx', 
            var=cg_x)
        self.register_output(
            name='cgy',  
            var=cg_y)
        self.register_output(
            name='cgz', 
            var=cg_z)
        
        cg_vector = self.create_output('cg_vector', shape=(3, ), val=0)
        cg_vector[0] = cg_x
        cg_vector[1] = cg_y
        cg_vector[2] = cg_z
        # endregion

@dataclass
class tbwrandomdrag_outputs:
    forces:m3l.Variable
    moments:m3l.Variable
    D:m3l.Variable
    Cf:m3l.Variable

class TBW_Random_Drag_Model(m3l.ExplicitOperation):
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
        return TBW_Random_Drag_Model_CSDL(
            module=self,
            geometry_units=self.parameters['geometry_units'],
        )

    def evaluate(self, atmos: AtmosphericProperties, ac_states : AcStates, 
                 area: Union[m3l.Variable, None], 
                 chord: Union[m3l.Variable, None], vlm_outputs: VLMOutputs,
                 wing_viscous_cf: Union[m3l.Variable, None], wing_interference_cf: Union[m3l.Variable, None], 
                 wing_wave_cf: Union[m3l.Variable, None]
                 ) -> tbwrandomdrag_outputs:
        self.name = f"{self.counter}_tbw_random_drag_model"
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
        self.arguments['area'] = area
        # self.arguments['area'] = tbw_area_outputs_plus_1_point_0g.wing_area
        self.arguments['induced_drag_vlm'] = vlm_outputs.cd_induced_drag

        self.arguments['wing_viscous_cf'] = wing_viscous_cf
        self.arguments['wing_interference_cf'] = wing_interference_cf
        self.arguments['wing_wave_cf'] = wing_wave_cf

        forces = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(self.num_nodes, 3), operation=self)
        D = m3l.Variable(name='D', shape=(self.num_nodes, 3), operation=self)
        Cf = m3l.Variable(name='Cf', shape=(self.num_nodes, 1), operation=self)
        
        outputs = tbwrandomdrag_outputs(
            forces=forces,
            moments=moments,
            D = D,
            Cf = Cf,
            )
        return outputs

class TBW_Random_Drag_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='random_drag')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('geometry_units', default='ft')
        self.parameters.declare(name='reference_area', default=137.3107)  # 1478 ft^2 = 137.3107 m^2
        self.parameters.declare(name='wing_induced_cf', default=0.0075) # 15 drag counts
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        geometry_units = self.parameters['geometry_units']
        reference_area_m2 = self.parameters['reference_area']

        ft2_2_m2 = 0.092903
        ft2m = 0.3048

        area = self.declare_variable('area',
                                          shape=(num_nodes, 1))
        chord = self.declare_variable('chord',
                                          shape=(num_nodes, 1))
        induced_drag_vlm = self.declare_variable('induced_drag_vlm',
                                          shape=(num_nodes, 1))

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

        wing_viscous_cf = self.declare_variable(name='wing_viscous_cf', shape=(num_nodes, 1), val=0.0135)
        wing_interference_cf = self.declare_variable(name='wing_interference_cf', shape=(num_nodes, 1), val=0.0015)
        wing_wave_cf = self.declare_variable(name='wing_wave_cf', shape=(num_nodes, 1), val=0.0005)

        #drag coefficients from tbw report M70-GE
        Cf = (wing_viscous_cf + wing_interference_cf + wing_wave_cf)*area_fraction  + (0.664 / (Re)**0.5)*0.
        Cf_wo_area_fraction = (wing_viscous_cf + wing_interference_cf + wing_wave_cf + (induced_drag_vlm - induced_drag_vlm))
        Cf_viscous = wing_viscous_cf * area_fraction
        Cf_interference = wing_interference_cf * area_fraction
        Cf_wave = wing_wave_cf * area_fraction
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


# @dataclass
# class tbwrange_outputs:
#     Flight_Range: m3l.Variable = None

# class TBW_Range_Model(m3l.ExplicitOperation):
#     def initialize(self, kwargs):
#         # parameters
#         self.parameters.declare('name', types=str)
#         # self.parameters.declare('counter', types = str)
#         self.num_nodes = 1

#     def assign_attributes(self):
#         self.name = self.parameters['name']

#     def compute(self) -> csdl.Model:
#         return TBW_Range_Model_CSDL(
#             module=self,
#             )

#     def evaluate(self, 
#                  tbw_mass: MassProperties, tbw_prop: tbwpycycle,
#                  vlm_outputs: VLMOutputs, tbw_random_drag: tbwrandomdrag_outputs, 
#                  V: Union[m3l.Variable, None],
#                  g: Union[m3l.Variable, None],
#                 #  W_f: Union[m3l.Variable, None],
#                  ) -> tbwrange_outputs:
#         self.arguments = {}
#         self.arguments['Total_lift_value_vlm'] = vlm_outputs.Total_lift
#         self.arguments['Total_drag_value_vlm'] = vlm_outputs.Total_drag
#         self.arguments['random_drag_forces'] = tbw_random_drag.D
#         self.arguments['TSFC'] = tbw_prop.SFC
#         self.arguments['F'] = tbw_prop.forces
#         self.arguments['W_i'] = tbw_mass.mass
#         self.arguments['V'] = V
#         self.arguments['g'] = g
#         # self.arguments['W_f'] = W_f

#         Flight_Range = m3l.Variable(name='Flight_Range', shape=(self.num_nodes,1), operation=self)
       
#         outputs = tbwrange_outputs(
#             Flight_Range=Flight_Range,
#         )
#         return outputs

# # class TBW_Range_Model_CSDL(ModuleCSDL):
# #     def initialize(self):
# #         self.parameters.declare(name='name', default='L_over_D')
# #         self.parameters.declare('num_nodes', default=1)
# #         return

# #     def define(self):
# #         name = self.parameters['name']
# #         num_nodes = self.parameters['num_nodes']

# #         self.add(TBW_Range_Model_CSDL_Residual(), name='Range_Model_Residual')

# #         # Implicit Solving ---------------------------------------------------------------------------------------------------------
# #         solve_res = self.create_implicit_operation(TBW_Range_Model_CSDL_Residual())
# #         solve_res.declare_state(state='Flight_Range', residual='R_range', val=2.e8)
# #         solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=100,iprint=False)
# #         solve_res.linear_solver = csdl.DirectSolver()

# #         Total_lift_value_vlm = self.declare_variable('Total_lift_value_vlm', shape=(num_nodes, 1))
# #         Total_drag_value_vlm = self.declare_variable('Total_drag_value_vlm', shape=(num_nodes, 1))
# #         random_drag_forces = self.declare_variable('random_drag_forces', shape=(num_nodes, 1))
# #         TSFC = self.declare_variable('TSFC', shape=(num_nodes, 1))
# #         W_i = self.declare_variable('W_i', shape=(num_nodes, 1))
# #         # W_f = self.declare_variable('W_f', shape=(num_nodes, 1))
# #         V = self.declare_variable('V', shape=(num_nodes, 1))
# #         g = self.declare_variable('g', shape=(num_nodes, 1))
# #         F = self.declare_variable('F', shape=(num_nodes, 3))

# #         Flight_Range = solve_res(Total_lift_value_vlm,Total_drag_value_vlm,random_drag_forces,
# #                                  TSFC, W_i, V, g, F)

# #         # flight_range = self.create_output(name='Flight_Range', shape=(num_nodes,1), val=0)
# #         # flight_range[:,0] = Flight_Range

# #         # linear and nonlinear solver ----------------------------------------------------------------------------------------
# #         self.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=100,iprint=False,atol=1E-6,)
# #         self.linear_solver    = csdl.DirectSolver()

# class TBW_Range_Model_CSDL_Residual(ModuleCSDL):
#     def initialize(self):
#         self.parameters.declare(name='name', default='L_over_D')
#         self.parameters.declare('num_nodes', default=1)
#         return

#     def define(self):
#         name = self.parameters['name']
#         num_nodes = self.parameters['num_nodes']

#         Total_lift_value_vlm = self.declare_variable('Total_lift_value_vlm', shape=(num_nodes, 1))
#         Total_drag_value_vlm = self.declare_variable('Total_drag_value_vlm', shape=(num_nodes, 1))
#         random_drag_forces = self.declare_variable('random_drag_forces', shape=(num_nodes, 1))

#         abc_Drag = Total_drag_value_vlm + (random_drag_forces)
#         abc_Lift_over_Drag = Total_lift_value_vlm / abc_Drag
#         self.register_output(name = 'Lift_over_Drag', var = abc_Lift_over_Drag)

#         TSFC = self.declare_variable('TSFC', shape=(num_nodes, 1))
#         W_i = self.declare_variable('W_i', shape=(num_nodes, 1))
#         # W_f = self.declare_variable('W_f', shape=(num_nodes, 1))
#         V = self.declare_variable('V', shape=(num_nodes, 1))
#         g = self.declare_variable('g', shape=(num_nodes, 1))
#         F = self.declare_variable('F', shape=(num_nodes, 3))
#         Flight_Range = self.declare_variable('Flight_Range', shape=(num_nodes, 1))

#         W_f = TSFC * F[0,0] / (60*60) * Flight_Range / (V*343.)
#         # R_range = Flight_Range - abc_Lift_over_Drag * 1./(TSFC * 0.454/(60*60*4.448)) * V*343./g * csdl.log(W_i/(W_i - W_f)) + F[0,0]*0

#         # self.register_output('R_range', R_range)

#         # Flight_Range = abc_Lift_over_Drag * 1./(TSFC * 0.454/(60*60*4.448)) * V*343./g * csdl.log(W_i/(W_i - W_f)) + F[0,0]*0
        
#         # Metric system
#         # Flight_Range = abc_Lift_over_Drag * 1./(TSFC * 0.454/(60*60*4.448)) * V*343./g * csdl.log(W_i/(W_i - W_f)) + F[0,0]*0
#         # self.register_output('Flight_Range', Flight_Range)

#         R_range = Flight_Range - abc_Lift_over_Drag * 1./(TSFC * 0.454/(60*60*4.448)) * V*343./g * csdl.log(W_i/(W_i - W_f)) + F[0,0]*0
#         self.register_output('R_range', R_range)
#         return

class TBW_Range_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='L_over_D')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']

        Total_lift_value_vlm = self.declare_variable('Total_lift_value_vlm', shape=(num_nodes, 1))
        Total_drag_value_vlm = self.declare_variable('Total_drag_value_vlm', shape=(num_nodes, 1))
        random_drag_forces = self.declare_variable('random_drag_forces', shape=(num_nodes, 1))

        abc_Drag = Total_drag_value_vlm + (random_drag_forces)
        abc_Lift_over_Drag = Total_lift_value_vlm / abc_Drag
        self.register_output(name = 'Lift_over_Drag', var = abc_Lift_over_Drag)

        TSFC = self.declare_variable('TSFC', shape=(num_nodes, 1))
        W_i = self.declare_variable('W_i', shape=(num_nodes, 1))
        # W_f = self.declare_variable('W_f', shape=(num_nodes, 1))
        V = self.declare_variable('V', shape=(num_nodes, 1))
        g = self.declare_variable('g', shape=(num_nodes, 1))
        F = self.declare_variable('F', shape=(num_nodes, 3))
        # Flight_Range = self.declare_variable('Flight_Range', shape=(num_nodes, 1))

        W_f = TSFC*0.454 * F[0,0]/4.448 * 4.
        Flight_Range = abc_Lift_over_Drag * 1./(TSFC * 0.454/(60*60*4.448)) * V*343./g * csdl.log(W_i/(W_i - W_f)) * 0.000539957
        self.register_output('Flight_Range', Flight_Range)
        return


# @dataclass
# class tbwrangedist_outputs:
#     Range_Distribution: m3l.Variable

# class TBW_Range_Distribution_Model(m3l.ExplicitOperation):
#     def initialize(self, kwargs):
#         # parameters
#         self.parameters.declare('name', types=str)
#         self.parameters.declare('geometry_units', default='m', values=['m', 'ft'])
#         self.parameters.declare('counter', types = str)
#         self.parameters.declare('num_nodes', default=100)

#     def assign_attributes(self):
#         self.name = self.parameters['name']
#         self.geometry_units = self.parameters['geometry_units']
#         self.counter = self.parameters['counter']

#     def compute(self) -> csdl.Model:
#         return TBW_Range_Distribution_Model_CSDL(
#             module=self,
#             geometry_units=self.parameters['geometry_units'],
#             num_nodes = self.parameters['num_nodes'],
#         )

#     def evaluate(self, Tbw_range: tbwrange_outputs, 
#                  ) -> tbwrangedist_outputs:
#         num_nodes = self.parameters['num_nodes']
#         self.name = f"{self.counter}_tbw_random_drag_model"

#         self.arguments = {}
#         self.arguments['Flight_Range'] = Tbw_range.Flight_Range

#         Range_Distribution = m3l.Variable(name='Range_Distribution', shape=(num_nodes, 1), operation=self)
 
#         outputs = tbwrangedist_outputs(
#             Range_Distribution=Range_Distribution,
#         )
#         return outputs

# class TBW_Range_Distribution_Model_CSDL(ModuleCSDL):
#     def initialize(self):
#         self.parameters.declare(name='name', default='viscous_drag')
#         self.parameters.declare('num_nodes', default=100)
#         self.parameters.declare('geometry_units', default='m')
#         return

#     def define(self):
#         name = self.parameters['name']
#         num_nodes = self.parameters['num_nodes']
#         geometry_units = self.parameters['geometry_units']

#         Flight_Range = self.declare_variable('Flight_Range', shape=(num_nodes, 1))






#         Range_Distribution = self.create_output(name='Range_Distribution', shape=(num_nodes, 1), val=0)
#         D_1[:,0] = D


@dataclass
class tbwconstranaly_outputs:
    W_S: m3l.Variable
    T_W: m3l.Variable
    Constraint_Stall: m3l.Variable
    Constraint_Climb: m3l.Variable
    Constraint_Maneuver: m3l.Variable
    Constr_Obj: m3l.Variable

class TBW_Constraints_Analysis_Model(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        # parameters
        self.parameters.declare('name', types=str)
        self.parameters.declare('geometry_units', default='m', values=['m', 'ft'])
        # self.parameters.declare('counter', types = str)
        self.parameters.declare('num_nodes', default=1)

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.geometry_units = self.parameters['geometry_units']

    def compute(self) -> csdl.Model:
        return TBW_Constraints_Analysis_Model_CSDL(
            module=self,
            geometry_units=self.parameters['geometry_units'],
            num_nodes = self.parameters['num_nodes'],
        )

    def evaluate(self, atmos: AtmosphericProperties, 
                #  ac_states : AcStates, 
                 tbw_mass: MassProperties, tbw_prop: TbwpropulsionProperties,
                 vlm_outputs: VLMOutputs, tbw_random_drag: tbwrandomdrag_outputs, 
                 tbwarea: geometryOutputs,  
                 wing_span_dv: Union[m3l.Variable, None], V: Union[m3l.Variable, None], 
                 SFC_tbw: Union[m3l.Variable, None],
                 ) -> tbwconstranaly_outputs:
        
        num_nodes = self.parameters['num_nodes']
        # self.name = f"{self.counter}_tbw_constraints_analysis_model"

        self.arguments = {}
        self.arguments['span'] = wing_span_dv
        self.arguments['area'] = tbwarea.wing_area
        self.arguments['rho'] = atmos.density
        self.arguments['thrust'] = tbw_prop.thrust
        # self.arguments['TSFC'] = tbw_prop.SFC
        self.arguments['TSFC'] = SFC_tbw
        self.arguments['weight'] = tbw_mass.mass
        self.arguments['vlm_lift'] = vlm_outputs.Total_lift
        self.arguments['vlm_drag'] = vlm_outputs.Total_drag
        self.arguments['ran_drag'] = tbw_random_drag.D
        self.arguments['CD0'] = tbw_random_drag.Cf
        self.arguments['V'] = V

        W_S = m3l.Variable(name='W_S', shape=(num_nodes, 1), operation=self)
        T_W = m3l.Variable(name='T_W', shape=(num_nodes, 1), operation=self)
        Constraint_Stall = m3l.Variable(name='Constraint_Stall', shape=(num_nodes, 1), operation=self)
        Constraint_Climb = m3l.Variable(name='Constraint_Climb', shape=(num_nodes, 1), operation=self)
        Constraint_Maneuver = m3l.Variable(name='Constraint_Maneuver', shape=(num_nodes, 1), operation=self)
        Constr_Obj = m3l.Variable(name='Constr_Obj', shape=(num_nodes, 1), operation=self)
 
        outputs = tbwconstranaly_outputs(
            W_S = W_S,
            T_W = T_W,
            Constraint_Stall = Constraint_Stall,
            Constraint_Climb = Constraint_Climb,
            Constraint_Maneuver = Constraint_Maneuver,
            Constr_Obj = Constr_Obj,
        )
        return outputs

class TBW_Constraints_Analysis_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='viscous_drag')
        self.parameters.declare('num_nodes', default=100)
        self.parameters.declare('geometry_units', default='m')
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        geometry_units = self.parameters['geometry_units']

        span = self.declare_variable('span', shape=(num_nodes, 1))
        S = self.declare_variable('area', shape=(num_nodes, 1))
        T = self.declare_variable('thrust', shape=(num_nodes, 1))
        W = self.declare_variable('weight', shape=(num_nodes, 1))
        rho = self.declare_variable('rho', shape=(num_nodes, 1))
        vlm_lift = self.declare_variable('vlm_lift', shape=(num_nodes, 1))
        vlm_drag = self.declare_variable('vlm_drag', shape=(num_nodes, 1))
        ran_drag = self.declare_variable('ran_drag', shape=(num_nodes, 1))
        CD0 = self.declare_variable('CD0', shape=(num_nodes, 1))
        V = self.declare_variable('V', shape=(num_nodes, 1))

        n = 2.5

        span_Me = span * 0.3048
        S_Me = S * 0.092903
        # span_Me = span * 1.
        # S_Me = S * 1.
        V_Me = V * 343.
        D = vlm_drag + ran_drag

        AR = span_Me**2. / S_Me

        e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64 # straight-wing
        # e = 0.75 # oswald efficiency

        q = 0.5 * rho * V_Me**2.
        pi_e_AR_q = 3.1415 * e * AR * q
        G = (T - D) / (W*9.8)

        W_S = W / S_Me
        T_W = T / (W*9.8)

        # use equation --------------
        lam = 0.
        Clmax = 1.7
        CLmax = 0.9 * Clmax * np.cos(lam)
        Vstall = (2 / (rho*CLmax) * W_S)**0.5
        # e = 4.61 * (1 - 0.045 * A**0.68) * (csdl.cos(wingLambda))**0.15 - 3.1 # swept-wing
        # # use constants --------------
        # CLmax = 1.5 # 1.53 # 1.8
        # Vstall = 45.6741
        # # Vstall = 81.5 * 0.51444

        C1 = CLmax * 0.5 * rho * Vstall**2.
        A2 = CD0 * q
        B2 = 1/pi_e_AR_q
        C2 = G * 1.
        A3 = A2 * 1.
        B3 = n**2 * B2
        Constraint_Stall = W_S - C1

        T_W_Climb = (A2/W_S + B2*W_S + C2)
        T_W_Maneuver = (A3/W_S + B3*W_S)

        Constraint_Climb = T_W - T_W_Climb
        Constraint_Maneuver = T_W - T_W_Maneuver
        # Constraint_Stall = W_S - (CLmax * 0.5 * rho * Vstall**2.)
        # Constraint_Climb = T_W - (G + W_S/pi_e_AR_q + CD0_q_W_S)
        # Constraint_Maneuver = T_W - (CD0_q_W_S + n**2 / pi_e_AR_q)

        self.register_output('AR', AR)
        self.register_output('S_Me', S_Me)

        self.register_output('C1', C1)
        self.register_output('A2', A2)
        self.register_output('B2', B2)
        self.register_output('C2', C2)
        self.register_output('A3', A3)
        self.register_output('B3', B3)

        self.register_output('T_W_Climb', T_W_Climb)
        self.register_output('T_W_Maneuver', T_W_Maneuver)

        self.register_output('W_S', W_S)
        self.register_output('T_W', T_W)
        self.register_output('Constraint_Stall', Constraint_Stall)
        self.register_output('Constraint_Climb', Constraint_Climb)
        self.register_output('Constraint_Maneuver', Constraint_Maneuver)

        # # old objective: T_W -----------------------------
        # # a = 0.5
        # # Constr_Obj = a * (-1.) * W_S + (1-a) * T_W
        # Constr_Obj = 1. * T_W
        # # Constr_Obj = (-1.) * W_S
        # self.register_output('Constr_Obj', Constr_Obj)

        # new objective: fuel consumption -----------------------------
        abc_Drag = vlm_drag + ran_drag
        abc_Lift_over_Drag = vlm_lift / abc_Drag
        self.register_output(name = 'Lift_over_Drag', var = abc_Lift_over_Drag)

        TSFC = self.declare_variable('TSFC', shape=(num_nodes, 1))
        V = self.declare_variable('V', shape=(num_nodes, 1))
        g = 9.8
        Range = 6482000.

        # Wf
        Constr_Obj = W * (1. - 1. / csdl.exp(Range / abc_Lift_over_Drag * (TSFC * 0.454/(60*60*4.448)) * g/(V*343)))
        # Constr_Obj = 1. * T_W + (TSFC + V) * 0.
        self.register_output('Constr_Obj', Constr_Obj)

