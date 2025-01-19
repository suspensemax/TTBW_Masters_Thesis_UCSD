from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl
import numpy as np
import lsdo_geo as lg
from caddee.core.caddee_core.system_model.design_scenario.design_condition.design_condition import AcStates, AtmosphericProperties
import m3l
from typing import List
from typing import Union, Tuple
from dataclasses import dataclass

class DragComponent:
    def __init__(self, 
                 component_type : str, 
                 wetted_area : m3l.Variable,
                 characteristic_length : m3l.Variable, 
                 characteristic_diameter : Union[m3l.Variable, float, None] = None,
                 thickness_to_chord: Union[m3l.Variable, float, None] = None,
                 multiplicity :  int = 1,
                 Q : Union[int, float] = 1.1,
                 x_cm : Union[int, float] = 0.25,
                 sweep : Union[int, float, m3l.Variable] = 0.,
                 ) -> None:
        
        self.component_type = component_type
        self.wetted_area = wetted_area
        self.characteristic_length = characteristic_length
        self.characteristic_diameter = characteristic_diameter
        self.thickness_to_chord = thickness_to_chord
        self.multiplicity = multiplicity
        self.Q = Q
        self.x_cm = x_cm
        self.sweep = sweep

        acceptable_component_types = [
            'wing', 'tail', 'fuselage', 'boom', 'strut', 'pylon', 'flat_plate', 'canopy', 'nacelle'
        ]
        if component_type not in acceptable_component_types:
            raise ValueError(f"Unknown 'component_type' argument. Acceptable component types are {acceptable_component_types}")

        if (component_type in ['wing', 'tail', 'strut', 'pylon']) and (thickness_to_chord is None):
            raise ValueError(f"Component type '{component_type}' but did not specify 'thickness_to_chord'")


@dataclass
class Tbwviscousdrag:
    forces:m3l.Variable
    moments:m3l.Variable

class Tbw_Viscous_Drag_Model(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        # parameters
        self.parameters.declare('name', types=str)
        self.parameters.declare('geometry_units', default='m', values=['m', 'ft'])
        self.num_nodes = 1

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.geometry_units = self.parameters['geometry_units']

    def compute(self) -> csdl.Model:
        return Tbw_Viscous_Drag_Model_CSDL(
            module=self,
            geometry_units=self.parameters['geometry_units'],
            drag_comp_list=self.drag_comp_list,
        )

    def evaluate(self, atmos: AtmosphericProperties, ac_states : AcStates, 
                 wing_area: Union[m3l.Variable, None], htail_area: Union[m3l.Variable, None], strut_area: Union[m3l.Variable, None], jury_area: Union[m3l.Variable, None],
                   chord: Union[m3l.Variable, None], drag_comp_list : List[DragComponent],):
        self.name = "tbw_viscous_drag_model"
        self.arguments = {}
        self.arguments['density'] = atmos.density
        # self.arguments['dynamic_viscosity'] = atmos.dynamic_viscosity
        # self.arguments['a'] = atmos.speed_of_sound
        self.arguments['u'] = ac_states.u
        self.arguments['v'] = ac_states.v
        self.arguments['w'] = ac_states.w
        self.arguments['theta'] = ac_states.theta
        self.arguments['psi'] = ac_states.psi
        # self.arguments['S_ref'] = s_ref
        # self.neglect_list = ['S_ref']     
        self.arguments['chord'] = chord
        self.arguments['wing_area'] = wing_area
        self.arguments['htail_area'] = htail_area
        self.arguments['strut_area'] = strut_area
        self.arguments['jury_area'] = jury_area

        self.drag_comp_list = drag_comp_list

        forces = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(self.num_nodes, 3), operation=self)
        
        outputs = Tbwviscousdrag(
            forces=forces,
            moments=moments,
        )
        return outputs


class Tbw_Viscous_Drag_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='viscous_drag')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('geometry_units', default='m')
        self.parameters.declare(name='reference_area_wing', default=137.3107)  # 1478 ft^2 = 137.3107 m^2
        self.parameters.declare(name='reference_area_htail', default=137.3107)  # 297.62 ft^2 = 27.66 m^2
        self.parameters.declare(name='reference_area_strut', default=137.3107)  # 296.24 ft^2 = 27.55 m^2
        self.parameters.declare(name='reference_area_jury', default=137.3107)  # 12 ft^2 = 1.11 m^2
        self.parameters.declare('drag_comp_list', types=list)
        # self.parameters.declare(name='wing_viscous_cf', default=0.00185)
        # self.parameters.declare(name='wing_viscous_cf', default=0.05)
        self.parameters.declare(name='wing_viscous_cf', default=0.0075)
        self.parameters.declare(name='htail_viscous_cf', default=0.0075)
        self.parameters.declare(name='strut_viscous_cf', default=0.005)
        self.parameters.declare(name='jury_viscous_cf', default=0.0005)

        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        geometry_units = self.parameters['geometry_units']
        reference_area_wing_m2 = self.parameters['reference_area_wing']
        reference_area_htail_m2 = self.parameters['reference_area_htail']
        reference_area_strut_m2 = self.parameters['reference_area_strut']
        reference_area_jury_m2 = self.parameters['reference_area_jury']
        drag_comp_list = self.parameters['drag_comp_list']

        rho = 1.225  # kg/m^3
        kinematic_viscosity = 1.4207E-5  # m^2/s

        ft2_2_m2 = 0.092903
        ft2m = 0.3048

        wing_area = self.declare_variable('wing_area',
                                          shape=(num_nodes, 1))
        htail_area = self.declare_variable('htail_area',
                                          shape=(num_nodes, 1))
        strut_area = self.declare_variable('strut_area',
                                          shape=(num_nodes, 1))
        jury_area = self.declare_variable('jury_area',
                                          shape=(num_nodes, 1))
        chord = self.declare_variable('chord',
                                          shape=(num_nodes, 1))

        rho = self.declare_variable('density', shape=(num_nodes, 1))
        mu = self.declare_variable('dynamic_viscosity', shape=(num_nodes, 1))
        a = self.declare_variable('a', shape=(num_nodes, 1))

        u = self.declare_variable('u', shape=(num_nodes, 1))
        v = self.declare_variable('v', shape=(num_nodes, 1))
        w = self.declare_variable('w', shape=(num_nodes, 1))

        if geometry_units == 'ft':
            wing_area_m2 = wing_area * ft2_2_m2
            htail_area_m2 = htail_area * ft2_2_m2
            strut_area_m2 = strut_area * ft2_2_m2
            jury_area_m2 = jury_area * ft2_2_m2
            chord_m = chord * ft2m
        elif geometry_units == 'm':
            wing_area_m2 = wing_area * 1.
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
                                      shape=(num_nodes, 1), units='rad')
        psi = self.declare_variable(name='psi',
                                    shape=(num_nodes, 1), units='rad')

        gamma = self.declare_variable(name='gamma',
                                      shape=(num_nodes, 1), units='rad', val=0)

        x = self.declare_variable(name='x',
                                  shape=(num_nodes, 1), units='rad', val=0)
        y = self.declare_variable(name='y',
                                  shape=(num_nodes, 1), units='rad', val=0)
        z = self.declare_variable(name='z',
                                  shape=(num_nodes, 1), units='rad', val=0)

        V_inf = (u**2 + v**2 + w**2)**0.5

        VTAS = V_inf + (p+q+r+phi+gamma+x+y+z) * 0
        qBar = 0.5 * rho * VTAS**2
        # M = V_inf / a


        Theta = self.declare_variable('theta', shape=(num_nodes, 1))
        Psi = self.declare_variable('psi', shape=(num_nodes, 1))
        self.register_output('theta_test', Theta * 1)
        self.register_output('psi_test', Psi * 1)

        theta = csdl.arctan(-w/u)
        psi = csdl.arcsin(v/V_inf)

        area_fraction_wing = wing_area_m2 / reference_area_wing_m2
        area_fraction_htail = htail_area_m2 / reference_area_htail_m2
        area_fraction_strut = strut_area_m2 / reference_area_strut_m2
        area_fraction_jury = jury_area_m2 / reference_area_jury_m2
        self.register_output('area_fraction__wing', var=area_fraction_wing)
        self.register_output('area_fraction__htail', var=area_fraction_htail)
        self.register_output('area_fraction__strut', var=area_fraction_strut)
        self.register_output('area_fraction__jury', var=area_fraction_jury)
        self.print_var(var=area_fraction_wing)
        self.print_var(var=area_fraction_htail)
        self.print_var(var=area_fraction_strut)
        self.print_var(var=area_fraction_jury)

        C_D0 = self.create_input('C_D0_compute', val=0, shape=(num_nodes, 1))
        D = 0
        # Loop over components to compute C_D0
        for i, comp in enumerate(drag_comp_list):
            if geometry_units == 'ft':
                S_wet = csdl.expand(self.declare_variable(f"comp_{i}_s_wet", shape=(1, )), shape=(num_nodes, 1)) * ft2_2_m2
                l = csdl.expand(self.declare_variable(f"comp_{i}_l", shape=(1, )), shape=(num_nodes, 1)) * ft2m
            else:
                S_wet = csdl.expand(self.declare_variable(f"comp_{i}_s_wet", shape=(1, )), shape=(num_nodes, 1))
                l = csdl.expand(self.declare_variable(f"comp_{i}_l", shape=(1, )), shape=(num_nodes, 1))

            Q = comp.Q

            Re = VTAS*chord_m/kinematic_viscosity


            if comp.component_type in ['wing', 'strut', 'tail', 'pylon']:  
                    if comp.component_type in ['wing']:             
                        Cf = ( 0.664 / (Re)**0.5 ) * 0 + self.parameters['wing_viscous_cf']*area_fraction_wing
                                    
                        self.register_output(f'Cf_{i}', var=Cf*1)

                        D = D + qBar * Cf * wing_area_m2

                    elif comp.component_type in ['strut']:             
                        Cf = ( 0.664 / (Re)**0.5 ) * 0 + self.parameters['strut_viscous_cf']*area_fraction_strut
                        
                        self.register_output(f'Cf_{i}', var=Cf*1)

                        D = D + qBar * Cf * strut_area_m2

                    elif comp.component_type in ['tail']:             
                        Cf = ( 0.664 / (Re)**0.5 ) * 0 + self.parameters['htail_viscous_cf']*area_fraction_htail

                        self.register_output(f'Cf_{i}', var=Cf*1)

                        D = D + qBar * Cf * htail_area_m2

                    elif comp.component_type in ['pylon']:             
                        Cf = ( 0.664 / (Re)**0.5 ) * 0 + self.parameters['jury_viscous_cf']*area_fraction_jury   

                        self.register_output(f'Cf_{i}', var=Cf*1)

                        D = D + qBar * Cf * jury_area_m2                     


        self.register_output(name='D', var=D)

        F = self.create_output(name='F', shape=(num_nodes, 3), val=0)
        # F[:, 0] = (D * -1.) + 140000
        F[:, 0] = D * -1. 
        # F[:, 0] = D * -1. * csdl.cos(theta) * csdl.cos(psi) 

        M = self.create_output(name='M', shape=(num_nodes, 3), val=0)
        M[:, 0] = D * 0
        return

# # Module imports
# import numpy as np
# import caddee.api as cd
# import m3l
# from python_csdl_backend import Simulator

# if __name__ == '__main__':
#     from examples.advanced_examples.ex_tbw_geometry_setup_trial_1 import (wing_meshes, htail_meshes,  
#                                                                 left_strut_meshes, right_strut_meshes, 
#                                                                 left_jury_meshes, right_jury_meshes, 
#                                                                 vtail_meshes, S_ref, system_model, wing_AR, drag_comp_list, h_tail_area, jury_area, strut_area)
#     # system_model = m3l.Model()
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
#     wing_area_value = system_model.create_input('wing_area', val=wing_ref_area)
#     htail_area_value = system_model.create_input('htail_area', val=wing_ref_area)
#     strut_area_value = system_model.create_input('strut_area', val=wing_ref_area)
#     jury_area_value = system_model.create_input('jury_area', val=wing_ref_area)
#     chord_value = system_model.create_input('wing_chord', val=wing_ref_chord)
#     tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
#                                                                wing_area = wing_area_value, htail_area = htail_area_value, strut_area = strut_area_value, jury_area = jury_area_value,
#                                                             #    area = [wing_area_value,htail_area_value,strut_area_value,strut_area_value, jury_area_value, jury_area_value], 
#                                                                chord = chord_value, drag_comp_list=drag_comp_list)
#     system_model.register_output(tbw_viscous_drag_outputs)
#     caddee_csdl_model = system_model.assemble_csdl()
#     sim = Simulator(caddee_csdl_model,analytics=True)
#     sim.run()
#     cd.print_caddee_outputs(system_model, sim, compact_print=True)
#     # print('Area', sim['tbw_viscous_drag_model.area'])