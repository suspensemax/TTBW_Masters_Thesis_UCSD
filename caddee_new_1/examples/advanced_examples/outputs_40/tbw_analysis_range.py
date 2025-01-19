
# Module imports
import numpy as np
import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
# from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
# from lsdo_rotor import BEMParameters, evaluate_multiple_BEM_models, BEM
# PittPeters, PittPetersParameters, evaluate_multiple_pitt_peters_models
from VAST import FluidProblem, VASTFluidSover, VASTNodalForces
# from lsdo_acoustics import Acoustics, evaluate_multiple_acoustic_models
# from lsdo_motor import evaluate_multiple_motor_sizing_models, evaluate_multiple_motor_analysis_models, MotorAnalysis, MotorSizing
from aframe import BeamMassModel, EBBeam, EBBeamForces
# from caddee.utils.helper_functions.geometry_helpers import  make_vlm_camber_mesh
import time 
import pickle
import pandas as pd

# from examples.advanced_examples.outputs_baseline.tbw_geometry_1_trial_1 import (wing_meshes, htail_meshes, wing_span_dv, wing_root_chord_dv, 
#                                                                         wing_mid_chord_left_dv, wing_tip_chord_left_dv,   
#                                                                         wing_twist_coefficients, wing_sweep_coefficients, 
#                                                                 left_strut_sweep_coefficients, right_strut_sweep_coefficients, 
#                                                                 left_jury_translation_x_coefficients, right_jury_translation_x_coefficients,
#                                                                 left_strut_meshes, right_strut_meshes, 
#                                                                 left_jury_meshes, right_jury_meshes, 
#                                                                 vtail_meshes, S_ref_area, system_model, wing_AR, wing_sweep_angle,
#                                                                 h_tail_area, jury_area, strut_area, angle_beta,
#                                                                 wing_box_beam_mesh, left_strut_box_beam_mesh, right_strut_box_beam_mesh,
#                                                                 left_jury_box_beam_mesh, right_jury_box_beam_mesh, num_wing_beam,
#                                                                 num_strut_beam, num_jury_beam)


from examples.advanced_examples.outputs_40.tbw_geometry_1_trial_1 import (wing_meshes, htail_meshes, wing_span_dv, wing_root_chord_dv, 
                                                                        wing_mid_chord_left_dv, wing_tip_chord_left_dv,   
                                                                        wing_twist_coefficients, 
                                                                left_strut_meshes, right_strut_meshes, 
                                                                left_jury_meshes, right_jury_meshes, 
                                                                vtail_meshes, S_ref_area, system_model, wing_AR, 
                                                                h_tail_area, jury_area, strut_area, angle_beta,
                                                                wing_box_beam_mesh, left_strut_box_beam_mesh, right_strut_box_beam_mesh,
                                                                left_jury_box_beam_mesh, right_jury_box_beam_mesh, num_wing_beam,
                                                                num_strut_beam, num_jury_beam)


t2 = time.time()
t3 = time.time()
caddee = cd.CADDEE()

# region tbw area
from caddee.utils.aircraft_models.tbw.tbw_area_ffd import tbwArea
tbw_area_model_plus_1_point_0g = tbwArea(
    name='TBW_area_plus_1_point_0g',
    counter = 'plus_1_point_0_g',
)
tbw_area_outputs_plus_1_point_0g = tbw_area_model_plus_1_point_0g.evaluate(wing_span_dv = wing_span_dv, 
                                                                           wing_root_chord_dv = wing_root_chord_dv, 
                                                                           wing_mid_chord_left_dv = wing_mid_chord_left_dv,
                                                                           wing_tip_chord_left_dv = wing_tip_chord_left_dv,
                                                                           area_wing = S_ref_area,
                                                                           strut_area = strut_area,
                                                                           AR_wing= wing_AR, 
                                                                           )

system_model.register_output(tbw_area_outputs_plus_1_point_0g)
# endregion

# region tbw weights
from caddee.core.csdl_core.system_model_csdl.sizing_group_csdl.sizing_models_csdl.tbw_weights import TBWMassProperties_m3l

tbw_wt = TBWMassProperties_m3l(
name='TBW_Mass_Properties',
exclude_wing = True,
full_wing = False,
geometry_units='ft',
)
tbw_mass_properties= tbw_wt.evaluate(area = tbw_area_outputs_plus_1_point_0g)
system_model.register_output(tbw_mass_properties)
# endregion

# region Beam sizing 
# create the aframe dictionaries:
joints, bounds, beams = {}, {}, {}

youngs_modulus = 73.1E9 #73.1Gpa - Aluminum 2024-T4
poissons_ratio = 0.33
# density = 6200.  # SI - only wing mass - 5820
# density = 3450.  # SI - only wing mass - 5820
# density = 2780.  # SI - only wing mass - 5820
density = 4150.  # SI - only wing mass - 5820

# density_wing = 5800.
# density_strut_jury = 3000.

density_wing = density
density_strut_jury = density

beams['wing_beam'] = {'E': youngs_modulus,
                        'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                        'rho': density_wing, 'cs': 'box',
                        'nodes': list(range(num_wing_beam))}
beams['right_strut_beam'] = {'E': youngs_modulus,
                        'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                        'rho': density_strut_jury, 'cs': 'box',
                        'nodes': list(range(num_strut_beam))}
beams['left_strut_beam'] = {'E': youngs_modulus,
                                'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                                'rho': density_strut_jury, 'cs': 'box',
                                'nodes': list(range(num_strut_beam))}
beams['right_jury_beam'] = {'E': youngs_modulus,
                                'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                                'rho': density_strut_jury, 'cs': 'box',
                                'nodes': list(range(num_jury_beam))}
beams['left_jury_beam'] = {'E': youngs_modulus,
                            'G': youngs_modulus / (2 * (1 + poissons_ratio)),
                            'rho': density_strut_jury, 'cs': 'box',  
                            'nodes': list(range(num_jury_beam))}


bounds['wing_root'] = {'beam': 'wing_beam', 'node': 10, 'fdim': [1, 1, 1, 1, 1, 1]}
bounds['left_strut_root'] = {'beam': 'left_strut_beam', 'node': 0, 'fdim': [1, 1, 1, 1, 1, 1]}
bounds['right_strut_root'] = {'beam': 'right_strut_beam', 'node': 0, 'fdim': [1, 1, 1, 1, 1, 1]}

joints['wing_leftstrut'] = {'beams': ['wing_beam','left_strut_beam'],'nodes': [16, 8]}
joints['wing_rightstrut'] = {'beams': ['wing_beam', 'right_strut_beam'], 'nodes': [4, 8]}
joints['wing_leftjury'] = {'beams': ['wing_beam', 'left_jury_beam'], 'nodes': [14, 0]}
joints['wing_rightjury'] = {'beams': ['wing_beam', 'right_jury_beam'], 'nodes': [6, 0]}
joints['leftstrut_leftjury'] = {'beams': ['left_strut_beam', 'left_jury_beam'], 'nodes': [4, 2]}
joints['rightstrut_rightjury'] = {'beams': ['right_strut_beam', 'right_jury_beam'], 'nodes': [4, 2]}

inches_to_units = 0.0254
# inches_to_units = 1/12
ft_to_m = 0.3048

sizing = True

wing_beam_ttop_inches_skin = np.array([0.15, 0.25, 0.50, 0.625, 0.6875, 0.480, 0.230, 0.238, 0.240, 0.245, 
                                    0.40, 0.245, 0.240, 0.238, 0.230, 0.480, 0.6875, 0.625, 0.5, 0.25, 0.15]) # incorrect??
wing_beam_ttop_metres_skin = wing_beam_ttop_inches_skin * inches_to_units * 0.93

wing_beam_tbot_inches_skin = np.array([0.15, 0.25, 0.50, 0.625, 0.6875, 0.480, 0.230, 0.238, 0.240, 0.245, 
                                    0.40, 0.245, 0.240, 0.238, 0.230, 0.480, 0.6875, 0.625, 0.5, 0.25, 0.15]) # incorrect??
wing_beam_tbot_metres_skin = wing_beam_tbot_inches_skin * inches_to_units * 0.8
# wing_beam_tbot_metres_skin = wing_beam_tbot_inches_skin * inches_to_units

# wing_beam_tweb_metres = 0.4 * np.ones(num_wing_beam) * inches_to_units 
wing_beam_tweb_metres = 0.3 * np.ones(num_wing_beam) * inches_to_units 
# wing_beam_tweb_metres = 0.40 * np.ones(num_wing_beam) * inches_to_units 
# wing_beam_tweb_metres = wing_beam_ttop_metres_skin * 0.75
# wing_beam_tweb_metres = wing_beam_ttop_metres_skin
# wing_beam_ttop_metres_skin = wing_beam_tweb_metres
# wing_beam_tbot_metres_skin = wing_beam_tweb_metres

# wing_beam_t_top = system_model.create_input(name='wing_beam_ttop' ,val=wing_beam_ttop_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127 # skin thickness
wing_beam_t_top = system_model.create_input(name='wing_beam_ttop' ,val=wing_beam_ttop_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127 # skin thickness
wing_beam_t_bot = system_model.create_input(name='wing_beam_tbot' ,val=wing_beam_tbot_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)
wing_beam_tweb = system_model.create_input(name='wing_beam_tweb' ,val=wing_beam_tweb_metres, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)

# strut_beam_inches_skin = np.array([0.490, 0.425, 0.365, 0.330, 0.290, 0.3125, 0.345, 0.390, 0.425]) # w/o constraint
strut_beam_inches_skin = np.array([0.490, 0.425, 0.365, 0.330, 0.3125, 0.290, 0.270, 0.240, 0.220]) #with constraint

# n_values = num_strut_beam
# expanded_strut_beam_inches_skin = np.interp(
#     np.linspace(0, len(strut_beam_inches_skin) - 1, n_values),
#     np.arange(len(strut_beam_inches_skin)),
#     strut_beam_inches_skin
# )

strut_beam_metres_skin = strut_beam_inches_skin * inches_to_units
# strut_beam_metres_skin = expanded_strut_beam_inches_skin * inches_to_units
# strut_beam_metres = 0.45 * np.ones(num_strut_beam) * inches_to_units 
strut_beam_metres = 0.30 * np.ones(num_strut_beam) * inches_to_units 
# strut_beam_metres = strut_beam_metres_skin * 0.75

left_strut_beam_t_top = system_model.create_input(name='left_strut_beam_ttop' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127
left_strut_beam_t_bot = system_model.create_input(name='left_strut_beam_tbot' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)
left_strut_beam_tweb = system_model.create_input(name='left_strut_beam_tweb' ,val=strut_beam_metres, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)

right_strut_beam_t_top = system_model.create_input(name='right_strut_beam_ttop' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3) # lower=0.00127
right_strut_beam_t_bot = system_model.create_input(name='right_strut_beam_tbot' ,val=strut_beam_metres_skin, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)
right_strut_beam_tweb = system_model.create_input(name='right_strut_beam_tweb' ,val=strut_beam_metres, dv_flag=sizing, lower=0.05 * inches_to_units, upper=1. * inches_to_units, scaler=1E3)

jury_beam_metres = 0.15 * inches_to_units * np.ones((num_jury_beam, ))

left_jury_beam_t_top = system_model.create_input(name='left_jury_beam_ttop' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3) # lower=0.00127
left_jury_beam_t_bot = system_model.create_input(name='left_jury_beam_tbot' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)
left_jury_beam_tweb = system_model.create_input(name='left_jury_beam_tweb' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)
right_jury_beam_t_top = system_model.create_input(name='right_jury_beam_ttop' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3) # lower=0.00127
right_jury_beam_t_bot = system_model.create_input(name='right_jury_beam_tbot' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)
right_jury_beam_tweb = system_model.create_input(name='right_jury_beam_tweb' ,val=jury_beam_metres, dv_flag=sizing, lower=0.01 * inches_to_units, upper=0.75 * inches_to_units, scaler=1E3)


beam_mass_model = BeamMassModel(
beams=beams,
name='beam_mass_model',
mesh_units = 'ft',
)

# wing_beam_mass_props = beam_mass_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes],
#                                     width=[wing_box_beam_mesh.width], height=[wing_box_beam_mesh.height], 
#                                     t_top=[wing_beam_t_top], t_bot=[wing_beam_t_bot] ,t_web=[wing_beam_tweb])

# wing_beam_mass_props = beam_mass_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes * ft_to_m, left_strut_box_beam_mesh.beam_nodes * ft_to_m, right_strut_box_beam_mesh.beam_nodes * ft_to_m, left_jury_box_beam_mesh.beam_nodes * ft_to_m, right_jury_box_beam_mesh.beam_nodes * ft_to_m],
#                                     width=[wing_box_beam_mesh.width * ft_to_m * ft_to_m, left_strut_box_beam_mesh.width * ft_to_m, right_strut_box_beam_mesh.width * ft_to_m, left_jury_box_beam_mesh.width * ft_to_m, right_jury_box_beam_mesh.width * ft_to_m], 
#                                     height=[wing_box_beam_mesh.height * ft_to_m, left_strut_box_beam_mesh.height * ft_to_m, right_strut_box_beam_mesh.height * ft_to_m, left_jury_box_beam_mesh.height * ft_to_m, right_jury_box_beam_mesh.height * ft_to_m], 
#                                     t_top=[wing_beam_t_top, left_strut_beam_t_top, right_strut_beam_t_top, left_jury_beam_t_top, right_jury_beam_t_top],
#                                     t_bot=[wing_beam_t_bot, left_strut_beam_t_bot, right_strut_beam_t_bot, left_jury_beam_t_bot, right_jury_beam_t_bot],
#                                     t_web=[wing_beam_tweb, left_strut_beam_tweb, right_strut_beam_tweb, left_jury_beam_tweb, right_jury_beam_tweb])

wing_beam_mass_props = beam_mass_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes, left_strut_box_beam_mesh.beam_nodes, right_strut_box_beam_mesh.beam_nodes, left_jury_box_beam_mesh.beam_nodes, right_jury_box_beam_mesh.beam_nodes],
                                    width=[wing_box_beam_mesh.width, left_strut_box_beam_mesh.width, right_strut_box_beam_mesh.width, left_jury_box_beam_mesh.width, right_jury_box_beam_mesh.width], 
                                    height=[wing_box_beam_mesh.height, left_strut_box_beam_mesh.height, right_strut_box_beam_mesh.height, left_jury_box_beam_mesh.height, right_jury_box_beam_mesh.height], 
                                    t_top=[wing_beam_t_top, left_strut_beam_t_top, right_strut_beam_t_top, left_jury_beam_t_top, right_jury_beam_t_top],
                                    t_bot=[wing_beam_t_bot, left_strut_beam_t_bot, right_strut_beam_t_bot, left_jury_beam_t_bot, right_jury_beam_t_bot],
                                    t_web=[wing_beam_tweb, left_strut_beam_tweb, right_strut_beam_tweb, left_jury_beam_tweb, right_jury_beam_tweb])

system_model.register_output(wing_beam_mass_props)

total_mass_props_model = cd.TotalMassPropertiesM3L(
name=f"total_mass_properties_model_tbw"
)

total_mass_props = total_mass_props_model.evaluate(component_mass_properties=[wing_beam_mass_props, tbw_mass_properties])
system_model.register_output(total_mass_props)
system_model.add_objective(total_mass_props.mass, scaler=1e-5)
# system_model.add_constraint(total_mass_props.mass, scaler=1e-5)

sizing_full = True

if sizing_full:

    # region +2.5g sizing
    sizing_2_point_5g_condition = cd.CruiseCondition(
    name='plus_2_point_5g_sizing',
    stability_flag=False,
    )

    h_2_point_5g = system_model.create_input('altitude_2_point_5g', val=13106) # in 'm'
    # h_2_point_5g = system_model.create_input('altitude_2_point_5g', val=43000) # in 'ft'
    M_2_point_5g = system_model.create_input('mach_2_point_5g', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_2_point_5g = system_model.create_input('range_2_point_5g', val=23380342.065) # in 'ft'
    # r_2_point_5g = system_model.create_input('range_2_point_5g', val=12006064000) # in 'mm'
    r_2_point_5g = system_model.create_input('range_2_point_5g', val=6482000) # in 'm' #3500 nautical miles
    theta_2_point_5g = system_model.create_input('pitch_angle_2_point_5g', val=np.deg2rad(14.5), dv_flag=True, lower=np.deg2rad(0), upper=np.deg2rad(50))

    # ac sates + atmos
    ac_states_2_point_5g, atmos_2_point_5g = sizing_2_point_5g_condition.evaluate(mach_number=M_2_point_5g, pitch_angle=theta_2_point_5g, cruise_range=r_2_point_5g, altitude=h_2_point_5g)
    system_model.register_output(ac_states_2_point_5g)
    system_model.register_output(atmos_2_point_5g)

    # region VAST solver
    vlm_model = VASTFluidSover(
    name='plus_2_point_5g_vlm_model',
    surface_names=[
        'wing_mesh_plus_2_point_5g',
        'htail_mesh_plus_2_point_5g',
        'right_strut_mesh_plus_2_point_5g',
        'left_strut_mesh_plus_2_point_5g',
        # 'right_jury_mesh_plus_2_point_5g',
        # 'left_jury_mesh_plus_2_point_5g',
    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        (1, ) + htail_meshes.vlm_mesh.shape[1:],
        (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # mesh_unit='m',
    cl0=[0.57, 0., 0., 0.]
    # cl0=[0., 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0.57, 0.]
    # cl0=[0.8,]
    # cl0=[0.60, 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    plus_2_point_5g_elevator = system_model.create_input('plus_2_point_5g_elevator', val=np.deg2rad(20), dv_flag=True, lower=np.deg2rad(-10), upper=np.deg2rad(40))

    # Evaluate VLM outputs and register them as outputs
    vlm_outputs_2_point_5g = vlm_model.evaluate(
    atmosphere=atmos_2_point_5g,
    ac_states=ac_states_2_point_5g,
    meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh,left_strut_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh],
    # meshes=[wing_camber_surface],
    deflections=[None, plus_2_point_5g_elevator, None, None],
    # deflections=[None, plus_2_point_5g_elevator],
    # deflections=[None],
    wing_AR=wing_AR,
    eval_pt=tbw_mass_properties.cg_vector,
    )
    system_model.register_output(vlm_outputs_2_point_5g)

    # Nodal forces
    vlm_force_mapping_model = VASTNodalForces(
    name='vast_plus_2_point_5g_nodal_forces',
    surface_names=[
        f'wing_mesh_plus_2_point_5g',
        # 'htail_mesh_plus_2_point_5g',
        # 'right_strut_mesh_plus_2_point_5g',
        # 'left_strut_mesh_plus_2_point_5g',
        # 'right_jury_mesh_plus_2_point_5g',
        # 'left_jury_mesh_plus_2_point_5g',
    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        # (1, ) + htail_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    initial_meshes=[
        wing_meshes.vlm_mesh,
        # htail_meshes.vlm_mesh,
        # right_strut_meshes.vlm_mesh,
        # left_strut_meshes.vlm_mesh,
        # right_jury_meshes.vlm_mesh,
        # left_jury_meshes.vlm_mesh,
    ]
    )

    oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=[vlm_outputs_2_point_5g.panel_forces[0]], nodal_force_meshes=[wing_meshes.oml_mesh])
    
    wing_oml_forces = oml_forces[0]
    # htail_oml_forces = oml_forces[1]
    # right_strut_oml_forces = oml_forces[2]
    # left_strut_oml_forces = oml_forces[3]
    # right_jury_oml_forces = oml_forces[4]
    # left_jury_oml_forces = oml_forces[5]

    system_model.register_output(wing_oml_forces)
    # system_model.register_output(htail_oml_forces)
    # system_model.register_output(right_strut_oml_forces)
    # system_model.register_output(left_strut_oml_forces)
    # system_model.register_output(left_jury_oml_forces)
    # system_model.register_output(right_jury_oml_forces)

    # endregion

    beam_force_map_model = EBBeamForces(
    name='eb_beam_force_map_plus_2_point_5_g',
    beams=beams,
    exclude_middle=True,
    )

    structural_wing_mesh_forces_2_point_5g = beam_force_map_model.evaluate(
    beam_mesh=wing_box_beam_mesh.beam_nodes,
    nodal_forces=wing_oml_forces,
    nodal_forces_mesh=wing_meshes.oml_mesh
    )

    number_fixed = 3
    left_strut_oml_forces_2_point_5g = m3l.Variable(name = 'left_strut_mesh_forces', shape=(num_strut_beam,number_fixed), value = np.zeros(((num_strut_beam,number_fixed))))
    right_strut_oml_forces_2_point_5g = m3l.Variable(name = 'right_strut_mesh_forces', shape=(num_strut_beam,number_fixed), value = np.zeros(((num_strut_beam,number_fixed))))
    left_jury_oml_forces_2_point_5g = m3l.Variable(name = 'left_jury_mesh_forces', shape=(num_jury_beam,number_fixed), value = np.zeros(((num_jury_beam,number_fixed))))
    right_jury_oml_forces_2_point_5g = m3l.Variable(name = 'right_jury_mesh_forces', shape=(num_jury_beam,number_fixed), value = np.zeros(((num_jury_beam,number_fixed))))

    beam_displacement_model = EBBeam(
    name='eb_beam_2_point_5g',
    beams=beams,
    bounds=bounds,
    joints=joints,
    mesh_units='ft',
    )

    plus_2_point_5g_eb_beam_outputs_wing = beam_displacement_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes, left_strut_box_beam_mesh.beam_nodes, right_strut_box_beam_mesh.beam_nodes, left_jury_box_beam_mesh.beam_nodes, right_jury_box_beam_mesh.beam_nodes],
                                    width=[wing_box_beam_mesh.width, left_strut_box_beam_mesh.width, right_strut_box_beam_mesh.width, left_jury_box_beam_mesh.width, right_jury_box_beam_mesh.width], 
                                    height=[wing_box_beam_mesh.height, left_strut_box_beam_mesh.height, right_strut_box_beam_mesh.height, left_jury_box_beam_mesh.height, right_jury_box_beam_mesh.height], 
                                    t_top=[wing_beam_t_top, left_strut_beam_t_top, right_strut_beam_t_top, left_jury_beam_t_top, right_jury_beam_t_top],
                                    t_bot=[wing_beam_t_bot, left_strut_beam_t_bot, right_strut_beam_t_bot, left_jury_beam_t_bot, right_jury_beam_t_bot],
                                    t_web=[wing_beam_tweb, left_strut_beam_tweb, right_strut_beam_tweb, left_jury_beam_tweb, right_jury_beam_tweb], 
                                    forces = [structural_wing_mesh_forces_2_point_5g,left_strut_oml_forces_2_point_5g,right_strut_oml_forces_2_point_5g,left_jury_oml_forces_2_point_5g, right_jury_oml_forces_2_point_5g])

    # plus_2_point_5g_eb_beam_outputs_wing = beam_displacement_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes,],
    #                                 width=[wing_box_beam_mesh.width,], 
    #                                 height=[wing_box_beam_mesh.height,], 
    #                                 t_top=[wing_beam_t_top,],
    #                                 t_bot=[wing_beam_t_bot,],
    #                                 t_web=[wing_beam_tweb,])


    # system_model.add_constraint(plus_1_point_0g_eb_beam_outputs.bot_buckling, upper=1)
    system_model.add_constraint(plus_2_point_5g_eb_beam_outputs_wing.top_buckling[0], upper=1.3)
    # system_model.add_constraint(plus_2_point_5g_eb_beam_outputs_wing.displacements[0], lower=-1.2, upper=0.5) # baseline case
    # system_model.add_constraint(plus_2_point_5g_eb_beam_outputs_wing.displacements[0], lower=-1.5, upper=0.5)
    yield_strength=324E6
    # yield_strength=280E7
    # yield_strength=100E7
    FoS=1.5
    # # system_model.add_constraint(plus_2_point_5g_eb_beam_outputs_wing.stresses, upper=350E6/1., scaler=0.5e-8)
    system_model.add_constraint(plus_2_point_5g_eb_beam_outputs_wing.stresses[0], upper=yield_strength / FoS, scaler=1e-8) #90,000 PSI 
    # system_model.add_constraint(plus_2_point_5g_eb_beam_outputs_wing.stresses[0], upper=yield_strength / FoS, scaler=1e-8) #90,000 PSI 

    system_model.register_output(plus_2_point_5g_eb_beam_outputs_wing)
    # print(plus_2_point_5g_eb_beam_outputs_wing.stresses.shape)
    # print(plus_2_point_5g_eb_beam_outputs_wing.stresses[0].shape)

    # maximiize

    # wing_beam_mass_after = plus_2_point_5g_eb_beam_outputs_wing.mass
    # wing_beam_mass_after = plus_2_point_5g_eb_beam_outputs_wing.mass[0]
    # left_strut_beam_mass_after = plus_2_point_5g_eb_beam_outputs_wing.mass[1]
    # right_strut_beam_mass_after = plus_2_point_5g_eb_beam_outputs_wing.mass[2]
    # left_jury_beam_mass_after = plus_2_point_5g_eb_beam_outputs_wing.mass[3]
    # right_jury_beam_mass_after = plus_2_point_5g_eb_beam_outputs_wing.mass[4]

    # # NOTE:
    # # We are creating a new MassProperties object for the wing beam where
    # # we combine the cg and inertia tensor that the beam analysis model
    # # outputs with the beam mass, which is based on the skin, and spar
    # # thicknesses. This is because the +2.5g condition will size the top
    # # skin while the -1g condition will size the bottom skin (if buckling
    # # is considered).

    wing_beam_mass_props = cd.MassProperties(
    # mass=plus_2_point_5g_eb_beam_outputs_wing.mass,
    mass=plus_2_point_5g_eb_beam_outputs_wing.struct_mass,
    # cg_vector=plus_2_point_5g_eb_beam_outputs_wing.cg_vector,
    cg_vector=plus_2_point_5g_eb_beam_outputs_wing.cg_vector_not_list,
    # inertia_tensor=plus_2_point_5g_eb_beam_outputs_wing.inertia_tensor,
    inertia_tensor=plus_2_point_5g_eb_beam_outputs_wing.inertia_tensor_not_list,
    )

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    throttle_plus_2_point_5g = system_model.create_input('throttle_plus_2_point_5g', val = 0.75, shape=(1, ), dv_flag=True, lower=0., upper=1.5)
    tbw_left_prop_model_plus_2_point_5g = tbwPropulsionModel(
        name='TBW_Propulsion_plus_2_point_5g',
        counter = 'plus_2_point_5g',
    )
    tbw_left_propulsion_outputs_plus_2_point_5g = tbw_left_prop_model_plus_2_point_5g.evaluate(throttle = throttle_plus_2_point_5g)
    system_model.register_output(tbw_left_propulsion_outputs_plus_2_point_5g)
    # endregion

    # region Viscous drag
    # # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # # system_model.register_output('S_area',S_ref.value)
    # wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 m
    # tbw_viscous_drag_model_plus_2_point_5g = Tbw_Viscous_Drag_Model(
    #         name = 'tbw_viscous_drag_plus_1_point_0g',
    #         geometry_units='ft',
    #         counter = 'plus_2_point_5g', 
    # )
    # # wing_ref_area = S_ref.value  
    # # wing_ref_area = tbw_area_outputs_plus_2_point_5g.wing_area 
    # # area_value = system_model.create_input('area_plus_2_point_5_g', val=wing_ref_area)
    # chord_value = system_model.create_input('chord_plus_2_point_5g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_2_point_5g = tbw_viscous_drag_model_plus_2_point_5g.evaluate(atmos = atmos_2_point_5g, ac_states=ac_states_2_point_5g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs_2_point_5g)
    # # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    # system_model.register_output(tbw_viscous_drag_outputs_plus_2_point_5g)

    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location_2 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 import Tbw_Viscous_Drag_Model
    from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_sweep_3 import Tbw_Viscous_Drag_Model

    # system_model.register_output('S_area',S_ref.value)
    wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 m
    tbw_viscous_drag_model_plus_2_point_5g = Tbw_Viscous_Drag_Model(
            name = 'tbw_viscous_drag_plus_2_point_5g',
            geometry_units='ft',
            counter = 'plus_2_point_5g', 
    )
    # wing_ref_area = S_ref.value  
    # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
    # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
    wing_sweep_angle_plus_2_point_5g = system_model.create_input(name='wing_sweep_angle_plus_2_point_5g', shape=(1,), val=np.deg2rad(14.06),)
    chord_value = system_model.create_input('chord_plus_2_point_5g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_2_point_5g = tbw_viscous_drag_model_plus_2_point_5g.evaluate(atmos = atmos_2_point_5g, ac_states=ac_states_2_point_5g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs_2_point_5g, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_2_point_5g, ) # tbw_viscous_drag_try_model_sweep_2 & tbw_viscous_drag_model_63_location
    # tbw_viscous_drag_outputs_plus_2_point_5g = tbw_viscous_drag_model_plus_2_point_5g.evaluate(atmos = atmos_2_point_5g, ac_states=ac_states_2_point_5g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs_2_point_5g, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_2_point_5g, angle_beta = angle_beta,) # tbw_viscous_drag_model_63_location_2
    tbw_viscous_drag_outputs_plus_2_point_5g = tbw_viscous_drag_model_plus_2_point_5g.evaluate(atmos = atmos_2_point_5g, ac_states=ac_states_2_point_5g, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = chord_value, vlm_outputs = vlm_outputs_2_point_5g, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = wing_sweep_angle_plus_2_point_5g, mach_number = M_2_point_5g, angle_beta = angle_beta,) # tbw_viscous_drag_model_63_location_2
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(tbw_viscous_drag_outputs_plus_2_point_5g)


    # endregion

    plus_2_point_5g_trim_variables = sizing_2_point_5g_condition.assemble_trim_residual(
    # mass_properties=[tbw_mass_properties],
    mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    # mass_properties=[wing_beam_mass_props],
    aero_propulsive_outputs=[vlm_outputs_2_point_5g, tbw_left_propulsion_outputs_plus_2_point_5g, tbw_viscous_drag_outputs_plus_2_point_5g],
    # aero_propulsive_outputs=[vlm_outputs_2_point_5g, tbw_left_propulsion_outputs],
    # aero_propulsive_outputs=[vlm_outputs_2_point_5g, tbw_viscous_drag_outputs],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs],
    # aero_propulsive_outputs=[vlm_outputs_2_point_5g],
    ac_states=ac_states_2_point_5g,
    load_factor=2.5,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(plus_2_point_5g_trim_variables)

    system_model.add_constraint(plus_2_point_5g_trim_variables.accelerations, equals=0.)
    # endregion

    # region -1.0g sizing
    sizing_minus_1_point_0_g_condition = cd.CruiseCondition(
    name='minus_1_point_0_g_sizing',
    stability_flag=False,
    )

    h_minus_1_point_0_g = system_model.create_input('altitude_minus_1_point_0_g', val=13106) # in 'm'
    # h_minus_1_point_0_g = system_model.create_input('altitude_minus_1_point_0_g', val=43000) # in 'ft'
    M_minus_1_point_0_g = system_model.create_input('mach_minus_1_point_0_g', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
    # r_minus_1_point_0_g = system_model.create_input('range_minus_1_point_0_g', val=23380342.065) # in 'ft'
    # r_minus_1_point_0_g = system_model.create_input('range_minus_1_point_0_g', val=12006064000) # in 'm'
    r_minus_1_point_0_g = system_model.create_input('range_minus_1_point_0_g', val=6482000) # in 'nautical miles'
    theta_minus_1_point_0_g = system_model.create_input('pitch_angle_minus_1_point_0_g', val=np.deg2rad(-15), dv_flag=True, lower=np.deg2rad(-40), upper=np.deg2rad(40))

    # ac sates + atmos
    ac_states_minus_1_point_0_g, atmos_minus_1_point_0_g = sizing_minus_1_point_0_g_condition.evaluate(mach_number=M_minus_1_point_0_g, pitch_angle=theta_minus_1_point_0_g, cruise_range=r_minus_1_point_0_g, altitude=h_minus_1_point_0_g)
    system_model.register_output(ac_states_minus_1_point_0_g)
    system_model.register_output(atmos_minus_1_point_0_g)

    # region VAST solver
    vlm_model = VASTFluidSover(
    name='minus_1g_vlm_model',
    surface_names=[
        'wing_mesh_minus_1_point_0_g_minus_1_point_0_g',
        'htail_mesh_minus_1_point_0_g_minus_1_point_0_g',
        'right_strut_mesh_minus_1_point_0_g',
        'left_strut_mesh_minus_1_point_0_g',
        # 'right_jury_mesh_inus_1_point_0_g',
        # 'left_jury_mesh_minus_1_point_0_g',
    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        (1, ) + htail_meshes.vlm_mesh.shape[1:],
        (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
    mesh_unit='ft',
    # mesh_unit='m',
    cl0=[0.5, 0., 0., 0.]
    # cl0=[0., 0., 0., 0.]
    # cl0=[0.625]
    # cl0=[0.01]
    # cl0=[0.52, 0.]
    # cl0=[0., 0., 0., 0., 0., 0.]
    )

    minus_1_point_0_g_elevator = system_model.create_input('minus_1_point_0_g_elevator', val=np.deg2rad(-7), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

    # Evaluate VLM outputs and register them as outputs
    vlm_outputs_minus_1_point_0g = vlm_model.evaluate(
    atmosphere=atmos_minus_1_point_0_g,
    ac_states=ac_states_minus_1_point_0_g,
    meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh,left_strut_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh],
    # meshes=[wing_meshes.vlm_mesh],
    # meshes=[wing_camber_surface],
    deflections=[None, minus_1_point_0_g_elevator, None, None],
    # deflections=[None, minus_1_point_0_g_elevator],
    # deflections=[None],
    wing_AR=wing_AR,
    eval_pt=tbw_mass_properties.cg_vector,
    )
    system_model.register_output(vlm_outputs_minus_1_point_0g)

    # Nodal forces
    vlm_force_mapping_model = VASTNodalForces(
    name='vast_minus_1_point_0g_nodal_forces',
    surface_names=[
        f'wing_mesh_minus_1_point_0g',
        # f'htail_mesh_minus_1_point_0g',
        # f'right_strut_mesh_minus_1_point_0g',
        # f'left_strut_mesh_minus_1_point_0g',
        # f'right_jury_mesh_minus_1_point_0g',
        # f'left_jury_mesh_minus_1_point_0g',

    ],
    surface_shapes=[
        (1, ) + wing_meshes.vlm_mesh.shape[1:],
        # (1, ) + htail_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
        # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],
        # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
    ],
    initial_meshes=[
        wing_meshes.vlm_mesh,
        # htail_meshes.vlm_mesh,
        # right_strut_meshes.vlm_mesh,
        # left_strut_meshes.vlm_mesh,
        # right_jury_meshes.vlm_mesh,
        # left_jury_meshes.vlm_mesh,
    ]
    )

    # oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=vlm_outputs.panel_forces, nodal_force_meshes=[wing_oml_mesh, htail_meshes.oml_mesh, right_strut_oml_mesh, left_strut_oml_mesh, right_jury_meshes.oml_mesh, left_jury_meshes.oml_mesh])
    # oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=vlm_outputs.panel_forces, nodal_force_meshes=[wing_oml_mesh, htail_meshes.oml_mesh, right_strut_oml_mesh, left_strut_oml_mesh]) 
    # oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=vlm_outputs.panel_forces, nodal_force_meshes=[wing_meshes.oml_mesh,htail_meshes.oml_mesh])
    oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=[vlm_outputs_minus_1_point_0g.panel_forces[0]], nodal_force_meshes=[wing_meshes.oml_mesh])
    
    wing_oml_forces = oml_forces[0]
    # htail_oml_forces = oml_forces[1]
    # right_strut_oml_forces = oml_forces[2]
    # left_strut_oml_forces = oml_forces[3]
    # right_jury_oml_forces = oml_forces[4]
    # left_jury_oml_forces = oml_forces[5]

    system_model.register_output(wing_oml_forces)
    # system_model.register_output(htail_oml_forces)
    # system_model.register_output(left_strut_oml_forces)
    # system_model.register_output(right_strut_oml_forces)
    # system_model.register_output(left_jury_oml_forces)
    # system_model.register_output(right_jury_oml_forces)

    # endregion

    beam_force_map_model = EBBeamForces(
    name='eb_beam_force_map_minus_1_point_0g',
    beams=beams,
    exclude_middle=True,
    )

    structural_wing_mesh_forces_minus_1_point_0g = beam_force_map_model.evaluate(
    beam_mesh=wing_box_beam_mesh.beam_nodes,
    nodal_forces=wing_oml_forces,
    nodal_forces_mesh=wing_meshes.oml_mesh
    )

    number_fixed = 3
    left_strut_oml_forces_minus_1_point_0g = m3l.Variable(name = 'left_strut_mesh_forces', shape=(num_strut_beam,number_fixed), value = np.zeros(((num_strut_beam,number_fixed))))
    right_strut_oml_forces_minus_1_point_0g = m3l.Variable(name = 'right_strut_mesh_forces', shape=(num_strut_beam,number_fixed), value = np.zeros(((num_strut_beam,number_fixed))))
    left_jury_oml_forces_minus_1_point_0g = m3l.Variable(name = 'left_jury_mesh_forces', shape=(num_jury_beam,number_fixed), value = np.zeros(((num_jury_beam,number_fixed))))
    right_jury_oml_forces_minus_1_point_0g = m3l.Variable(name = 'right_jury_mesh_forces', shape=(num_jury_beam,number_fixed), value = np.zeros(((num_jury_beam,number_fixed))))


    beam_displacement_model = EBBeam(
    name='eb_beam_minus_1_point_0g',
    beams=beams,
    bounds=bounds,
    joints=joints,
    mesh_units='ft',
    )

    # minus_1_point_0g_eb_beam_outputs_wing = beam_displacement_model.evaluate(beam_mesh=wing_box_beam_mesh, t_top=wing_beam_t_top, t_bot=wing_beam_t_bot, t_web=wing_beam_tweb, forces=structural_wing_mesh_forces_1_point_0g)

    minus_1_point_0g_eb_beam_outputs_wing = beam_displacement_model.evaluate(beam_nodes=[wing_box_beam_mesh.beam_nodes, left_strut_box_beam_mesh.beam_nodes, right_strut_box_beam_mesh.beam_nodes, left_jury_box_beam_mesh.beam_nodes, right_jury_box_beam_mesh.beam_nodes],
                                    width=[wing_box_beam_mesh.width, left_strut_box_beam_mesh.width, right_strut_box_beam_mesh.width, left_jury_box_beam_mesh.width, right_jury_box_beam_mesh.width], 
                                    height=[wing_box_beam_mesh.height, left_strut_box_beam_mesh.height, right_strut_box_beam_mesh.height, left_jury_box_beam_mesh.height, right_jury_box_beam_mesh.height], 
                                    t_top=[wing_beam_t_top, left_strut_beam_t_top, right_strut_beam_t_top, left_jury_beam_t_top, right_jury_beam_t_top],
                                    t_bot=[wing_beam_t_bot, left_strut_beam_t_bot, right_strut_beam_t_bot, left_jury_beam_t_bot, right_jury_beam_t_bot],
                                    t_web=[wing_beam_tweb, left_strut_beam_tweb, right_strut_beam_tweb, left_jury_beam_tweb, right_jury_beam_tweb], 
                                    forces = [structural_wing_mesh_forces_minus_1_point_0g,left_strut_oml_forces_minus_1_point_0g,right_strut_oml_forces_minus_1_point_0g,left_jury_oml_forces_minus_1_point_0g, right_jury_oml_forces_minus_1_point_0g])
    
    yield_strength=324E6
    FoS=1.5

    system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.bot_buckling[0], upper=1.3)
    # system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.top_buckling, upper=1.3)
    # system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.displacements, lower=-0.5, upper=0.5)
    # system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.displacements[0], lower=-0.5, upper=0.5)
    # system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.displacements[0], lower=-1.2, upper=0.5)
    # system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.stresses, upper=350E6/1., scaler=0.5e-8)
    system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.stresses[0], upper=yield_strength/FoS, scaler=0.5e-8)
    # system_model.add_constraint(minus_1_point_0g_eb_beam_outputs_wing.stresses, upper=620E6/1., scaler=0.5e-8) #90,000 PSI 

    system_model.register_output(minus_1_point_0g_eb_beam_outputs_wing)

    # # NOTE:s
    # # We are creating a new MassProperties object for the wing beam where
    # # we combine the cg and inertia tensor that the beam analysis model
    # # outputs with the beam mass, which is based on the skin, and spar
    # # thicknesses. This is because the +2.5g condition will size the top
    # # skin while the -1g condition will size the bottom skin (if buckling
    # # is considered).

    wing_beam_mass_props = cd.MassProperties(
    # mass=minus_1_point_0g_eb_beam_outputs_wing.mass,
    mass=minus_1_point_0g_eb_beam_outputs_wing.struct_mass,
    # cg_vector=minus_1_point_0g_eb_beam_outputs_wing.cg_vector,
    cg_vector=minus_1_point_0g_eb_beam_outputs_wing.cg_vector_not_list,
    # inertia_tensor=minus_1_point_0g_eb_beam_outputs_wing.inertia_tensor,
    inertia_tensor=minus_1_point_0g_eb_beam_outputs_wing.inertia_tensor_not_list,
    )

    # region Propulsion Solvers
    from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
    throttle_minus_1_point_0g = system_model.create_input('throttle_minus_1_point_0g', val = 0.5, shape=(1, ), dv_flag=True, lower=0., upper=1.)
    tbw_left_prop_model_minus_1_point_0g = tbwPropulsionModel(
        name='TBW_Propulsion_minus_1_point_0g',
        counter = 'minus_1_point_0g',
    )
    tbw_left_propulsion_outputs_minus_1_point_0g = tbw_left_prop_model_minus_1_point_0g.evaluate(throttle = throttle_minus_1_point_0g)
    system_model.register_output(tbw_left_propulsion_outputs_minus_1_point_0g)
    # endregion

    # region Viscous drag
    # # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # # system_model.register_output('S_area',S_ref.value)
    # wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 m
    # tbw_viscous_drag_model_minus_1_point_0g = Tbw_Viscous_Drag_Model(
    #         name = 'tbw_viscous_drag_plus_1_point_0g',
    #         geometry_units='ft',
    #         counter = 'minus_1_point_0g', 
    # )
    # # wing_ref_area = S_ref.value  
    # # wing_ref_area = tbw_area_outputs_minus_1_point_0g.wing_area 
    # # area_value = system_model.create_input('area_minus_1_point_0_g', val=wing_ref_area)
    # chord_value = system_model.create_input('chord_minus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_minus_1_point_0g = tbw_viscous_drag_model_minus_1_point_0g.evaluate(atmos = atmos_minus_1_point_0_g, ac_states=ac_states_minus_1_point_0_g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs_minus_1_point_0g)
    # # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    # system_model.register_output(tbw_viscous_drag_outputs_minus_1_point_0g)

    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
    # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_63_location_2 import Tbw_Viscous_Drag_Model
    # from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model_sweep_2 import Tbw_Viscous_Drag_Model
    from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_sweep_3 import Tbw_Viscous_Drag_Model

    # system_model.register_output('S_area',S_ref.value)
    wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 m
    tbw_viscous_drag_model_minus_1_point_0g = Tbw_Viscous_Drag_Model(
            name = 'tbw_viscous_drag_minus_1_point_0g',
            geometry_units='ft',
            counter = 'minus_1_point_0g', 
    )
    # wing_ref_area = S_ref.value  
    # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
    # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
    wing_sweep_angle_minus_1_point_0_g = system_model.create_input(name='wing_sweep_angle_minus_1_point_0_g', shape=(1,), val=np.deg2rad(14.06),)
    chord_value = system_model.create_input('chord_minus_1_point_0g', val=wing_ref_chord)

    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area,) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, sweep_angle = Cd_wave) # tbw_viscous_drag_try_model
    # tbw_viscous_drag_outputs_minus_1_point_0g = tbw_viscous_drag_model_minus_1_point_0g.evaluate(atmos = atmos_2_point_5g, ac_states=ac_states_2_point_5g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
    #                                                                                         chord = chord_value, vlm_outputs = vlm_outputs_2_point_5g, 
    #                                                                                         h_tail_area = h_tail_area, strut_area = strut_area, 
    #                                                                                         sweep_angle = wing_sweep_angle, mach_number = M_2_point_5g, ) # tbw_viscous_drag_try_model_sweep_2 & tbw_viscous_drag_model_63_location
    tbw_viscous_drag_outputs_minus_1_point_0g = tbw_viscous_drag_model_minus_1_point_0g.evaluate(atmos = atmos_minus_1_point_0_g, ac_states=ac_states_minus_1_point_0_g, 
                                                                                            tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, 
                                                                                            chord = chord_value, vlm_outputs = vlm_outputs_minus_1_point_0g, 
                                                                                            h_tail_area = h_tail_area, strut_area = strut_area, 
                                                                                            sweep_angle = wing_sweep_angle_minus_1_point_0_g, mach_number = M_minus_1_point_0_g, angle_beta = angle_beta,) # tbw_viscous_drag_model_63_location_2
    # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
    #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(tbw_viscous_drag_outputs_minus_1_point_0g)
    # endregion

    from caddee.utils.aircraft_models.tbw.Tbw_L_over_D_final import Tbw_L_over_D_Model

    tbw_L_over_D_minus_1_point_0g = Tbw_L_over_D_Model(
        name = 'tbw_L_over_D_final_minus_1_point_0g',
        counter = 'minus_1_point_0g', 
    )
    # tbw_L_over_D_outputs_minus_1_point_0g = tbw_L_over_D_minus_1_point_0g.evaluate(Total_lift_value_vlm = vlm_outputs.Total_lift, Total_drag_value_vlm = vlm_outputs.Total_drag, viscous_drag_forces = tbw_viscous_drag_outputs_minus_1_point_0g.forces)
    tbw_L_over_D_outputs_minus_1_point_0g = tbw_L_over_D_minus_1_point_0g.evaluate(vlm_outputs = vlm_outputs_minus_1_point_0g, tbw_viscous_drag = tbw_viscous_drag_outputs_minus_1_point_0g)
    # tbw_L_over_D_outputs_minus_1_point_0g = tbw_L_over_D_minus_1_point_0g.evaluate()
    # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
    system_model.register_output(tbw_L_over_D_outputs_minus_1_point_0g)

    minus_1_point_0_g_trim_variables = sizing_minus_1_point_0_g_condition.assemble_trim_residual(
    # mass_properties=[tbw_mass_properties],
    mass_properties=[tbw_mass_properties, wing_beam_mass_props],
    aero_propulsive_outputs=[vlm_outputs_minus_1_point_0g, tbw_left_propulsion_outputs_minus_1_point_0g, tbw_viscous_drag_outputs_minus_1_point_0g],
    # aero_propulsive_outputs=[vlm_outputs_minus_1_point_0g, tbw_viscous_drag_outputs],
    # aero_propulsive_outputs=[tbw_viscous_drag_outputs],
    # aero_propulsive_outputs=[vlm_outputs_minus_1_point_0g],
    ac_states=ac_states_minus_1_point_0_g,
    load_factor=-1.0,
    ref_pt=tbw_mass_properties.cg_vector,
    # ref_pt=wing_beam_mass_props.cg_vector,
    )
    system_model.register_output(minus_1_point_0_g_trim_variables)

    system_model.add_constraint(minus_1_point_0_g_trim_variables.accelerations, equals=0.)
    # system_model.add_constraint(plus_minus_1_point_0_g_trim_variables.du_dt, equals=0.)
    # system_model.add_constraint(plus_minus_1_point_0_g_trim_variables.dv_dt, equals=0.)
    # system_model.add_constraint(plus_minus_1_point_0_g_trim_variables.dw_dt, equals=0.)
    # system_model.add_constraint(plus_minus_1_point_0_g_trim_variables.dp_dt, equals=0.)
    # system_model.add_constraint(plus_minus_1_point_0_g_trim_variables.dq_dt, equals=0.)
    # system_model.add_constraint(plus_minus_1_point_0_g_trim_variables.dr_dt, equals=0.)
    # system_model.add_objective(minus_1_point_0_g_trim_variables.accelerations, scaler=1e-3)
    # system_model.add_objective(tbw_L_over_D_outputs_minus_1_point_0g.L_over_D, scaler = -1)

    # endregion

    twist_geo_dv = True # span and chord and twist working

    if twist_geo_dv:
        sizing_1_point_0g_condition = cd.CruiseCondition(
        name='plus_1_point_0g_sizing',
        stability_flag=False,
        )

        h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=13106) # in 'm'
        # h_1_point_0g = system_model.create_input('altitude_1_point_0g', val=43000) # in 'ft'
        M_1_point_0g = system_model.create_input('mach_1_point_0g', val=0.70, dv_flag=False, lower=0.65, upper=0.85)
        # r_1_point_0g = system_model.create_input('range_1_point_0g', val=23380342.065) # in 'ft'
        # r_1_point_0g = system_model.create_input('range_1_point_0g', val=12006064000) # in 'm'
        r_1_point_0g = system_model.create_input('range_1_point_0g', val=6482000) # in m # 'nautical miles'
        theta_1_point_0g = system_model.create_input('pitch_angle_1_point_0g', val=np.deg2rad(0.), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

        # ac sates + atmos
        ac_states_1_point_0g, atmos_1_point_0g = sizing_1_point_0g_condition.evaluate(mach_number=M_1_point_0g, pitch_angle=theta_1_point_0g, cruise_range=r_1_point_0g, altitude=h_1_point_0g)
        system_model.register_output(ac_states_1_point_0g)
        system_model.register_output(atmos_1_point_0g)

        # region VAST solver
        vlm_model = VASTFluidSover(
        name='plus_1g_vlm_model',
        surface_names=[
            'wing_mesh_plus_1_point_0g',
            'htail_mesh_plus_1_point_0g',
            'right_strut_mesh_plus_1_point_0g',
            'left_strut_mesh_plus_1_point_0g',
            # 'right_jury_mesh_plus_1_point_0g',
            # 'left_jury_mesh_plus_1_point_0g',
        ],
        surface_shapes=[
            (1, ) + wing_meshes.vlm_mesh.shape[1:],
            (1, ) + htail_meshes.vlm_mesh.shape[1:],
            (1, ) + right_strut_meshes.vlm_mesh.shape[1:],
            (1, ) + left_strut_meshes.vlm_mesh.shape[1:],
            # (1, ) + right_jury_meshes.vlm_mesh.shape[1:],
            # (1, ) + left_jury_meshes.vlm_mesh.shape[1:],
        ],
        fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
        mesh_unit='ft',
        # OutputsVLM = None,
        # mesh_unit='m',
        cl0=[0.68, 0., 0., 0.]
        # cl0=[0., 0., 0., 0.]
        # cl0=[0.625]
        # cl0=[0.01]
        # cl0=[0.72, 0.]
        # cl0=[0., 0., 0., 0., 0., 0.]
        )

        plus_1_point_0_g_elevator = system_model.create_input('plus_1_point_0_g_elevator', val=np.deg2rad(0), dv_flag=True, lower=np.deg2rad(-20), upper=np.deg2rad(20))

        # Evaluate VLM outputs and register them as outputs
        vlm_outputs = vlm_model.evaluate(
        atmosphere=atmos_1_point_0g,
        ac_states=ac_states_1_point_0g,
        meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh,left_strut_meshes.vlm_mesh],
        # meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh],
        # meshes=[wing_meshes.vlm_mesh],
        # meshes=[wing_camber_surface],
        deflections=[None, plus_1_point_0_g_elevator, None, None],
        # deflections=[None, plus_1_point_0_g_elevator],
        # deflections=[None],
        # wing_AR=wing_AR,
        wing_AR=tbw_area_outputs_plus_1_point_0g.wing_AR,
        eval_pt=tbw_mass_properties.cg_vector,
        )
        system_model.register_output(vlm_outputs)
        # endregion

        # region Propulsion Solvers
        from caddee.utils.aircraft_models.tbw.tbw_propulsion import tbwPropulsionModel
        throttle_plus_1_point_0g = system_model.create_input('throttle_plus_1_point_0g', val = 0.23, shape=(1, ), dv_flag=True, lower=0., upper=1.)
        tbw_left_prop_model_plus_1_point_0g = tbwPropulsionModel(
            name='TBW_Propulsion_plus_1_point_0g',
            counter = 'plus_1_point_0g',
        )
        tbw_left_propulsion_outputs_plus_1_point_0g = tbw_left_prop_model_plus_1_point_0g.evaluate(throttle = throttle_plus_1_point_0g)
        system_model.register_output(tbw_left_propulsion_outputs_plus_1_point_0g)
        # endregion

        # region Viscous drag
        # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model import TbwViscousDragModel
        # from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbw_Viscous_Drag_Model
        from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_try_model import Tbw_Viscous_Drag_Model
        # system_model.register_output('S_area',S_ref.value)
        wing_ref_chord = 110.286  # MAC: 110.286 ft = 33.6151728 m
        tbw_viscous_drag_model_plus_1_point_0g = Tbw_Viscous_Drag_Model(
                name = 'tbw_viscous_drag_plus_1_point_0g',
                geometry_units='ft',
                counter = 'plus_1_point_0g', 
        )
        # wing_ref_area = S_ref.value  
        # wing_ref_area = tbw_area_outputs_plus_1_point_0g.wing_area 
        # area_value = system_model.create_input('area_plus_1_point_0_g', val=wing_ref_area)
        chord_value = system_model.create_input('chord_plus_1_point_0g', val=wing_ref_chord)

        tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
                                                                                                tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs, 
                                                                                                h_tail_area = h_tail_area, strut_area = strut_area,)
        # tbw_viscous_drag_outputs_plus_1_point_0g = tbw_viscous_drag_model_plus_1_point_0g.evaluate(atmos = atmos_1_point_0g, ac_states=ac_states_1_point_0g, 
        #                                                                                         tbw_area_outputs_plus_1_point_0g = tbw_area_outputs_plus_1_point_0g, chord = chord_value, vlm_outputs = vlm_outputs,)
        # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
        system_model.register_output(tbw_viscous_drag_outputs_plus_1_point_0g)
        # endregion

        from caddee.utils.aircraft_models.tbw.Tbw_L_over_D_final import Tbw_L_over_D_Model

        tbw_L_over_D_plus_1_point_0g = Tbw_L_over_D_Model(
            name = 'tbw_L_over_D_final_plus_1_point_0g',
            counter = 'plus_1_point_0g', 
        )
        # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(Total_lift_value_vlm = vlm_outputs.Total_lift, Total_drag_value_vlm = vlm_outputs.Total_drag, viscous_drag_forces = tbw_viscous_drag_outputs_plus_1_point_0g.forces)
        tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate(vlm_outputs = vlm_outputs, tbw_viscous_drag = tbw_viscous_drag_outputs_plus_1_point_0g)
        # tbw_L_over_D_outputs_plus_1_point_0g = tbw_L_over_D_plus_1_point_0g.evaluate()
        # tbw_viscous_drag_outputs = tbw_viscous_drag_model.evaluate(area = area_value, chord = chord_value)
        system_model.register_output(tbw_L_over_D_outputs_plus_1_point_0g)

        plus_1_point_0g_trim_variables = sizing_1_point_0g_condition.assemble_trim_residual(
        # mass_properties=[tbw_mass_properties],
        mass_properties=[tbw_mass_properties, wing_beam_mass_props],
        aero_propulsive_outputs=[vlm_outputs, tbw_left_propulsion_outputs_plus_1_point_0g, tbw_viscous_drag_outputs_plus_1_point_0g],
        # aero_propulsive_outputs=[vlm_outputs, tbw_viscous_drag_outputs_plus_1_point_0g],
        # aero_propulsive_outputs=[tbw_viscous_drag_outputs_plus_1_point_0g],
        # aero_propulsive_outputs=[vlm_outputs],
        ac_states=ac_states_1_point_0g,
        load_factor=1.0,
        ref_pt=tbw_mass_properties.cg_vector,
        # ref_pt=wing_beam_mass_props.cg_vector,
        )
        system_model.register_output(plus_1_point_0g_trim_variables)
                
        system_model.add_constraint(plus_1_point_0g_trim_variables.accelerations, equals=0.)
        # system_model.add_objective(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)
        system_model.add_constraint(tbw_L_over_D_outputs_plus_1_point_0g.L_over_D, scaler = -1)


    # from caddee.utils.aircraft_models.tbw.tbw_range import Tbw_range_Model
    from caddee.utils.aircraft_models.tbw.tbw_range_2 import Tbw_range_Model

    tbw_range_plus_1_point_0g = Tbw_range_Model(
        name = 'tbw_range_final',
        counter = 'plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g', 
    )
    # tbw_range_outputs_plus_1_point_0g = tbw_range_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props.mass, thrust = tbw_left_propulsion_outputs_plus_1_point_0g.thrust)
    # tbw_range_outputs_plus_1_point_0g = tbw_range_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, total_mass_props = total_mass_props, thrust = tbw_left_propulsion_outputs_plus_1_point_0g)
    tbw_range_outputs_plus_1_point_0g = tbw_range_plus_1_point_0g.evaluate(L_over_D = tbw_L_over_D_outputs_plus_1_point_0g, wing_beam_mass_props = wing_beam_mass_props.mass, 
                                                                           tbw_mass_properties = tbw_mass_properties.mass, 
                                                                           thrust = tbw_left_propulsion_outputs_plus_1_point_0g.thrust)
    
    system_model.register_output(tbw_range_outputs_plus_1_point_0g)

    # system_model.add_objective(tbw_range_outputs_plus_1_point_0g.range, scaler = -1)

    caddee_csdl_model = system_model.assemble_csdl()

    # Enforce beam symmetry

    if sizing:
        wing_beam_ttop = caddee_csdl_model.declare_variable('wing_beam_ttop', shape=(num_wing_beam, ))
        wing_beam_tbot = caddee_csdl_model.declare_variable('wing_beam_tbot', shape=(num_wing_beam, ))
        wing_beam_tweb = caddee_csdl_model.declare_variable('wing_beam_tweb', shape=(num_wing_beam, ))

        n = int((num_wing_beam - 1) / 2)

        left_ttop = wing_beam_ttop[0:n]
        left_tbot = wing_beam_tbot[0:n]
        left_tweb = wing_beam_tweb[0:n]

        right_ttop = caddee_csdl_model.create_output('right_ttop', shape=(n, ))
        right_tbot = caddee_csdl_model.create_output('right_tbot', shape=(n, ))
        right_tweb = caddee_csdl_model.create_output('right_tweb', shape=(n, ))
        indices = np.flip(np.arange(n+1, num_wing_beam, 1, dtype=int))
        for i in range(n):
            right_ttop[i] = wing_beam_ttop[int(indices[i])]
            right_tbot[i] = wing_beam_tbot[int(indices[i])]
            right_tweb[i] = wing_beam_tweb[int(indices[i])]


        caddee_csdl_model.register_output('ttop_symmetry', right_ttop-left_ttop)
        caddee_csdl_model.register_output('tbot_symmetry', right_tbot-left_tbot)
        caddee_csdl_model.register_output('tweb_symmetry', right_tweb-left_tweb)

        caddee_csdl_model.add_constraint('ttop_symmetry', equals=0, scaler=1e-2)
        caddee_csdl_model.add_constraint('tbot_symmetry', equals=0, scaler=1e-2)
        caddee_csdl_model.add_constraint('tweb_symmetry', equals=0, scaler=1e-2)


    sim = Simulator(caddee_csdl_model,name = 'beam_full_outputs_trial_3', analytics=True)
    sim['plus_2_point_5g_vlm_model.plus_2_point_5g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])
    sim['minus_1g_vlm_model.minus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])
    sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.evaluation_pt'] = np.array([0., 0., 2.8])


    sim.run()
    cd.print_caddee_outputs(system_model, sim, compact_print=True)

    print('accelerations', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.accelerations'])
    print('accelerations +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.accelerations'])
    print('accelerations -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.accelerations'])
    exit()

    prob = CSDLProblem(problem_name='who_knows', simulator=sim)
    optimizer = SLSQP(prob, maxiter=400, ftol=1E-5)
    optimizer.solve()
    optimizer.print_results()

    print('Wing Span', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_span_dv'])
    print('wing tip chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_tip_chord_left_dv'])
    print('wing root chord', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_root_chord_dv'])
    print('wing mid chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_mid_chord_left_dv'])
    print('Wing Fuselage Connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_fuselage_connection'])
    print('gearpod left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_left_strut_connection'])
    print('gearpod right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_right_strut_connection'])
    print('wing left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_left_strut_connection'])
    print('wing right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_right_strut_connection'])
    print('wing_twist_coefficients', sim[wing_twist_coefficients.name])
    print('wing_AR', sim['plus_1_point_0_g_tbw_area_model.wing_AR'])

    counter = 1

    # region 1.0g
    print('Total Forces: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['plus_1_point_0g_sizing_total_forces_moments_model.total_moments'])
    print('Total cL: ', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.accelerations'])
    print('tbw_cg', sim['TBW_Mass_Properties.cg_vector'])
    print('tbw mass', sim['TBW_Mass_Properties.mass'])
    print('tbw area fraction weights', sim['TBW_Mass_Properties.area_fraction_weights'])
    print('du_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.du_dt'])
    print('dv_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt', sim['plus_1_point_0g_sizing_eom_model.eom_solve_model.dw_dt'])
    print('tbw pitch angle', sim['pitch_angle_1_point_0g'])
    print('tbw elevator deflection', sim['plus_1_point_0_g_elevator'])
    print('VLM F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1g_vlm_model.F'])
    print('Inertial F', sim['plus_1_point_0g_sizing_inertial_loads_model.F_inertial'])
    print('tbw viscous force', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F', sim['plus_1_point_0g_sizing_total_forces_moments_model.plus_1_point_0g_tbw_prop_model.F'])
    print('throttle', sim['plus_1_point_0g_tbw_prop_model.throttle'])
    print('Lift over Drag', sim['plus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('VLM induced_drag_for_each_surface', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.induced_drag_for_each_surface'])
    print('VLM total_area_vlm_surfaces', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_area_vlm_surfaces_actual'])
    print('VLM check_constant', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.check_constant'])
    print('VLM cd_induced_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('VLM total_CD', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CD'])
    print('VLM total_drag', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_drag'])
    print('qBar_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar'])
    print('qBar_area_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.qBar_area'])
    print('area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('Cf_viscous', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous'])
    # print('Cf_viscous_1', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_viscous_1'])
    print('Cf_interference', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_interference'])
    print('Cf_wave', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wave'])
    print('cf_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'])
    print('total_cf_with_area_fraction', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf'] + sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.cd_induced_drag'])
    print('Cf_wo_area_fraction_total_viscous_drag_model', sim['plus_1_point_0g_tbw_viscous_drag_model.Cf_wo_area_fraction'])
    print('Wing Span', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_span_dv'])
    print('wing tip chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_tip_chord_left_dv'])
    print('wing root chord', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_root_chord_dv'])
    print('wing mid chord left', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_mid_chord_left_dv'])
    print('Wing Fuselage Connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_fuselage_connection'])
    print('gearpod left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_left_strut_connection'])
    print('gearpod right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.gearpod_right_strut_connection'])
    print('wing left strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_left_strut_connection'])
    print('wing right strut connection', sim['parameterization_solver_with_inputs_wing_span_dv_wing_tip_chord_left_dv_wing_root_chord_dv_wing_mid_chord_left_dv_wing_fuselage_connection_gearpod_left_strut_connection_gearpod_right_strut_connection_wing_left_strut_connection_wing_right_strut_connection_left_strut_left_jury_connection_right_strut_right_jury_connection_.wing_right_strut_connection'])
    print('wing_AR', sim['plus_1_point_0_g_tbw_area_model.wing_AR'])
    print('wing_AR_vlm', sim['plus_1g_vlm_model.plus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.wing_AR_vlm'])
    print('wing_twist_coefficients', sim[wing_twist_coefficients.name])
    # endregion
    print('/n new haaha_2.5 coming up')
    # region 2.5g
    print('Total Forces +2.5g: ', sim['plus_2_point_5g_sizing_total_forces_moments_model.total_forces'])
    print('Total Moments +2.5g: ', sim['plus_2_point_5g_sizing_total_forces_moments_model.total_moments'])
    print('Total cL +2.5g: ', sim['plus_2_point_5g_vlm_model.plus_2_point_5g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.accelerations'])
    print('tbw mass +2.5g', sim['TBW_Mass_Properties.mass'])
    print('beam mass +2.5g', sim['beam_mass_model.mass'])
    print('total mass +2.5g', sim['plus_2_point_5g_sizing_total_mass_properties_model.total_mass'])
    print('struct mass abc +2.5g', wing_beam_mass_props.mass)
    print('struct mass +2.5g', sim['eb_beam_2_point_5g.Aframe.comp_model.struct_mass'])
    print('du_dt +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.du_dt'])
    print('dv_dt +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.dw_dt'])
    print('dp_dt +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.dp_dt'])
    print('dq_dt +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.dq_dt'])
    print('dr_dt +2.5g', sim['plus_2_point_5g_sizing_eom_model.eom_solve_model.dr_dt'])
    print('tbw pitch angle +2.5g', sim['plus_2_point_5g_sizing.pitch_angle'])
    print('throttle +2.5g', sim['plus_2_point_5g_tbw_prop_model.throttle'])
    print('elevator deflection +2.5g', sim['plus_2_point_5g_elevator'])
    print('VLM F +2.5g', sim['plus_2_point_5g_sizing_total_forces_moments_model.plus_2_point_5g_vlm_model.F'])
    print('Inertial F +2.5g', sim['plus_2_point_5g_sizing_total_forces_moments_model.plus_2_point_5g_sizing_inertial_loads_model.F_inertial'])
    print('tbw viscous force +2.5g', sim['plus_2_point_5g_sizing_total_forces_moments_model.plus_2_point_5g_tbw_viscous_drag_model.F'])
    print('Propulsion F +2.5g', sim['plus_2_point_5g_sizing_total_forces_moments_model.plus_2_point_5g_tbw_prop_model.F'])
    # print('Lift over Drag +2.5g', sim['plus_2_point_5g_tbw_L_over_D_model.Lift_over_Drag'])
    print('area_fraction +2.5g', sim['plus_2_point_5g_tbw_viscous_drag_model.area_fraction'])
    print('wing stress +2.5g', plus_2_point_5g_eb_beam_outputs_wing.stresses[0])
    vm_stress_wing_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.wing_beam_stress']
    vm_stress_left_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.left_strut_beam_stress']
    vm_stress_right_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.right_strut_beam_stress']
    vm_stress_left_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.left_jury_beam_stress']
    vm_stress_right_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.right_jury_beam_stress']
    print('vm_stress_wing +2.5g', np.max(vm_stress_wing_2_point_5g, axis=1))
    spanwise_wing_max_stress_2_point_5g = np.max(vm_stress_wing_2_point_5g, axis=1)
    print('vm_stress_left_strut +2.5g', np.max(vm_stress_left_strut_2_point_5g, axis=1))
    print('vm_stress_right_strut +2.5g', np.max(vm_stress_right_strut_2_point_5g, axis=1))
    print('vm_stress_left_jury +2.5g', np.max(vm_stress_left_jury_2_point_5g, axis=1))
    print('vm_stress_right_jury +2.5g', np.max(vm_stress_right_jury_2_point_5g, axis=1))

    displ_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.wing_beam_displacement']
    print("Beam displacement wing(ft) +2.5g: ", displ_2_point_5g)
    print('Tip displacement wing(ft) +2.5g: ', displ_2_point_5g[-1, 2])
    spanwise_wing_z_disp_2_point_5g = displ_2_point_5g[:, 2]
    print('spanwise_wing_z_disp +2.5g',spanwise_wing_z_disp_2_point_5g)
    displ1_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.left_strut_beam_displacement']
    print("Beam displacement left strut(ft) +2.5g: ", displ1_2_point_5g)
    print('Tip displacement left strut(ft) +2.5g: ', displ1_2_point_5g[-1, 2])

    displ2_2_point_5g = sim['eb_beam_2_point_5g.Aframe.comp_model.right_strut_beam_displacement']
    print("Beam displacement right strut(ft) +2.5g: ", displ2_2_point_5g)
    print('Tip displacement right strut(ft) +2.5g: ', displ2_2_point_5g[-1, 2])


    print('wing_top_buckling', sim['eb_beam_2_point_5g.Aframe.comp_model.wing_beam_top_bkl'])
   # Thicknesses
    top_t_wing_2_point_5g = sim['eb_beam_2_point_5g.Aframe.wing_beam_ttop']
    bot_t_wing_2_point_5g = sim['eb_beam_2_point_5g.Aframe.wing_beam_tbot']
    web_t_wing_2_point_5g = sim['eb_beam_2_point_5g.Aframe.wing_beam_tweb']
    top_t_left_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.left_strut_beam_ttop']
    bot_t_left_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.left_strut_beam_tbot']
    web_t_left_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.left_strut_beam_tweb']
    top_t_right_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.right_strut_beam_ttop']
    bot_t_right_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.right_strut_beam_tbot']
    web_t_right_strut_2_point_5g = sim['eb_beam_2_point_5g.Aframe.right_strut_beam_tweb']
    top_t_left_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.left_jury_beam_ttop']
    bot_t_left_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.left_jury_beam_tbot']
    web_t_left_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.left_jury_beam_tweb']
    top_t_right_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.right_jury_beam_ttop']
    bot_t_right_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.right_jury_beam_tbot']
    web_t_right_jury_2_point_5g = sim['eb_beam_2_point_5g.Aframe.right_jury_beam_tweb']

    spanwise_wing_node_y_loc_2_point_5g = sim['eb_beam_2_point_5g.Aframe.wing_beam_mesh'][:, 1]
    spanwise_wing_z_force_2_point_5g = sim['eb_beam_force_map_plus_2_point_5_g.wing_beam_forces'][:, 2]
    spanwise_wing_width_2_point_5g = sim['eb_beam_2_point_5g.Aframe.wing_beam_width']
    spanwise_wing_height_2_point_5g = sim['eb_beam_2_point_5g.Aframe.wing_beam_height']
    sol_dict_wing_2_point_5g = {'Spanwise loc (ft)': spanwise_wing_node_y_loc_2_point_5g,
                    'Width (ft)': spanwise_wing_width_2_point_5g,
                    'Height (ft)': spanwise_wing_height_2_point_5g,
                    'Node z force (N)': spanwise_wing_z_force_2_point_5g,
                    'Displacement (ft)': spanwise_wing_z_disp_2_point_5g}
    sol_dict_wing_1_2_point_5g = {
                'Stress': spanwise_wing_max_stress_2_point_5g,}
    sol_dict_wing_2_2_point_5g = {
        'wing_beam_ttop':top_t_wing_2_point_5g,
        'wing_beam_tbot':bot_t_wing_2_point_5g,
        'wing_beam_tweb':web_t_wing_2_point_5g,
        }
    sol_dict_wing_3_2_point_5g = {
        'left_strut_beam_ttop':top_t_left_strut_2_point_5g,
        'left_strut_beam_tbot':bot_t_left_strut_2_point_5g,
        'left_strut_beam_tweb':web_t_left_strut_2_point_5g,
        'right_strut_beam_ttop':top_t_right_strut_2_point_5g,
        'right_strut_beam_tbot':bot_t_right_strut_2_point_5g,
        'right_strut_beam_tweb':web_t_right_strut_2_point_5g,
        }
    sol_dict_wing_4_2_point_5g = {
        'left_jury_beam_ttop':top_t_left_jury_2_point_5g,
        'left_jury_beam_tbot':bot_t_left_jury_2_point_5g,
        'left_jury_beam_tweb':web_t_left_jury_2_point_5g,
        'right_jury_beam_ttop':top_t_right_jury_2_point_5g,
        'right_jury_beam_tbot':bot_t_right_jury_2_point_5g,
        'right_jury_beam_tweb':web_t_right_jury_2_point_5g,
        }


    nodal_sol_df_wing_2_point_5g = pd.DataFrame(data=sol_dict_wing_2_point_5g)
    nodal_sol_df_wing_2_point_5g.to_excel(f'{counter}_sol_dict_2_point_5g_wing_wing_strut_jury_{np.rad2deg(theta_2_point_5g.value)}deg.xlsx')
    #print(nodal_sol_df_wing)
    nodal_sol_df_wing_1_2_point_5g = pd.DataFrame(data=sol_dict_wing_1_2_point_5g)
    nodal_sol_df_wing_1_2_point_5g.to_excel(f'{counter}_sol_dict_2_point_5g_wing_1_wing_strut_jury_{np.rad2deg(theta_2_point_5g.value)}deg.xlsx')   
    #print(nodal_sol_df_wing_1)  
    nodal_sol_df_wing_2_2_point_5g = pd.DataFrame(data=sol_dict_wing_2_2_point_5g)
    nodal_sol_df_wing_2_2_point_5g.to_excel(f'{counter}_sol_dict_2_point_5g_wing_2_wing_strut_jury_{np.rad2deg(theta_2_point_5g.value)}deg.xlsx') 
    nodal_sol_df_wing_3_2_point_5g = pd.DataFrame(data=sol_dict_wing_3_2_point_5g)
    nodal_sol_df_wing_3_2_point_5g.to_excel(f'{counter}_sol_dict_2_point_5g_wing_3_wing_strut_jury_{np.rad2deg(theta_2_point_5g.value)}deg.xlsx') 
    nodal_sol_df_wing_4_2_point_5g = pd.DataFrame(data=sol_dict_wing_4_2_point_5g)
    nodal_sol_df_wing_4_2_point_5g.to_excel(f'{counter}_sol_dict_2_point_5g_wing_4_wing_strut_jury_{np.rad2deg(theta_2_point_5g.value)}deg.xlsx') 
    # endregion
    print('/n new haaha -1.0 coming up')
    # region -1.0g
    print('Total Forces -1.0g: ', sim['minus_1_point_0_g_sizing_total_forces_moments_model.total_forces'])
    print('Total Moments -1.0g: ', sim['minus_1_point_0_g_sizing_total_forces_moments_model.total_moments'])
    print('Total cL -1.0g: ', sim['minus_1g_vlm_model.minus_1g_vlm_model.VLMSolverModel.VLM_outputs.LiftDrag.total_CL'])
    print('accelerations -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.accelerations'])
    print('tbw mass -1.0g', sim['TBW_Mass_Properties.mass'])
    print('beam mass -1.0g', sim['beam_mass_model.mass'])
    print('total mass -1.0g', sim['minus_1_point_0_g_sizing_total_mass_properties_model.total_mass'])
    print('struct mass abc', wing_beam_mass_props.mass)
    print('struct mass', sim['eb_beam_minus_1_point_0g.Aframe.comp_model.struct_mass'])
    print('du_dt -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.du_dt'])
    print('dv_dt -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.dv_dt'])
    print('dw_dt -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.dw_dt'])
    print('dp_dt -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.dp_dt'])
    print('dq_dt -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.dq_dt'])
    print('dr_dt -1.0g', sim['minus_1_point_0_g_sizing_eom_model.eom_solve_model.dr_dt'])
    print('tbw pitch angle -1.0g', sim['pitch_angle_minus_1_point_0_g'])
    print('throttle -1.0g', sim['minus_1_point_0g_tbw_prop_model.throttle'])
    print('elevator deflection', sim['minus_1_point_0_g_elevator'])
    print('VLM F -1.0g', sim['minus_1_point_0_g_sizing_total_forces_moments_model.minus_1g_vlm_model.F'])
    print('Inertial F -1.0g', sim['minus_1_point_0_g_sizing_inertial_loads_model.F_inertial'])
    print('tbw viscous force -1.0g', sim['minus_1_point_0g_tbw_viscous_drag_model.F'])
    print('Propulsion F -1.0g', sim['minus_1_point_0_g_sizing_total_forces_moments_model.minus_1_point_0g_tbw_prop_model.F'])
    print('Lift over Drag -1.0g', sim['minus_1_point_0g_tbw_L_over_D_model.Lift_over_Drag'])
    print('area_fraction -1.0g', sim['minus_1_point_0g_tbw_viscous_drag_model.area_fraction'])
    print('wing stress -1.0g', minus_1_point_0g_eb_beam_outputs_wing.stresses[0])
    vm_stress_wing_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.wing_beam_stress']
    vm_stress_left_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.left_strut_beam_stress']
    vm_stress_right_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.right_strut_beam_stress']
    vm_stress_left_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.left_jury_beam_stress']
    vm_stress_right_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.right_jury_beam_stress']
    print('vm_stress_wing -1.0g', np.max(vm_stress_wing_minus_1_point_0g, axis=1))
    spanwise_wing_max_stress_minus_1_point_0g = np.max(vm_stress_wing_minus_1_point_0g, axis=1)
    print('vm_stress_left_strut -1.0g', np.max(vm_stress_left_strut_minus_1_point_0g, axis=1))
    print('vm_stress_right_strut -1.0g', np.max(vm_stress_right_strut_minus_1_point_0g, axis=1))
    print('vm_stress_left_jury -1.0g', np.max(vm_stress_left_jury_minus_1_point_0g, axis=1))
    print('vm_stress_right_jury -1.0g', np.max(vm_stress_right_jury_minus_1_point_0g, axis=1))

    # Displacement 

    displ_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.wing_beam_displacement']
    print("Beam displacement wing(ft) -1.0g: ", displ_minus_1_point_0g)
    print('Tip displacement wing(ft) -1.0g: ', displ_minus_1_point_0g[-1, 2])
    spanwise_wing_z_disp_minus_1_point_0g = displ_minus_1_point_0g[:, 2]
    print('spanwise_wing_z_disp -1.0g',spanwise_wing_z_disp_minus_1_point_0g)
    displ1_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.left_strut_beam_displacement']
    print("Beam displacement left strut(ft) -1.0g: ", displ1_minus_1_point_0g)
    print('Tip displacement left strut(ft) -1.0g: ', displ1_minus_1_point_0g[-1, 2])

    displ2_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.comp_model.right_strut_beam_displacement']
    print("Beam displacement right strut(ft) -1.0g: ", displ2_minus_1_point_0g)
    print('Tip displacement right strut(ft) -1.0g: ', displ2_minus_1_point_0g[-1, 2])

    print('wing_top_buckling', sim['eb_beam_minus_1_point_0g.Aframe.comp_model.wing_beam_top_bkl'])
    print('wing_bot_buckling', sim['eb_beam_minus_1_point_0g.Aframe.comp_model.wing_beam_bot_bkl'])
    # Thickness
    top_t_wing_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.wing_beam_ttop']
    bot_t_wing_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.wing_beam_tbot']
    web_t_wing_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.wing_beam_tweb']
    top_t_left_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.left_strut_beam_ttop']
    bot_t_left_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.left_strut_beam_tbot']
    web_t_left_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.left_strut_beam_tweb']
    top_t_right_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.right_strut_beam_ttop']
    bot_t_right_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.right_strut_beam_tbot']
    web_t_right_strut_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.right_strut_beam_tweb']
    top_t_left_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.left_jury_beam_ttop']
    bot_t_left_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.left_jury_beam_tbot']
    web_t_left_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.left_jury_beam_tweb']
    top_t_right_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.right_jury_beam_ttop']
    bot_t_right_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.right_jury_beam_tbot']
    web_t_right_jury_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.right_jury_beam_tweb']

    spanwise_wing_node_y_loc_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.wing_beam_mesh'][:, 1]
    spanwise_wing_z_force_minus_1_point_0g = sim['eb_beam_force_map_minus_1_point_0g.wing_beam_forces'][:, 2]
    spanwise_wing_width_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.wing_beam_width']
    spanwise_wing_height_minus_1_point_0g = sim['eb_beam_minus_1_point_0g.Aframe.wing_beam_height']
    sol_dict_wing_minus_1_point_0g = {'Spanwise loc (ft)': spanwise_wing_node_y_loc_minus_1_point_0g,
                'Width (ft)': spanwise_wing_width_minus_1_point_0g,
                'Height (ft)': spanwise_wing_height_minus_1_point_0g,
                'Node z force (N)': spanwise_wing_z_force_minus_1_point_0g,
                'Displacement (ft)': spanwise_wing_z_disp_minus_1_point_0g}
    sol_dict_wing_1_minus_1_point_0g = {
            'Stress': spanwise_wing_max_stress_minus_1_point_0g,}
    sol_dict_wing_2_minus_1_point_0g = {
    'wing_beam_ttop':top_t_wing_minus_1_point_0g,
    'wing_beam_tbot':bot_t_wing_minus_1_point_0g,
    'wing_beam_tweb':web_t_wing_minus_1_point_0g,
    }
    sol_dict_wing_3_minus_1_point_0g = {
    'left_strut_beam_ttop':top_t_left_strut_minus_1_point_0g,
    'left_strut_beam_tbot':bot_t_left_strut_minus_1_point_0g,
    'left_strut_beam_tweb':web_t_left_strut_minus_1_point_0g,
    'right_strut_beam_ttop':top_t_right_strut_minus_1_point_0g,
    'right_strut_beam_tbot':bot_t_right_strut_minus_1_point_0g,
    'right_strut_beam_tweb':web_t_right_strut_minus_1_point_0g,
    }
    sol_dict_wing_4_minus_1_point_0g = {
    'left_jury_beam_ttop':top_t_left_jury_minus_1_point_0g,
    'left_jury_beam_tbot':bot_t_left_jury_minus_1_point_0g,
    'left_jury_beam_tweb':web_t_left_jury_minus_1_point_0g,
    'right_jury_beam_ttop':top_t_right_jury_minus_1_point_0g,
    'right_jury_beam_tbot':bot_t_right_jury_minus_1_point_0g,
    'right_jury_beam_tweb':web_t_right_jury_minus_1_point_0g,
    }
    counter = 1111
    
    nodal_sol_df_wing_minus_1_point_0g = pd.DataFrame(data=sol_dict_wing_minus_1_point_0g)
    nodal_sol_df_wing_minus_1_point_0g.to_excel(f'{counter}_sol_dict_minus_1_point_0g_wing_wing_strut_jury_{np.rad2deg(theta_minus_1_point_0_g.value)}deg.xlsx')
    #print(nodal_sol_df_wing)
    nodal_sol_df_wing_1_minus_1_point_0g = pd.DataFrame(data=sol_dict_wing_1_minus_1_point_0g)
    nodal_sol_df_wing_1_minus_1_point_0g.to_excel(f'{counter}_sol_dict_minus_1_point_0g_wing_1_wing_strut_jury_{np.rad2deg(theta_minus_1_point_0_g.value)}deg.xlsx')   
    #print(nodal_sol_df_wing_1)  
    nodal_sol_df_wing_2_minus_1_point_0g = pd.DataFrame(data=sol_dict_wing_2_minus_1_point_0g)
    nodal_sol_df_wing_2_minus_1_point_0g.to_excel(f'{counter}_sol_dict_minus_1_point_0g_wing_2_wing_strut_jury_{np.rad2deg(theta_minus_1_point_0_g.value)}deg.xlsx') 
    nodal_sol_df_wing_3_minus_1_point_0g = pd.DataFrame(data=sol_dict_wing_3_minus_1_point_0g)
    nodal_sol_df_wing_3_minus_1_point_0g.to_excel(f'{counter}_sol_dict_minus_1_point_0g_wing_3_wing_strut_jury_{np.rad2deg(theta_minus_1_point_0_g.value)}deg.xlsx') 
    nodal_sol_df_wing_4_minus_1_point_0g = pd.DataFrame(data=sol_dict_wing_4_minus_1_point_0g)
    nodal_sol_df_wing_4_minus_1_point_0g.to_excel(f'{counter}_sol_dict_minus_1_point_0g_wing_4_wing_strut_jury_{np.rad2deg(theta_minus_1_point_0_g.value)}deg.xlsx') 
    # endregion
    print('/n range equation')
    print('total_fuel_burned',sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_range_model.total_fuel_burned'])
    print('final_weight',sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_range_model.final_weight'])
    print('range_cruise_m3l',sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_range_model.range_cruise_m3l'])
    print('range_aaa',sim['plus_1_point_0g_and_2_point_5g_and_minus_1_point_0g_tbw_range_model.range_aaa'])
