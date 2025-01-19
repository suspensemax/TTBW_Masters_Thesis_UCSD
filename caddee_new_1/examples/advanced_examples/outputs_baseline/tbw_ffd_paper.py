"TBW baseline fixed wing and span location geometry working? backup"

# import numpy as np
import numpy as np
import caddee.api as cd 
import lsdo_geo as lg
import m3l
from caddee import GEOMETRY_FILES_FOLDER
from caddee.utils.helper_functions.geometry_helpers import make_rotor_mesh, make_vlm_camber_mesh, make_1d_box_beam_mesh, compute_component_surface_area, BladeParameters
from caddee.utils.aircraft_models.drag_models.drag_build_up import DragComponent
import lsdo_geo.splines.b_splines as bsp
import gc
gc.enable()
from dataclasses import dataclass, field
import csdl
import math

# @dataclass
# class geometryOutputs: 
#     wing_AR : m3l.Variable
#     # S_ref : m3l.Variable 
#     wing_area : m3l.Variable
#     # wing_span : m3l.Variable

# region True or False
geometry_plot = False
components_plot = False
do_plots = False
print_coordinates = False
plot_htail_mesh = False
plot_left_strut_mesh = False
plot_right_strut_mesh = False

# region Plots 
# VLM 
mesh_flag_wing = False
mesh_flag_wing_oml = False
mesh_flag_htail = False
mesh_flag_htail_oml = False
mesh_flag_left_strut = False
mesh_flag_left_strut_oml = False
mesh_flag_right_strut = False
mesh_flag_right_strut_oml = False
mesh_flag_vtail_helper = False
mesh_flag_fuselage = False
mesh_flag_left_jury_helper = False
mesh_flag_right_jury_helper = False
mesh_flag_gearpod = False
plot_unupdated_vlm_mesh = False
plot_updated_vlm_mesh = False
# beam
wing_beam_mesh_flag = False
left_strut_beam_mesh_flag = False
right_strut_beam_mesh_flag = False
left_jury_beam_mesh_flag = False
right_jury_beam_mesh_flag = False
# endregion

# endregion

# Instantiate system model
system_model = m3l.Model()

# Importing and refitting the geometrywing_oml_mesh
geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_try.stp', parallelize=True)
geometry.refit(parallelize=True, order=(4, 4))

if geometry_plot:
    geometry.plot()

plot_meshes = False
# geometry_dv = True

# region Declaring Components
wing = geometry.declare_component(component_name='wing', b_spline_search_names=['Wing'])
if components_plot:
    wing.plot()
htail = geometry.declare_component(component_name='htail', b_spline_search_names=['Htail'])
if components_plot:
    htail.plot()
strut = geometry.declare_component(component_name='strut', b_spline_search_names=['Strut'])
if components_plot:
    strut.plot()
strut_right = geometry.declare_component(component_name='strut_right', b_spline_search_names=['Strut, 0'])
if components_plot:
    strut_right.plot()
strut_left = geometry.declare_component(component_name='strut_left', b_spline_search_names=['Strut, 1'])
if components_plot:
    strut_left.plot()
jury = geometry.declare_component(component_name='jury', b_spline_search_names=['Jury'])
if components_plot:
    jury.plot()
jury_right = geometry.declare_component(component_name='jury_right', b_spline_search_names=['Jury, 0'])
if components_plot:
    jury_right.plot()
jury_left = geometry.declare_component(component_name='jury_left', b_spline_search_names=['Jury, 1'])
if components_plot:
    jury_left.plot()
vtail = geometry.declare_component(component_name='vtail', b_spline_search_names=['Vtail'])
if components_plot:
    vtail.plot()
Fuselage = geometry.declare_component(component_name='Fuselage', b_spline_search_names=['Fuselage'])
if components_plot:
    Fuselage.plot()
gearpod = geometry.declare_component(component_name='gearpod', b_spline_search_names=['Gear Pod'])
if components_plot:
    gearpod.plot()
wing_strut_jury = geometry.declare_component(component_name='wing_strut_jury', b_spline_search_names=['Wing', 'Strut', 'Jury'])
components_plot = True
if components_plot:
    wing_strut_jury.plot()



# wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2), order=(2, 2, 2))
wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 2, 2), order=(2, 2, 2))
left_strut_ffd_block = lg.construct_ffd_block_around_entities(name='left_strut_ffd_block', entities=strut_left, num_coefficients=(2, 5, 2), order=(2, 2, 2))
right_strut_ffd_block = lg.construct_ffd_block_around_entities(name='right_strut_ffd_block', entities=strut_right, num_coefficients=(2, 5, 2), order=(2, 2, 2))
wing
wing_ffd_block.plot
exit()
# endregion

# region geometry dv

# geometry_dv = False
geometry_dv = True

wing_span_dv = system_model.create_input(name='wing_span_dv', shape = (1,), val = 170, dv_flag = geometry_dv, lower = 140., upper = 200., scaler= 1.e-1)
# wing_span_dv = system_model.create_input(name='wing_span_dv', shape = (1,), val = 171, dv_flag = geometry_dv, lower = 140., upper = 200., scaler =1.e-2)

# geometry_dv = False
# geometry_dv = True
# wing_tip_chord_left_dv = system_model.create_input(name='wing_tip_chord_left_dv', shape= (1,), val = 10.7, dv_flag = geometry_dv, lower=2.37, upper=5.13)
wing_tip_chord_left_dv = system_model.create_input(name='wing_tip_chord_left_dv', shape= (1,), val = 3.8, dv_flag = geometry_dv, lower=3.37, upper=4.13)
# wing_tip_chord_right_dv = system_model.create_input(name='wing_tip_chord_right_dv', shape= (1,), val = 3.8, dv_flag = geometry_dv, lower=3.37, upper=4.13, scaler =1.e-1)
wing_root_chord_dv = system_model.create_input(name='wing_root_chord_dv', shape= (1,), val = 10.7, dv_flag = geometry_dv, lower=9.64, upper=11.79)
# wing_root_chord_dv = system_model.create_input(name='wing_root_chord_dv', shape= (1,), val = 9.64, dv_flag = geometry_dv, lower=9.64, upper=12.79)
# wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 11.68, dv_flag = geometry_dv, lower = 7.70, upper = 11.68)
wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 9.5, dv_flag = geometry_dv, lower = 8.70, upper = 10.68)
# wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 8.7, dv_flag = geometry_dv, lower = 8.70, upper = 10.68)
# wing_mid_chord_right_dv = system_model.create_input(name = 'wing_mid_chord_right_dv', shape=(1,), val= 9.5, dv_flag = geometry_dv, lower = 8.70, upper = 10.68, scaler =1.e-1)

# left_strut_chord_dv = system_model.create_input(name='left_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)
# right_strut_chord_dv = system_model.create_input(name='right_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)
# left_jury_chord_dv = system_model.create_input(name='left_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)
# right_jury_chord_dv = system_model.create_input(name='right_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)

# endregion

# region mesh

# region Wing Mesh

num_spanwise_wing_vlm = 25
num_chordwise_wing_vlm = 7 

point00_wing = np.array([68.035, 85.291, 4.704]) # * ft2m # Right tip leading edge 
wing_le_right = point00_wing
point01_wing = np.array([71.790, 85.291, 4.704]) # * ft2m # Right tip trailing edge
wing_te_right = point01_wing
point10_wing = np.array([57.575, 49.280, 5.647]) # * ft2m # Right leading edge 1
point11_wing = np.array([67.088, 49.280, 5.507]) # * ft2m # Right trailing edge 1
point20_wing = np.array([56.172, 42.646, 5.821]) # * ft2m # Right leading edge 2
point21_wing = np.array([65.891, 42.646, 5.651]) # * ft2m # Right trailing edge 2
point30_wing = np.array([47.231, 0.000, 6.937]) # * ft2m # Center Leading Edge
wing_le_center = point30_wing
point31_wing = np.array([57.953, 0.000, 6.566]) # * ft2m # Center Trailing edge
wing_te_center = point31_wing
point40_wing = np.array([56.172, -42.646, 5.821]) # * ft2m # Left leading edge 2
point41_wing = np.array([65.891, -42.646, 5.651]) # * ft2m # Left trailing edge 2
point50_wing = np.array([57.575, -49.280, 5.647]) # * ft2m # Left leading edge 1
point51_wing = np.array([67.088, -49.280, 5.507]) # * ft2m # Left trailing edge 1
point60_wing = np.array([68.035, -85.291, 4.704]) # * ft2m # Left tip leading edge 
wing_le_left = point60_wing
point61_wing = np.array([71.790, -85.291, 4.704]) # * ft2m # Left tip trailing edge
wing_te_left = point61_wing

wing_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=wing,
    num_spanwise=num_spanwise_wing_vlm,
    num_chordwise=num_chordwise_wing_vlm,
    te_right=np.array([73., 85., 5.507]),
    te_left=np.array([73., -85., 5.507]),
    te_center=np.array([57.953, 0.000, 6.566]),
    le_left=np.array([66., -85., 5.647]),
    le_right=np.array([66., 85., 5.647]),
    le_center=np.array([47.231, 0.000, 6.937]),
    grid_search_density_parameter=100,
    le_interp='linear',
    te_interp='linear',
    off_set_x=0.2,
    bunching_cos=True,
    plot=mesh_flag_wing,
    mirror=True,
)

# endregion

# region htail mesh

num_spanwise_htail_vlm = 21
num_chordwise_htail_vlm = 5

num_spanwise_vlm_htail = 9
num_chordwise_vlm_htail = 4

point00_htail = np.array([132.002-10.0, 19.217+4.5, 18.993+3.5]) # * ft2m # Right tip leading edge
point01_htail = np.array([135.993, 19.217, 18.993]) # * ft2m # Right tip trailing edge
point10_htail = np.array([122.905, 0.000, 20.000]) # * ft2m # Center Leading Edge
point11_htail = np.array([134.308, 0.000, 20.000]) # * ft2m # Center Trailing edge
point20_htail = np.array([132.002-10, -19.217-4.5, 18.993+3.5]) # * ft2m # Left tip leading edge
point21_htail = np.array([135.993, -19.217, 18.993]) # * ft2m # Left tip trailing edge

htail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=htail,
    # num_spanwise=num_spanwise_htail_vlm,
    num_spanwise=num_spanwise_vlm_htail,
    # num_chordwise=num_chordwise_htail_vlm,
    num_chordwise=num_chordwise_vlm_htail,
    te_right=point01_htail,
    te_left=point21_htail,
    te_center=point11_htail,
    le_left=point20_htail,
    le_right=point00_htail,
    le_center=point10_htail,
    grid_search_density_parameter=100,
    plot=mesh_flag_htail,
    mirror=True,
)

# endregion

# region strut mesh

# region left strut mesh 

num_spanwise_strut_vlm = 21
num_chordwise_strut_vlm = 5 

vertex00_left_strut = np.array([55.573, -12.641, -4.200]) # left leading 1
vertex01_left_strut = np.array([55.034, -16.277, -3.204]) # left leading 2
vertex02_left_strut = np.array([56.422, -25.365, -0.713]) # left leading 3
vertex03_left_strut = np.array([58.313, -30.818, 0.782]) # left leading 4
vertex04_left_strut = np.array([58.089, -36.271, 2.276]) # left leading 5
vertex05_left_strut = np.array([59.477, -45.359, 4.767]) # left leading 6
vertex06_left_strut = np.array([61.090, -48.994, 5.763]) # left leading 7

vertex10_left_strut = np.array([57.309, -12.641, -4.200]) # left trailing 1
vertex11_left_strut = np.array([58.959, -16.277, -3.204]) # left trailing 2
vertex12_left_strut = np.array([60.348, -25.365, -0.713]) # left trailing 3
vertex13_left_strut = np.array([60.124, -30.818, 0.782]) # left trailing 4
vertex14_left_strut = np.array([62.014, -36.271, 2.276]) # left trailing 5
vertex15_left_strut = np.array([63.403, -45.359, 4.767]) # left trailing 6                
vertex16_left_strut = np.array([62.902, -48.994, 5.763]) # left trailing 7

left_strut_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=strut_left,
    num_spanwise=num_spanwise_strut_vlm,
    num_chordwise=num_chordwise_strut_vlm,
    te_right=vertex10_left_strut,
    # te_right=np.array([57.,-13.,-4.2]),
    te_left=vertex16_left_strut,
    # te_left=np.array([63., -49., 5.763]),
    # te_center=vertex13_left_strut,
    # te_center=np.array([60., -30.9, 0.782]),
    le_left=vertex06_left_strut,
    # le_left=np.array([61., -49., 5.763]),
    le_right=vertex00_left_strut,
    # le_right=np.array([55., -13., -4.200]),
    # le_center=vertex03_left_strut,
    # le_center=np.array([58., -30.8, 0.782]),
    grid_search_density_parameter=100,
    plot=mesh_flag_left_strut,
    mirror=False,
)

# endregion
  
# region right strut mesh

num_spanwise_strut_vlm = 21
num_chordwise_strut_vlm = 5 
vertex00_right_strut = np.array([55.573, 12.641, -4.200]) # right leading 1
vertex01_right_strut = np.array([55.034, 16.277, -3.204]) # right leading 2
vertex02_right_strut = np.array([56.422, 25.365, -0.713]) # right leading 3
vertex03_right_strut = np.array([58.313, 30.818, 0.782]) # right leading 4
vertex04_right_strut = np.array([58.089, 36.271, 2.276]) # right leading 5
vertex05_right_strut = np.array([59.477, 45.359, 4.767]) # right leading 6
vertex06_right_strut = np.array([61.090, 48.994, 5.763]) # right leading 7

vertex10_right_strut = np.array([57.309, 12.641, -4.200]) # right trailing 1
vertex11_right_strut = np.array([58.959, 16.277, -3.204]) # right trailing 2
vertex12_right_strut = np.array([60.348, 25.365, -0.713]) # right trailing 3
vertex13_right_strut = np.array([60.124, 30.818, 0.782]) # right trailing 4
vertex14_right_strut = np.array([62.014, 36.271, 2.276]) # right trailing 5
vertex15_right_strut = np.array([63.403, 45.359, 4.767]) # right trailing 6                
vertex16_right_strut = np.array([62.902, 48.994, 5.763]) # right trailing 7

right_strut_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=strut_right,
    num_spanwise=num_spanwise_strut_vlm,
    num_chordwise=num_chordwise_strut_vlm,
    te_right=vertex10_right_strut,
    # te_right=np.array([57.,-13.,-4.2]),
    te_left=vertex16_right_strut,
    # te_left=np.array([63., -49., 5.763]),
    # te_center=vertex13_right_strut,
    # te_center=np.array([60., -30.9, 0.782]),
    le_left=vertex06_right_strut,
    # le_left=np.array([61., -49., 5.763]),
    le_right=vertex00_right_strut,
    # le_right=np.array([55., -13., -4.200]),
    # le_center=vertex03_right_strut,
    # le_center=np.array([58., -30.8, 0.782]),
    grid_search_density_parameter=100,
    plot=mesh_flag_right_strut,
    mirror=False,
)

# endregion

# endregion

# region vtail mesh 
num_spanwise_vlm_vtail = 8
num_chordwise_vlm_vtail = 6
vtail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=vtail,
    num_spanwise=num_spanwise_vlm_vtail,
    num_chordwise=num_chordwise_vlm_vtail,
    le_left=np.array([105.343, 0., 3.5,]), 
    le_right=np.array([120.341, 0., 20.754,]),
    te_left=np.array([122.596, 0., 3.5,]),
    te_right=np.array([137.595, 0., 20.754,]),
    plot=mesh_flag_vtail_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# region jury mesh 

# region left jury mesh     
num_spanwise_vlm_jury = 5
num_chordwise_vlm_jury = 3
left_jury_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=jury_left,
    num_spanwise=num_spanwise_vlm_jury,
    num_chordwise=num_chordwise_vlm_jury,
    le_left=np.array([58.721, -30.818, 0.732,]), 
    le_right=np.array([58.052, -30.818, 6.181]),
    te_left=np.array([59.676, -30.818, 0.732]),
    te_right=np.array([59.237, -30.818, 6.181]),
    plot=mesh_flag_left_jury_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# region right jury mesh     
num_spanwise_vlm_jury = 5
num_chordwise_vlm_jury = 3
right_jury_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=jury_right,
    num_spanwise=num_spanwise_vlm_jury,
    num_chordwise=num_chordwise_vlm_jury,
    le_left=np.array([58.721, 30.818, 0.732,]), 
    le_right=np.array([58.052, 30.818, 6.181]),
    te_left=np.array([59.676, 30.818, 0.732]),
    te_right=np.array([59.237, 30.818, 6.181]),
    plot=mesh_flag_right_jury_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# endregion

FFD_working_old_5 = True # span and chord and twist working

if FFD_working_old_5: 
    
    # region True of False
    wing_plot = False
    htail_plot = False
    left_strut_plot = False
    right_strut_plot = False
    left_jury_plot = False
    right_jury_plot = False
    geometry_flag = False
    gearpod_plot = False

    # endregion

    # region Evaluations 

    wing_te_right_project = wing.project(wing_te_right)
    wing_te_left_project = wing.project(wing_te_left)
    wing_te_center_project = wing.project(wing_te_center)
    wing_le_left_project = wing.project(wing_le_left)
    wing_le_right_project = wing.project(wing_le_right)
    wing_le_center_project = wing.project(wing_le_center)
    wing_a_project = wing.project(point40_wing)
    wing_b_project = wing.project(point41_wing)
    wing_c_project = wing.project(point20_wing)
    wing_d_project = wing.project(point21_wing)

    left_strut_1 = strut_left.project(vertex06_left_strut)
    left_strut_2 = strut_left.project(vertex00_left_strut)
    left_strut_3 = strut_left.project(vertex16_left_strut)

    right_strut_1 = strut_right.project(vertex06_right_strut)
    right_strut_2 = strut_right.project(vertex00_right_strut)
    right_strut_3 = strut_right.project(vertex16_right_strut)

    left_jury_le_left=np.array([58.721, -30.818, 0.732,]) 
    left_jury_le_right=np.array([58.052, -30.818, 6.181])
    left_jury_bot = jury_left.project(left_jury_le_left)
    left_jury_top = jury_left.project(left_jury_le_right)

    right_jury_le_left=np.array([58.721, 30.818, 0.732,]) 
    right_jury_le_right=np.array([58.052, 30.818, 6.181])
    right_jury_bot = jury_right.project(right_jury_le_left)
    right_jury_top = jury_right.project(right_jury_le_right)

    #wing and fuselage connection
    wing_fuselage_connection_point = np.array([52.548, 0., 6.268])
    # fusleage_projection_on_wing = Fuselage.project(wing_le_center)
    # wing_projection_on_fuselage = wing.project(wing_le_center)
    fusleage_projection_on_wing = Fuselage.project(wing_fuselage_connection_point)
    wing_projection_on_fuselage = wing.project(wing_fuselage_connection_point)

    # wing and strut connection
    # left -ve y
    # wing_left_strut_connection_point = np.array([61.661, -48.994, 5.927])
    wing_left_strut_connection_point = np.array([62.309, -49.280, 6.095])
    wing_projection_on_left_strut = wing.project(wing_left_strut_connection_point)
    left_strut_projection_on_wing = strut_left.project(wing_left_strut_connection_point)
    #right +ve y
    # wing_right_strut_connection_point = np.array([61.661, 48.994, 5.927])
    wing_right_strut_connection_point = np.array([62.309, 49.280, 6.095])
    wing_projection_on_right_strut = wing.project(wing_right_strut_connection_point)
    right_strut_projection_on_wing = strut_right.project(wing_right_strut_connection_point)

    #jury and wing connection
    # left
    wing_left_jury_connection_point = np.array([58.633, -30.818, 6.181])
    wing_projection_on_left_jury = wing.project(wing_left_jury_connection_point)
    left_jury_projection_on_wing = jury_left.project(wing_left_jury_connection_point)
    # right 
    wing_right_jury_connection_point = np.array([58.633, 30.818, 6.181])
    wing_projection_on_right_jury = wing.project(wing_right_jury_connection_point)
    right_jury_projection_on_wing = jury_right.project(wing_right_jury_connection_point)

    # jury and strut connection
    # left
    left_jury_left_strut_connection_point = np.array([59.190, -30.818, 0.732])
    left_jury_projection_on_left_strut = jury_left.project(left_jury_left_strut_connection_point)
    left_strut_projection_on_left_jury = strut_left.project(left_jury_left_strut_connection_point)
    #right
    right_jury_right_strut_connection_point = np.array([59.190, 30.818, 0.732])
    right_jury_projection_on_right_strut = jury_right.project(right_jury_right_strut_connection_point)
    right_strut_projection_on_right_jury = strut_right.project(right_jury_right_strut_connection_point)

    # gearpod and strut
    # left
    # gearpod_left_strut_connection_point = np.array([55.573, -12.641, -4.200])
    gearpod_left_strut_connection_point = np.array([56.425, -12.641, -4.200])
    gearpod_projection_on_left_strut = gearpod.project(gearpod_left_strut_connection_point)
    # left_strut_projection_on_gearpod = strut_left.project(gearpod_left_strut_connection_point)
    left_strut_projection_on_gearpod = strut.project(gearpod_left_strut_connection_point)
    # right
    gearpod_right_strut_connection_point = np.array([56.425, 12.641, -4.200])
    gearpod_projection_on_right_strut = gearpod.project(gearpod_right_strut_connection_point)
    # right_strut_projection_on_gearpod = strut_right.project(gearpod_right_strut_connection_point)
    right_strut_projection_on_gearpod = strut.project(gearpod_right_strut_connection_point)

    # endregion

    constant_b_spline_curve_1_dof_space =  bsp.BSplineSpace(name='constant_b_spline_curve_1_dof_space', order=1, parametric_coefficients_shape=(1,))
    linear_b_spline_curve_2_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_2_dof_space', order=2, parametric_coefficients_shape=(2,))
    linear_b_spline_curve_3_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_3_dof_space', order=2, parametric_coefficients_shape=(3,))
    cubic_b_spline_curve_5_dof_space = bsp.BSplineSpace(name='cubic_b_spline_curve_5_dof_space', order=4, parametric_coefficients_shape=(5,))
    from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
    from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
    tbw_param_solver = ParameterizationSolver()

    # wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2), order=(2, 2, 2))
    wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing_strut_jury, num_coefficients=(2, 11, 2), order=(2, 2, 2))
    exit()
    wing_ffd_block_sect_param = VolumeSectionalParameterization(name='wing_ffd_sectional_parameterization', 
                                                                principal_parametric_dimension=1, 
                                                                parameterized_points=wing_ffd_block.coefficients,
                                                                parameterized_points_shape=wing_ffd_block.coefficients_shape)
    

    wing_ffd_block_sect_param.add_sectional_translation(name='wing_span_stretch', axis=1)
    wing_ffd_block_sect_param.add_sectional_stretch(name='wing_chord_stretch', axis=0)

    # wing_span_stretch_coefficients = m3l.Variable(name= 'wing_span_stretch_coefficients', shape=(2, ), value=np.array([0.,0.]))
    wing_span_stretch_coefficients = system_model.create_input(name= 'wing_span_stretch_coefficients', shape=(2, ), val=np.array([0.,0.]))
    wing_span_strech_b_spline = bsp.BSpline(name='wing_span_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=wing_span_stretch_coefficients, num_physical_dimensions=1)
        
    # wing_chord_stretch_coefficients = m3l.Variable(name= 'wing_chord_stretch_coefficients', shape=(5, ), value=np.array([0., 0., 0., 0., 0.]))
    wing_chord_stretch_coefficients = system_model.create_input(name= 'wing_chord_stretch_coefficients', shape=(5, ), val=np.array([0., 0., 0., 0., 0.]))
    wing_chord_stretch_b_spline = bsp.BSpline(name='wing_chord_b_spline', space=cubic_b_spline_curve_5_dof_space, 
                                              coefficients=wing_chord_stretch_coefficients, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sect_param.num_sections).reshape((-1,1))
    wing_span_stretch = wing_span_strech_b_spline.evaluate(section_parametric_coordinates)
    wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
            'wing_span_stretch': wing_span_stretch,
            'wing_chord_stretch': wing_chord_stretch,
    }
    # wing_plot = True
    wing_ffd_block_coefficients = wing_ffd_block_sect_param.evaluate(sectional_parameters, plot=wing_plot)
    wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=wing_plot)
    # endregion

    # region strut 

    # region left strut try 
    left_strut_ffd_block = lg.construct_ffd_block_around_entities(name='left_strut_ffd_block', entities=strut_left, num_coefficients=(2, 5, 2), order=(2, 2, 2))

    left_strut_ffd_block_sect_param = VolumeSectionalParameterization(name='left_strut_ffd_sectional_parameterization', 
                                                                principal_parametric_dimension=1, 
                                                                parameterized_points=left_strut_ffd_block.coefficients,
                                                                parameterized_points_shape=left_strut_ffd_block.coefficients_shape)

    left_strut_ffd_block_sect_param.add_sectional_translation(name='left_strut_span_stretch', axis=1)
    left_strut_ffd_block_sect_param.add_sectional_translation(name='left_strut_translation_x', axis=0)
    left_strut_ffd_block_sect_param.add_sectional_translation(name='left_strut_translation_z', axis=2)

    # strut_span_stretch_coefficients = m3l.Variable(name= 'strut_span_stretch_coefficients', shape=(2, ), value=np.array([0.,0.]))
    left_strut_span_stretch_coefficients = system_model.create_input(name= 'left_strut_span_stretch_coefficients', shape=(2, ), val=np.array([0.,0.]))
    left_strut_span_strech_b_spline = bsp.BSpline(name='left_strut_span_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=left_strut_span_stretch_coefficients, num_physical_dimensions=1)

    left_strut_translation_x_coefficients = system_model.create_input('left_strut_translation_x_coefficients', shape=(1, ), val=np.array([0]))
    left_strut_translation_z_coefficients = system_model.create_input('left_strut_translation_z_coefficients', shape=(1, ), val=np.array([0]))
    
    left_strut_ffd_block_translation_x_b_spline = bsp.BSpline(name='left_strut_translation_x_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=left_strut_translation_x_coefficients, 
                                                        num_physical_dimensions=1)
        
    left_strut_ffd_block_translation_z_b_spline = bsp.BSpline(name='left_strut_translation_z_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=left_strut_translation_z_coefficients, 
                                                        num_physical_dimensions=1)
        
    section_parametric_coordinates = np.linspace(0., 1., left_strut_ffd_block_sect_param.num_sections).reshape((-1,1))
    left_strut_translation_x = left_strut_ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
    left_strut_translation_z = left_strut_ffd_block_translation_z_b_spline.evaluate(section_parametric_coordinates)
    left_strut_span_stretch = left_strut_span_strech_b_spline.evaluate(section_parametric_coordinates)
    
    # section_parametric_coordinates = np.linspace(0., 1., left_strut_ffd_block_sect_param.num_sections).reshape((-1,1))
    # left_strut_span_stretch = left_strut_span_strech_b_spline.evaluate(section_parametric_coordinates)
    # left_strut_chord_stretch = left_strut_chord_stretch_b_spline.evaluate(section_parametric_coordinates)

    # sectional_parameters = {
    #         'left_strut_span_stretch': left_strut_span_stretch,
    #         # 'left_strut_chord_stretch': left_strut_chord_stretch,
    # }
    

    sectional_parameters = {
            'left_strut_translation_x': left_strut_translation_x,
            'left_strut_translation_z': left_strut_translation_z,
            'left_strut_span_stretch': left_strut_span_stretch,
    }

    left_strut_ffd_block_coefficients = left_strut_ffd_block_sect_param.evaluate(sectional_parameters, plot=left_strut_plot)
    left_strut_coefficients = left_strut_ffd_block.evaluate(left_strut_ffd_block_coefficients, plot=left_strut_plot)

    # endregion
    
    # region right strut new

    right_strut_ffd_block = lg.construct_ffd_block_around_entities(name='right_strut_ffd_block', entities=strut_right, num_coefficients=(2, 5, 2), order=(2, 2, 2))
    
    right_strut_ffd_block_sect_param = VolumeSectionalParameterization(name='right_strut_ffd_sectional_parameterization', 
                                                                principal_parametric_dimension=1, 
                                                                parameterized_points=right_strut_ffd_block.coefficients,
                                                                parameterized_points_shape=right_strut_ffd_block.coefficients_shape)

    right_strut_ffd_block_sect_param.add_sectional_translation(name='right_strut_span_stretch', axis=1)
    right_strut_ffd_block_sect_param.add_sectional_translation(name='right_strut_translation_x', axis=0)
    right_strut_ffd_block_sect_param.add_sectional_translation(name='right_strut_translation_z', axis=2)

    # strut_span_stretch_coefficients = m3l.Variable(name= 'strut_span_stretch_coefficients', shape=(2, ), value=np.array([0.,0.]))
    right_strut_span_stretch_coefficients = system_model.create_input(name= 'right_strut_span_stretch_coefficients', shape=(2, ), val=np.array([0.,0.]))
    right_strut_span_strech_b_spline = bsp.BSpline(name='right_strut_span_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=right_strut_span_stretch_coefficients, num_physical_dimensions=1)

    right_strut_translation_x_coefficients = system_model.create_input('right_strut_translation_x_coefficients', shape=(1, ), val=np.array([0]))
    right_strut_translation_z_coefficients = system_model.create_input('right_strut_translation_z_coefficients', shape=(1, ), val=np.array([0]))
    
    right_strut_ffd_block_translation_x_b_spline = bsp.BSpline(name='right_strut_translation_x_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=right_strut_translation_x_coefficients, 
                                                        num_physical_dimensions=1)
        
    right_strut_ffd_block_translation_z_b_spline = bsp.BSpline(name='right_strut_translation_z_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=right_strut_translation_z_coefficients, 
                                                        num_physical_dimensions=1)
        
    section_parametric_coordinates = np.linspace(0., 1., right_strut_ffd_block_sect_param.num_sections).reshape((-1,1))
    right_strut_translation_x = right_strut_ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
    right_strut_translation_z = right_strut_ffd_block_translation_z_b_spline.evaluate(section_parametric_coordinates)
    right_strut_span_stretch = right_strut_span_strech_b_spline.evaluate(section_parametric_coordinates)
    
    # section_parametric_coordinates = np.linspace(0., 1., right_strut_ffd_block_sect_param.num_sections).reshape((-1,1))
    # right_strut_span_stretch = right_strut_span_strech_b_spline.evaluate(section_parametric_coordinates)
    # right_strut_chord_stretch = right_strut_chord_stretch_b_spline.evaluate(section_parametric_coordinates)

    # sectional_parameters = {
    #         'right_strut_span_stretch': right_strut_span_stretch,
    #         # 'right_strut_chord_stretch': right_strut_chord_stretch,
    # }
    

    sectional_parameters = {
            'right_strut_translation_x': right_strut_translation_x,
            'right_strut_translation_z': right_strut_translation_z,
            'right_strut_span_stretch': right_strut_span_stretch,
    }

    right_strut_ffd_block_coefficients = right_strut_ffd_block_sect_param.evaluate(sectional_parameters, plot=right_strut_plot)
    right_strut_coefficients = right_strut_ffd_block.evaluate(right_strut_ffd_block_coefficients, plot=right_strut_plot)

    # endregion      

    # endregion

    # region Jury

    # region left Jury 

    left_jury_ffd_block = lg.construct_ffd_block_around_entities(name='left_jury_ffd_block', entities=jury_left, num_coefficients=(2, 2, 5), order=(2, 2, 2))
    
    left_jury_ffd_block_sect_param = VolumeSectionalParameterization(name='left_jury_ffd_sectional_parameterization', 
                                                                principal_parametric_dimension=1, 
                                                                parameterized_points=left_jury_ffd_block.coefficients,
                                                                parameterized_points_shape=left_jury_ffd_block.coefficients_shape)

    left_jury_ffd_block_sect_param.add_sectional_translation(name='left_jury_translation_x', axis= 0)
    left_jury_ffd_block_sect_param.add_sectional_translation(name='left_jury_translation_y', axis= 1)
    left_jury_ffd_block_sect_param.add_sectional_stretch(name='left_jury_stretch_z', axis=2)


    left_jury_translation_x_coefficients = system_model.create_input('left_jury_translation_x_coefficients', shape=(1, ), val=np.array([0]))
    left_jury_translation_y_coefficients = system_model.create_input('left_jury_translation_y_coefficients', shape=(1, ), val=np.array([0]))
    left_jury_stretch_z_coefficients = system_model.create_input('left_jury_stretch_z_coefficients', shape=(1, ), val=np.array([0]))

    left_jury_ffd_block_translation_x_b_spline = bsp.BSpline(name='left_jury_translation_x_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=left_jury_translation_x_coefficients, 
                                                        num_physical_dimensions=1)
        
    left_jury_ffd_block_translation_y_b_spline = bsp.BSpline(name='left_jury_translation_y_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=left_jury_translation_y_coefficients, 
                                                        num_physical_dimensions=1)
        
    left_jury_ffd_block_stretch_z_b_spline = bsp.BSpline(name='left_jury_stretch_z_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=left_jury_stretch_z_coefficients, 
                                                        num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., left_jury_ffd_block_sect_param.num_sections).reshape((-1,1))
    left_jury_translation_x = left_jury_ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
    left_jury_translation_y = left_jury_ffd_block_translation_y_b_spline.evaluate(section_parametric_coordinates)
    left_jury_stretch_z = left_jury_ffd_block_stretch_z_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
            'left_jury_translation_x': left_jury_translation_x,
            'left_jury_translation_y': left_jury_translation_y,
            'left_jury_stretch_z': left_jury_stretch_z,
    }
    
    left_jury_ffd_block_coefficients = left_jury_ffd_block_sect_param.evaluate(sectional_parameters, plot=left_jury_plot)
    left_jury_coefficients = left_jury_ffd_block.evaluate(left_jury_ffd_block_coefficients, plot=left_jury_plot)

    # endregion

    # region right Jury 

    right_jury_ffd_block = lg.construct_ffd_block_around_entities(name='right_jury_ffd_block', entities=jury_right, num_coefficients=(2, 2, 5), order=(2, 2, 2))
    
    right_jury_ffd_block_sect_param = VolumeSectionalParameterization(name='right_jury_ffd_sectional_parameterization', 
                                                                principal_parametric_dimension=1, 
                                                                parameterized_points=right_jury_ffd_block.coefficients,
                                                                parameterized_points_shape=right_jury_ffd_block.coefficients_shape)

    right_jury_ffd_block_sect_param.add_sectional_translation(name='right_jury_translation_x', axis=0)
    right_jury_ffd_block_sect_param.add_sectional_translation(name='right_jury_translation_y', axis=1)
    right_jury_ffd_block_sect_param.add_sectional_stretch(name='right_jury_stretch_z', axis=2)

    right_jury_translation_x_coefficients = system_model.create_input('right_jury_translation_x_coefficients', shape=(1, ), val=np.array([0]))
    right_jury_translation_y_coefficients = system_model.create_input('right_jury_translation_y_coefficients', shape=(1, ), val=np.array([0]))
    right_jury_stretch_z_coefficients = system_model.create_input('right_jury_stretch_z_coefficients', shape=(1, ), val=np.array([0]))

    right_jury_ffd_block_translation_x_b_spline = bsp.BSpline(name='right_jury_translation_x_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=right_jury_translation_x_coefficients, 
                                                        num_physical_dimensions=1)
        
    right_jury_ffd_block_translation_y_b_spline = bsp.BSpline(name='right_jury_translation_y_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=right_jury_translation_y_coefficients, 
                                                        num_physical_dimensions=1)
                
    right_jury_ffd_block_stretch_z_b_spline = bsp.BSpline(name='right_jury_stretch_z_bspline', space=constant_b_spline_curve_1_dof_space, coefficients=right_jury_stretch_z_coefficients, 
                                                        num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., right_jury_ffd_block_sect_param.num_sections).reshape((-1,1))
    right_jury_translation_x = right_jury_ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
    right_jury_translation_y = right_jury_ffd_block_translation_y_b_spline.evaluate(section_parametric_coordinates)
    right_jury_stretch_z = right_jury_ffd_block_stretch_z_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
            'right_jury_translation_x': right_jury_translation_x,
            'right_jury_translation_y': right_jury_translation_y,
            'right_jury_stretch_z': right_jury_stretch_z,
    }
    
    right_jury_ffd_block_coefficients = right_jury_ffd_block_sect_param.evaluate(sectional_parameters, plot=right_jury_plot)
    right_jury_coefficients = right_jury_ffd_block.evaluate(right_jury_ffd_block_coefficients, plot=right_jury_plot)    

    # endregion

    # endregion
