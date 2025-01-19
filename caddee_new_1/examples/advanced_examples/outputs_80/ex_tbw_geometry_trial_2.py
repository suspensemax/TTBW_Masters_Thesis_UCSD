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

# region True or False
geometry_plot = True
components_plot = False
print_coordinates = False
plot_htail_mesh = False
plot_left_strut_mesh = False
plot_right_strut_mesh = False

# region Plots 
# VLM 
mesh_flag_wing = False
mesh_flag_htail = False
mesh_flag_left_strut = False
mesh_flag_right_strut = False
mesh_flag_vtail_helper = False
mesh_flag_left_jury_helper = False
mesh_flag_right_jury_helper = False
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
geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_80.stp', parallelize=True)
geometry.refit(parallelize=True, order=(4, 4))

if geometry_plot:
    geometry.plot()
exit()
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
# endregion


# region geometry dv
wing_span_dv = system_model.create_input(name='wing_span_dv', shape = (1,), val = 170)
wing_tip_chord_left_dv = system_model.create_input(name='wing_tip_chord_left_dv', shape= (1,), val = 3.8)
wing_root_chord_dv = system_model.create_input(name='wing_root_chord_dv', shape= (1,), val = 10.7)
wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 9.5)
# endregion

# region meshes

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
vertex01_left_strut = np.array([55.315, -18.240, -3.271]) # left leading 2
vertex02_left_strut = np.array([57.405, -32.236, -0.947]) # left leading 3
vertex03_left_strut = np.array([59.717, -40.633, 0.447]) # left leading 4
vertex04_left_strut = np.array([59.914, -49.031, 1.840]) # left leading 5
vertex05_left_strut = np.array([62.005, -63.027, 4.164]) # left leading 6
vertex06_left_strut = np.array([63.898, -68.625, 5.093]) # left leading 7

vertex10_left_strut = np.array([57.309, -12.641, -4.200]) # left trailing 1
vertex11_left_strut = np.array([59.240, -18.240, -3.271]) # left trailing 2
vertex12_left_strut = np.array([61.331, -32.236, -0.947]) # left trailing 3
vertex13_left_strut = np.array([61.528, -40.633, 0.447]) # left trailing 4
vertex14_left_strut = np.array([63.840, -49.031, 1.840]) # left trailing 5
vertex15_left_strut = np.array([65.930, -63.027, 4.164]) # left trailing 6                
vertex16_left_strut = np.array([65.710, -68.625, 5.093]) # left trailing 7

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
vertex01_right_strut = np.array([55.315, 18.240, -3.271]) # right leading 2
vertex02_right_strut = np.array([57.405, 32.236, -0.947]) # right leading 3
vertex03_right_strut = np.array([59.717, 40.633, 0.447]) # right leading 4
vertex04_right_strut = np.array([59.914, 49.031, 1.840]) # right leading 5
vertex05_right_strut = np.array([62.005, 63.027, 4.164]) # right leading 6
vertex06_right_strut = np.array([63.898, 68.625, 5.093]) # right leading 7

vertex10_right_strut = np.array([57.309, 12.641, -4.200]) # right trailing 1
vertex11_right_strut = np.array([59.240, 18.240, -3.271]) # right trailing 2
vertex12_right_strut = np.array([61.331, 32.236, -0.947]) # right trailing 3
vertex13_right_strut = np.array([61.528, 40.633, 0.447]) # right trailing 4
vertex14_right_strut = np.array([63.840, 49.031, 1.840]) # right trailing 5
vertex15_right_strut = np.array([65.930, 63.027, 4.164]) # right trailing 6                
vertex16_right_strut = np.array([65.710, 68.625, 5.093]) # right trailing 7

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
# mesh_flag_left_jury_helper = True

left_jury_le_left = np.array([60.142, -40.666, 0.480])
left_jury_le_right = np.array([59.473, -40.666, 5.935])
left_jury_te_left = np.array([61.098, -40.666, 0.486])
left_jury_te_right = np.array([60.658, -40.666, 5.935])

left_jury_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=jury_left,
    num_spanwise=num_spanwise_vlm_jury,
    num_chordwise=num_chordwise_vlm_jury,
    le_left=left_jury_le_left, 
    le_right=left_jury_le_right,
    te_left=left_jury_te_left,
    te_right=left_jury_te_right,
    plot=mesh_flag_left_jury_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# region right jury mesh     
num_spanwise_vlm_jury = 5
num_chordwise_vlm_jury = 3

right_jury_le_left = np.array([60.142, 40.666, 0.480])
right_jury_le_right = np.array([59.473, 40.666, 5.935])
right_jury_te_left = np.array([61.098, 40.666, 0.486])
right_jury_te_right = np.array([60.658, 40.666, 5.935])

right_jury_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=jury_right,
    num_spanwise=num_spanwise_vlm_jury,
    num_chordwise=num_chordwise_vlm_jury,
    le_left=right_jury_le_left, 
    le_right=right_jury_le_right,
    te_left=right_jury_te_left,
    te_right=right_jury_te_right,
    plot=mesh_flag_right_jury_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# endregion

# endregion


def calculate_angle(point1, point2):
    # Calculate the vectors from the origin to each point
    vector1 = np.array(point1)
    vector2 = np.array(point2)
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# Example points
point1 = point40_wing
point2 = vertex05_left_strut

# Calculate the angle between the two points
angle_beta_not_m3l = calculate_angle(point1, point2)
angle_beta = system_model.create_input(name = 'angle_beta', shape = (1,), val = angle_beta_not_m3l)

# region beam mesh 
ft_to_m = 0.3048


# region wing beam mesh 
    
num_wing_beam = 21

wing_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=wing,
    num_beam_nodes=num_wing_beam,
    # te_right=np.array([73., 85., 5.507]),
    te_right=wing_te_right,
    # te_left=np.array([73., -85., 5.507]),
    te_left=wing_te_left,
    # te_center=np.array([57.953, 0.000, 6.566]),
    te_center=wing_te_center,
    # le_left=np.array([66., -85., 5.647]),
    le_left=wing_le_left,
    # le_right=np.array([66., 85., 5.647]),
    le_right=wing_le_right,
    # le_center=np.array([47.231, 0.000, 6.937]),
    le_center=wing_le_center,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=wing_beam_mesh_flag,
    le_interp='ellipse',
    te_interp='linear',
    horizontal = 'yes',
)

# endregion

# region strut beam mesh
num_strut_beam = 9
# num_strut_beam = 21

# region left strut beam mesh

left_strut_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=strut,
    num_beam_nodes=num_strut_beam,
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
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=left_strut_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'yes',
)
# endregion

# region right strut beam mesh
right_strut_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=strut,
    num_beam_nodes=num_strut_beam,
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
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=right_strut_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'yes',
)
# endregion
# endregion

# region jury beam mesh
# region left jury beam mesh
num_jury_beam = 3
left_jury_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=jury,
    num_beam_nodes=num_jury_beam,
    le_left=left_jury_le_left, 
    le_right=left_jury_le_right,
    te_left=left_jury_te_left,
    te_right=left_jury_te_right,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=left_jury_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'no',
)
# endregion

# region right jury beam mesh
num_jury_beam = 3
right_jury_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=jury,
    num_beam_nodes=num_jury_beam,
    le_left=right_jury_le_left, 
    le_right=right_jury_le_right,
    te_left=right_jury_te_left,
    te_right=right_jury_te_right,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=right_jury_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'no',
)
# endregion
# endregion

# endregion



# Component surface areas
component_list = [Fuselage, wing, htail, vtail, jury, gearpod, strut]
surface_area_list = compute_component_surface_area(
    component_list=component_list,
    geometry=geometry,
    parametric_mesh_grid_num=20,
    plot=False,
)

# # S_ref_area = surface_area_list[1] / 1.969576502
# S_ref_area = surface_area_list[1] / 1.84
# h_tail_area = surface_area_list[2] / 2.1
# v_tail_area = surface_area_list[3] / 2.18
# # jury_area = surface_area_list[4] / 2.11
# jury_area = surface_area_list[4] / 2.02
# # strut_area = surface_area_list[6] / 2.07
# strut_area = surface_area_list[6] / 1.67
S_ref_area = surface_area_list[1] / 1.969576502
h_tail_area = surface_area_list[2] / 2.071613543
v_tail_area = surface_area_list[3] / 2.18
jury_area = surface_area_list[4] / 2.079463433
strut_area = surface_area_list[6] / 2.038053769
wing_AR = ((wing_span_dv)**2 )/ S_ref_area
print('wing area', S_ref_area.value)
print('htail area', h_tail_area.value)
print('v_tail area', v_tail_area.value)
print('jury area', jury_area.value)
print('strut area', strut_area.value)
print('wing_AR', wing_AR.value)