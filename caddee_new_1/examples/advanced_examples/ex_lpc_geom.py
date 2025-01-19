import time
from lsdo_geo.core.geometry.geometry_functions import import_geometry
import m3l
import numpy as np
from python_csdl_backend import Simulator
from caddee import GEOMETRY_FILES_FOLDER
import lsdo_geo as lg
from caddee.utils.helper_functions.geometry_helpers import (make_rotor_mesh, make_vlm_camber_mesh, make_1d_box_beam_mesh, 
                                                            compute_component_surface_area, BladeParameters)
from caddee.utils.aircraft_models.drag_models.drag_build_up import DragComponent


geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'LPC_final_custom_blades.stp', parallelize=True)
geometry.refit(parallelize=True)

system_model = m3l.Model()
FFD = True
rotor_geom_dv = True

# region Declaring all components
# Wing, tails, fuselage
wing = geometry.declare_component(component_name='wing', b_spline_search_names=['Wing'])
# wing.plot()
h_tail = geometry.declare_component(component_name='h_tail', b_spline_search_names=['Tail_1'])
# h_tail.plot()
v_tail = geometry.declare_component(component_name='v_tail', b_spline_search_names=['Tail_2'])
# vtail.plot()
fuselage = geometry.declare_component(component_name='fuselage', b_spline_search_names=['Fuselage_***.main'])
# fuselage.plot()

# Nose hub
nose_hub = geometry.declare_component(component_name='weird_nose_hub', b_spline_search_names=['EngineGroup_10'])
# nose_hub.plot()


# Pusher prop
pp_disk = geometry.declare_component(component_name='pp_disk', b_spline_search_names=['Rotor-9-disk'])
# pp_disk.plot()
pp_blade_1 = geometry.declare_component(component_name='pp_blade_1', b_spline_search_names=['Rotor_9_blades, 0'])
# pp_blade_1.plot()
pp_blade_2 = geometry.declare_component(component_name='pp_blade_2', b_spline_search_names=['Rotor_9_blades, 1'])
# pp_blade_2.plot()
pp_blade_3 = geometry.declare_component(component_name='pp_blade_3', b_spline_search_names=['Rotor_9_blades, 2'])
# pp_blade_3.plot()
pp_blade_4 = geometry.declare_component(component_name='pp_blade_4', b_spline_search_names=['Rotor_9_blades, 3'])
# pp_blade_4.plot()
pp_hub = geometry.declare_component(component_name='pp_hub', b_spline_search_names=['Rotor_9_Hub'])
# pp_hub.plot()
pp_components = [pp_disk, pp_blade_1, pp_blade_2, pp_blade_3, pp_blade_4, pp_hub]

# Rotor: rear left outer
rlo_disk = geometry.declare_component(component_name='rlo_disk', b_spline_search_names=['Rotor_2_disk'])
# rlo_disk.plot()
rlo_blade_1 = geometry.declare_component(component_name='rlo_blade_1', b_spline_search_names=['Rotor_2_blades, 0'])
# rlo_blade_1.plot()
rlo_blade_2 = geometry.declare_component(component_name='rlo_blade_2', b_spline_search_names=['Rotor_2_blades, 1'])
# rlo_blade_2.plot()
rlo_hub = geometry.declare_component(component_name='rlo_hub', b_spline_search_names=['Rotor_2_Hub'])
# rlo_hub.plot()
rlo_boom = geometry.declare_component(component_name='rlo_boom', b_spline_search_names=['Rotor_2_Support'])
# rlo_boom.plot()
rlo_components = [rlo_disk, rlo_blade_1, rlo_blade_2, rlo_hub]

# Rotor: rear left inner
rli_disk = geometry.declare_component(component_name='rli_disk', b_spline_search_names=['Rotor_4_disk'])
# rli_disk.plot()
rli_blade_1 = geometry.declare_component(component_name='rli_blade_1', b_spline_search_names=['Rotor_4_blades, 0'])
# rli_blade_1.plot()
rli_blade_2 = geometry.declare_component(component_name='rli_blade_2', b_spline_search_names=['Rotor_4_blades, 1'])
# rli_blade_2.plot()
rli_hub = geometry.declare_component(component_name='rli_hub', b_spline_search_names=['Rotor_4_Hub'])
# rli_hub.plot()
rli_boom = geometry.declare_component(component_name='rli_boom', b_spline_search_names=['Rotor_4_Support'])
# rli_boom.plot()
rli_components = [rli_disk, rli_blade_1, rli_blade_2, rli_hub]

# Rotor: rear right inner
rri_disk = geometry.declare_component(component_name='rri_disk', b_spline_search_names=['Rotor_6_disk'])
# rri_disk.plot()
rri_blade_1 = geometry.declare_component(component_name='rri_blade_1', b_spline_search_names=['Rotor_6_blades, 0'])
# rri_blade_1.plot()
rri_blade_2 = geometry.declare_component(component_name='rri_blade_2', b_spline_search_names=['Rotor_6_blades, 1'])
# rri_blade_2.plot()
rri_hub = geometry.declare_component(component_name='rri_hub', b_spline_search_names=['Rotor_6_Hub'])
# rri_hub.plot()
rri_boom = geometry.declare_component(component_name='rri_boom', b_spline_search_names=['Rotor_6_Support'])
# rri_boom.plot()
rri_components = [rri_disk, rri_blade_1, rri_blade_2, rri_hub]

# Rotor: rear right outer
rro_disk = geometry.declare_component(component_name='rro_disk', b_spline_search_names=['Rotor_8_disk'])
# rro_disk.plot()
rro_blade_1 = geometry.declare_component(component_name='rro_blade_1', b_spline_search_names=['Rotor_8_blades, 0'])
# rro_blade_1.plot()
rro_blade_2 = geometry.declare_component(component_name='rro_blade_2', b_spline_search_names=['Rotor_8_blades, 1'])
# rro_blade_2.plot()
rro_hub = geometry.declare_component(component_name='rro_hub', b_spline_search_names=['Rotor_8_Hub'])
# rro_hub.plot()
rro_boom = geometry.declare_component(component_name='rro_boom', b_spline_search_names=['Rotor_8_Support'])
# rro_boom.plot()
rro_components = [rro_disk, rro_blade_1, rro_blade_2, rro_hub]

# Rotor: front left outer
flo_disk = geometry.declare_component(component_name='flo_disk', b_spline_search_names=['Rotor_1_disk'])
# flo_disk.plot()
flo_blade_1 = geometry.declare_component(component_name='flo_blade_1', b_spline_search_names=['Rotor_1_blades, 0'])
# flo_blade_1.plot()
flo_blade_2 = geometry.declare_component(component_name='flo_blade_2', b_spline_search_names=['Rotor_1_blades, 1'])
# flo_blade_2.plot()
flo_hub = geometry.declare_component(component_name='flo_hub', b_spline_search_names=['Rotor_1_Hub'])
# flo_hub.plot()
flo_boom = geometry.declare_component(component_name='flo_boom', b_spline_search_names=['Rotor_1_Support'])
# flo_boom.plot()
flo_components = [flo_disk, flo_blade_1, flo_blade_2, flo_hub]

# Rotor: front left inner
fli_disk = geometry.declare_component(component_name='fli_disk', b_spline_search_names=['Rotor_3_disk'])
# fli_disk.plot()
fli_blade_1 = geometry.declare_component(component_name='fli_blade_1', b_spline_search_names=['Rotor_3_blades, 0'])
# fli_blade_1.plot()
fli_blade_2 = geometry.declare_component(component_name='fli_blade_2', b_spline_search_names=['Rotor_3_blades, 1'])
# fli_blade_2.plot()
fli_hub = geometry.declare_component(component_name='fli_hub', b_spline_search_names=['Rotor_3_Hub'])
# fli_hub.plot()
fli_boom = geometry.declare_component(component_name='fli_boom', b_spline_search_names=['Rotor_3_Support'])
# fli_boom.plot()
fli_components = [fli_disk, fli_blade_1, fli_blade_2, fli_hub]

# Rotor: front right inner
fri_disk = geometry.declare_component(component_name='fri_disk', b_spline_search_names=['Rotor_5_disk'])
# fri_disk.plot()
fri_blade_1 = geometry.declare_component(component_name='fri_blade_1', b_spline_search_names=['Rotor_5_blades, 0'])
# fri_blade_1.plot()
fri_blade_2 = geometry.declare_component(component_name='fri_blade_2', b_spline_search_names=['Rotor_5_blades, 1'])
# fri_blade_2.plot()
fri_hub = geometry.declare_component(component_name='fri_hub', b_spline_search_names=['Rotor_5_Hub'])
# fri_hub.plot()
fri_boom = geometry.declare_component(component_name='fri_boom', b_spline_search_names=['Rotor_5_Support'])
# fri_boom.plot()
fri_components = [fri_disk, fri_blade_1, fri_blade_2, fri_hub]

# Rotor: front right outer
fro_disk = geometry.declare_component(component_name='fro_disk', b_spline_search_names=['Rotor_7_disk'])
# fro_disk.plot()
fro_blade_1 = geometry.declare_component(component_name='fro_blade_1', b_spline_search_names=['Rotor_7_blades, 0'])
# fro_blade_1.plot()
fro_blade_2 = geometry.declare_component(component_name='fro_blade_2', b_spline_search_names=['Rotor_7_blades, 1'])
# fro_blade_2.plot()
fro_hub = geometry.declare_component(component_name='fro_hub', b_spline_search_names=['Rotor_7_Hub'])
# fro_hub.plot()
fro_boom = geometry.declare_component(component_name='fro_boom', b_spline_search_names=['Rotor_7_Support'])
# fro_boom.plot()
fro_components = [fro_disk, fro_blade_1, fro_blade_2, fro_hub]
lift_rotor_related_components = [rlo_components, rli_components, rri_components, rro_components, 
                                 flo_components, fli_components, fri_components, fro_components]

boom_components = [rlo_boom, rli_boom, rri_boom, rro_boom, flo_boom, fli_boom, fri_boom, fro_boom]

# endregion

# region Defining key points
wing_te_right = wing.project(np.array([13.4, 25.250, 7.5]), plot=False)
wing_te_left = wing.project(np.array([13.4, -25.250, 7.5]), plot=False)
wing_te_center = wing.project(np.array([14.332, 0., 8.439]), plot=False)
wing_le_left = wing.project(np.array([12.356, -25.25, 7.618]), plot=False)
wing_le_right = wing.project(np.array([12.356, 25.25, 7.618]), plot=False)
wing_le_center = wing.project(np.array([8.892, 0., 8.633]), plot=False)
wing_qc = wing.project(np.array([10.25, 0., 8.5]), plot=False)

tail_te_right = h_tail.project(np.array([31.5, 6.75, 6.]), plot=False)
tail_te_left = h_tail.project(np.array([31.5, -6.75, 6.]), plot=False)
tail_le_right = h_tail.project(np.array([26.5, 6.75, 6.]), plot=False)
tail_le_left = h_tail.project(np.array([26.5, -6.75, 6.]), plot=False)
tail_te_center = h_tail.project(np.array([31.187, 0., 8.009]), plot=False)
tail_le_center = h_tail.project(np.array([27.428, 0., 8.009]), plot=False)
tail_qc = h_tail.project(np.array([24.15, 0., 8.]), plot=False)

fuselage_wing_qc = fuselage.project(np.array([10.25, 0., 8.5]), plot=False)
fuselage_wing_te_center = fuselage.project(np.array([14.332, 0., 8.439]), plot=False)
fuselage_tail_qc = fuselage.project(np.array([24.15, 0., 8.]), plot=False)
fuselage_tail_te_center = fuselage.project(np.array([31.187, 0., 8.009]), plot=False)

rlo_disk_pt = np.array([19.200, -18.750, 9.635])
rro_disk_pt = np.array([19.200, 18.750, 9.635])
rlo_boom_pt = np.array([12.000, -18.750, 7.613])
rro_boom_pt = np.array([12.000, 18.750, 7.613])

flo_disk_pt = np.array([5.070, -18.750, 7.355])
fro_disk_pt = np.array([5.070, 18.750, 7.355])
flo_boom_pt = np.array([12.200, -18.750, 7.615])
fro_boom_pt = np.array([12.200, 18.750, 7.615])

rli_disk_pt = np.array([18.760, -8.537, 9.919])
rri_disk_pt = np.array([18.760, 8.537, 9.919])
rli_boom_pt = np.array([11.500, -8.250, 7.898])
rri_boom_pt = np.array([11.500, 8.250, 7.898])

fli_disk_pt = np.array([4.630, -8.217, 7.659])
fri_disk_pt = np.array([4.630, 8.217, 7.659])
fli_boom_pt = np.array([11.741, -8.250, 7.900])
fri_boom_pt = np.array([11.741, 8.250, 7.900])

rlo_disk_center = rlo_disk.project(rlo_disk_pt)
rli_disk_center = rli_disk.project(rli_disk_pt)
rri_disk_center = rri_disk.project(rri_disk_pt)
rro_disk_center = rro_disk.project(rro_disk_pt)
flo_disk_center = flo_disk.project(flo_disk_pt)
fli_disk_center = fli_disk.project(fli_disk_pt)
fri_disk_center = fri_disk.project(fri_disk_pt)
fro_disk_center = fro_disk.project(fro_disk_pt)

rlo_disk_center_on_wing = wing.project(rlo_disk_pt)
rli_disk_center_on_wing = wing.project(rli_disk_pt)
rri_disk_center_on_wing = wing.project(rri_disk_pt)
rro_disk_center_on_wing = wing.project(rro_disk_pt)
flo_disk_center_on_wing = wing.project(flo_disk_pt)
fli_disk_center_on_wing = wing.project(fli_disk_pt)
fri_disk_center_on_wing = wing.project(fri_disk_pt)
fro_disk_center_on_wing = wing.project(fro_disk_pt)

boom_fro = fro_boom.project(fro_boom_pt)
boom_fri = fri_boom.project(fri_boom_pt)
boom_flo = flo_boom.project(flo_boom_pt)
boom_fli = fli_boom.project(fli_boom_pt)
boom_rro = rro_boom.project(rro_boom_pt)
boom_rri = rri_boom.project(rri_boom_pt)
boom_rli = rli_boom.project(rli_boom_pt)
boom_rlo = rlo_boom.project(rlo_boom_pt)

wing_boom_fro = wing.project(fro_boom_pt)
wing_boom_fri = wing.project(fri_boom_pt)
wing_boom_flo = wing.project(flo_boom_pt)
wing_boom_fli = wing.project(fli_boom_pt)
wing_boom_rro = wing.project(rro_boom_pt)
wing_boom_rri = wing.project(rri_boom_pt)
wing_boom_rli = wing.project(rli_boom_pt)
wing_boom_rlo = wing.project(rlo_boom_pt)

fuselage_nose = np.array([2.464, 0., 5.113])
fuselage_rear = np.array([31.889, 0., 7.798])
fuselage_nose_points_parametric = fuselage.project(fuselage_nose, grid_search_density_parameter=20)
fueslage_rear_points_parametric = fuselage.project(fuselage_rear)
fuselage_rear_point_on_pusher_disk_parametric = pp_disk.project(fuselage_rear)

fuselage_l1_parametric = fuselage.project(np.array([1.889, 0., 4.249]))
fuselage_l2_parametric = fuselage.project(np.array([31.889, 0., 7.798]))

fuselage_d1_parametric = fuselage.project(np.array([10.916, -2.945, 5.736]))
fuselage_d2_parametric = fuselage.project(np.array([10.916, 2.945, 5.736]))

fuselage_d1_parametric = fuselage.project(np.array([10.916, -2.945, 5.736]))
fuselage_d2_parametric = fuselage.project(np.array([10.916, 2.945, 5.736]))

# wing
wing_mid_le_parametric = wing.project(np.array([8.892, 0., 8.633]))
wing_mid_te_parametric = wing.project(np.array([14.332, 0, 8.439]))

# wing_le_right_parametric = wing.project(wing_le_right)
# wing_le_left__parametric = wing.project(wing_le_left)

# htail
h_tail_mid_le_parametric = h_tail.project(np.array([27.806, -6.520, 8.008]))
h_tail_mid_te_parametric = h_tail.project(np.array([30.050, -6.520, 8.008]))

# vtail
vtail_mid_le_parametric = v_tail.project(np.array([26.971, 0.0, 11.038]))
vtail_mid_te_parametric = v_tail.project(np.array([31.302, 0.0, 11.038]))

# boom
boom_l1_parametric = rlo_boom.project(np.array([20.000, -18.750, 7.613]))
boom_l2_parametric = rlo_boom.project(np.array([12.000, -18.750, 7.613]))

boom_d1_parametric = rlo_boom.project(np.array([15.600, -19.250, 7.613]))
boom_d2_parametric = rlo_boom.project(np.array([15.600, -18.250, 7.613]))

# lift hubs
hub_l1_parametric = rlo_hub.project(np.array([18.075, -18.750,9.525]))
hub_l2_parametric = rlo_hub.project(np.array([20.325, -18.750,9.525]))

# blade 
blade_tip_parametric = rlo_blade_2.project(np.array([14.200, -18.626, 9.040]))
blade_hub_parametric = rlo_blade_2.project(np.array([18.200, -18.512, 9.197]))
# endregion

# region rotor meshes
num_radial = 30
num_spanwise_vlm_rotor = 8
num_chord_vlm_rotor = 2

# Pusher prop
pp_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=pp_disk,
    origin=np.array([32.625, 0., 7.79]),
    y1=np.array([31.94, 0.00, 3.29]),
    y2=np.array([31.94, 0.00, 12.29]),
    z1=np.array([31.94, -4.50, 7.78]),
    z2=np.array([31.94, 4.45, 7.77]),
    create_disk_mesh=False,
    plot=False,
)

# Rear left outer
rlo_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rlo_disk,
    origin=np.array([19.2, -18.75, 9.01]),
    y1=np.array([19.2, -13.75, 9.01]),
    y2=np.array([19.2, -23.75, 9.01]),
    z1=np.array([14.2, -18.75, 9.01]),
    z2=np.array([24.2, -18.75, 9.01]),
    create_disk_mesh=False,
    plot=False,
)

# Rear right outer 
rro_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rro_disk,
    origin=np.array([19.2, 18.75, 9.01]),
    y1=np.array([19.2, 23.75, 9.01]),
    y2=np.array([19.2, 13.75, 9.01]),
    z1=np.array([14.2, 18.75, 9.01]),
    z2=np.array([24.2, 18.75, 9.01]),
    create_disk_mesh=False,
    plot=False,
)


# Front left outer 
flo_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=flo_disk,
    origin=np.array([5.07, -18.75, 6.73]),
    y1=np.array([5.070, -13.750, 6.730]),
    y2=np.array([5.070, -23.750, 6.730]),
    z1=np.array([0.070, -18.750, 6.730]),
    z2=np.array([10.070, -18.750, 6.730]),
    create_disk_mesh=False,
    plot=False,
)


# Front right outer 
fro_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=fro_disk,
    origin=np.array([5.07, 18.75, 6.73]),
    y1=np.array([5.070, 23.750, 6.730]),
    y2=np.array([5.070, 13.750, 6.730]),
    z1=np.array([0.070, 18.750, 6.730]),
    z2=np.array([10.070, 18.750, 6.730]),
    create_disk_mesh=False,
    plot=False,
)


# Rear left inner
rli_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rli_disk,
    origin=np.array([18.760, -8.537, 9.919]),
    y1=np.array([18.760, -3.499, 9.996]),
    y2=np.array([18.760, -13.401, 8.604]),
    z1=np.array([13.760, -8.450, 9.300]),
    z2=np.array([23.760, -8.450, 9.300]),
    create_disk_mesh=False,
    plot=False,
)

# Rear right inner
rri_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rri_disk,
    origin=np.array([18.760, 8.537, 9.919]),
    y1=np.array([18.760, 13.401, 8.604]),
    y2=np.array([18.760, 3.499, 9.996]),
    z1=np.array([13.760, 8.450, 9.300]),
    z2=np.array([23.760, 8.450, 9.300]),
    create_disk_mesh=False,
    plot=False,
)

# Front left inner
fli_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=fli_disk,
    origin=np.array([4.630, -8.217, 7.659]),
    y1=np.array([4.630, -3.179, 7.736]),
    y2=np.array([4.630, -13.081, 6.344]),
    z1=np.array([-0.370, -8.130, 7.040]),
    z2=np.array([9.630, -8.130, 7.040]),
    create_disk_mesh=False,
    plot=False,
)

# Front right inner
fri_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=fri_disk,
    origin=np.array([4.630, 8.217, 7.659]), 
    y1=np.array([4.630, 13.081, 6.344]),
    y2=np.array([4.630, 3.179, 7.736]),
    z1=np.array([-0.370, 8.130, 7.040]),
    z2=np.array([9.630, 8.130, 7.040]),
    create_disk_mesh=False,
    plot=False,
)

lift_rotor_mesh_list = [rlo_mesh, rli_mesh, rri_mesh, rro_mesh, flo_mesh, fli_mesh, fri_mesh, fro_mesh]
lift_rotor_mesh_list_oei_flo = [rlo_mesh, rli_mesh, rri_mesh, rro_mesh, fli_mesh, fri_mesh, fro_mesh]
lift_rotor_mesh_list_oei_fli = [rlo_mesh, rli_mesh, rri_mesh, rro_mesh, flo_mesh, fri_mesh, fro_mesh]
lift_rotor_mesh_list_oei_rlo = [rli_mesh, rri_mesh, rro_mesh, flo_mesh, fli_mesh, fri_mesh, fro_mesh]
lift_rotor_mesh_list_oei_rli = [rlo_mesh, rri_mesh, rro_mesh, flo_mesh, fli_mesh, fri_mesh, fro_mesh]

lift_rotor_origin_list = [mesh.thrust_origin for mesh in lift_rotor_mesh_list]
all_rotor_origin_list = lift_rotor_origin_list + [pp_mesh.thrust_origin]
# endregion

# region Projection for meshes
num_spanwise_vlm = 25
num_chordwise_vlm = 8

wing_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=wing,
    num_spanwise=num_spanwise_vlm,
    num_chordwise=num_chordwise_vlm,
    te_right=np.array([13.4, 25.250, 7.5]),
    te_left=np.array([13.4, -25.250, 7.5]),
    te_center=np.array([14.332, 0., 8.439]),
    le_left=np.array([12.356, -25.25, 7.618]),
    le_right=np.array([12.356, 25.25, 7.618]),
    le_center=np.array([8.892, 0., 8.633]),
    grid_search_density_parameter=100,
    off_set_x=0.2,
    bunching_cos=True,
    plot=False,
    mirror=True,
)

num_spanwise_vlm_htail = 8
num_chordwise_vlm_htail = 4

tail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=h_tail, 
    num_spanwise=num_spanwise_vlm_htail,
    num_chordwise=num_chordwise_vlm_htail,
    te_right=np.array([31.5, 6.75, 6.]),
    te_left=np.array([31.5, -6.75, 6.]),
    le_right=np.array([26.5, 6.75, 6.]),
    le_left=np.array([26.5, -6.75, 6.]),
    plot=False,
    mirror=True,
)

# geometry.plot_meshes([wing_meshes.vlm_mesh])
# endregion
num_spanwise_vlm = 17
num_chordwise_vlm = 5
leading_edge_line_parametric = wing.project(np.linspace(np.array([8.356, -26., 7.618]), np.array([8.356, 26., 7.618]), num_spanwise_vlm), 
                                 direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
trailing_edge_line_parametric = wing.project(np.linspace(np.array([15.4, -25.250, 7.5]), np.array([15.4, 25.250, 7.5]), num_spanwise_vlm), 
                                  direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
leading_edge_line = geometry.evaluate(leading_edge_line_parametric)
trailing_edge_line = geometry.evaluate(trailing_edge_line_parametric)
chord_surface = m3l.linspace(leading_edge_line, trailing_edge_line, num_chordwise_vlm)
upper_surface_wireframe_parametric = wing.project(chord_surface.value.reshape((num_chordwise_vlm,num_spanwise_vlm,3))+np.array([0., 0., 1.]), 
                                       direction=np.array([0., 0., -1.]), plot=False, grid_search_density_parameter=40.)
lower_surface_wireframe_parametric = wing.project(chord_surface.value.reshape((num_chordwise_vlm,num_spanwise_vlm,3))+np.array([0., 0., -1.]), 
                                       direction=np.array([0., 0., 1.]), plot=False, grid_search_density_parameter=40.)
upper_surface_wireframe = geometry.evaluate(upper_surface_wireframe_parametric)
lower_surface_wireframe = geometry.evaluate(lower_surface_wireframe_parametric)
camber_surface = m3l.linspace(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((num_chordwise_vlm, num_spanwise_vlm, 3))


if FFD:
    # region Parameterization
    from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
    from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
    from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
    import lsdo_geo.splines.b_splines as bsp

    constant_b_spline_curve_1_dof_space = bsp.BSplineSpace(name='constant_b_spline_curve_1_dof_space', order=1, parametric_coefficients_shape=(1,))
    linear_b_spline_curve_2_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_2_dof_space', order=2, parametric_coefficients_shape=(2,))
    linear_b_spline_curve_3_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_3_dof_space', order=3, parametric_coefficients_shape=(3,))
    cubic_b_spline_curve_5_dof_space = bsp.BSplineSpace(name='cubic_b_spline_curve_5_dof_space', order=4, parametric_coefficients_shape=(5,))

    # region Parameterization Setup
    parameterization_solver = ParameterizationSolver()

    # region Wing Parameterization setup
    wing_ffd_block = construct_ffd_block_around_entities('wing_ffd_block', entities=wing, num_coefficients=(2,11,2), order=(2,4,2))
    wing_ffd_block.coefficients.name = 'wing_ffd_block_coefficients'
    wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_sectional_parameterization',
                                                                                parameterized_points=wing_ffd_block.coefficients,
                                                                                parameterized_points_shape=wing_ffd_block.coefficients_shape,
                                                                                principal_parametric_dimension=1)
    wing_ffd_block_sectional_parameterization.add_sectional_stretch(name='sectional_wing_chord_stretch', axis=0)
    wing_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_wingspan_stretch', axis=1)
    # wing_ffd_block_sectional_parameterization.add_sectional_rotation(name='sectional_wing_twist', axis=1)
    wing_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_wing_translation_x', axis=0)
    wing_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_wing_translation_z', axis=2)

    wing_chord_stretch_coefficients = m3l.Variable(name='wing_chord_stretch_coefficients', shape=(3,), value=np.array([0., 0., 0.]))
    wing_chord_stretch_b_spline = bsp.BSpline(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                            coefficients=wing_chord_stretch_coefficients, num_physical_dimensions=1)

    wing_wingspan_stretch_coefficients = m3l.Variable(name='wing_wingspan_stretch_coefficients', shape=(2,), value=np.array([-0., 0.]))
    wing_wingspan_stretch_b_spline = bsp.BSpline(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=wing_wingspan_stretch_coefficients, num_physical_dimensions=1)

    # wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,), value=np.array([0., 0., 0., 0., 0.]))
    # wing_twist_b_spline = bsp.BSpline(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
    #                                           coefficients=wing_twist_coefficients, num_physical_dimensions=1)

    wing_translation_x_coefficients = m3l.Variable(name='wing_translation_x_coefficients', shape=(1,), value=np.array([0.]))
    wing_translation_x_b_spline = bsp.BSpline(name='wing_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                            coefficients=wing_translation_x_coefficients, num_physical_dimensions=1)

    wing_translation_z_coefficients = m3l.Variable(name='wing_translation_z_coefficients', shape=(1,), value=np.array([0.]))
    wing_translation_z_b_spline = bsp.BSpline(name='wing_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                            coefficients=wing_translation_z_coefficients, num_physical_dimensions=1)

    parameterization_solver.declare_state(name='wing_chord_stretch_coefficients', state=wing_chord_stretch_coefficients)
    parameterization_solver.declare_state(name='wing_wingspan_stretch_coefficients', state=wing_wingspan_stretch_coefficients, penalty_factor=1.e3)
    parameterization_solver.declare_state(name='wing_translation_x_coefficients', state=wing_translation_x_coefficients)
    parameterization_solver.declare_state(name='wing_translation_z_coefficients', state=wing_translation_z_coefficients)
    # endregion

    # region Horizontal Stabilizer setup
    h_tail_ffd_block = construct_ffd_block_around_entities('h_tail_ffd_block', entities=h_tail, num_coefficients=(2,11,2), order=(2,4,2))
    h_tail_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='h_tail_sectional_parameterization',
                                                                                parameterized_points=h_tail_ffd_block.coefficients,
                                                                                parameterized_points_shape=h_tail_ffd_block.coefficients_shape,
                                                                                principal_parametric_dimension=1)
    h_tail_ffd_block_sectional_parameterization.add_sectional_stretch(name='sectional_h_tail_chord_stretch', axis=0)
    h_tail_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_h_tail_span_stretch', axis=1)
    # h_tail_ffd_block_sectional_parameterization.add_sectional_rotation(name='sectional_h_tail_twist', axis=1)
    h_tail_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_h_tail_translation_x', axis=0)
    # Don't need to add translation_y because the span stretch covers that
    h_tail_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_h_tail_translation_z', axis=2)

    h_tail_chord_stretch_coefficients = m3l.Variable(name='h_tail_chord_stretch_coefficients', shape=(3,), value=np.array([0., 0., 0.]))
    h_tail_chord_stretch_b_spline = bsp.BSpline(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                            coefficients=h_tail_chord_stretch_coefficients, num_physical_dimensions=1)

    h_tail_span_stretch_coefficients = m3l.Variable(name='h_tail_span_stretch_coefficients', shape=(2,), value=np.array([-0., 0.]))
    h_tail_span_stretch_b_spline = bsp.BSpline(name='h_tail_span_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=h_tail_span_stretch_coefficients, num_physical_dimensions=1)

    # h_tail_twist_coefficients = m3l.Variable(name='h_tail_twist_coefficients', shape=(5,), value=np.array([0., 0., 0., 0., 0.]))
    # h_tail_twist_b_spline = bsp.BSpline(name='h_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
    #                                           coefficients=h_tail_twist_coefficients, num_physical_dimensions=1)

    h_tail_translation_x_coefficients = m3l.Variable(name='h_tail_translation_x_coefficients', shape=(1,), value=np.array([0.]))
    h_tail_translation_x_b_spline = bsp.BSpline(name='h_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                            coefficients=h_tail_translation_x_coefficients, num_physical_dimensions=1)
    h_tail_translation_z_coefficients = m3l.Variable(name='h_tail_translation_z_coefficients', shape=(1,), value=np.array([0.]))
    h_tail_translation_z_b_spline = bsp.BSpline(name='h_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                            coefficients=h_tail_translation_z_coefficients, num_physical_dimensions=1)

    parameterization_solver.declare_state(name='h_tail_chord_stretch_coefficients', state=h_tail_chord_stretch_coefficients)
    parameterization_solver.declare_state(name='h_tail_span_stretch_coefficients', state=h_tail_span_stretch_coefficients)
    parameterization_solver.declare_state(name='h_tail_translation_x_coefficients', state=h_tail_translation_x_coefficients)
    parameterization_solver.declare_state(name='h_tail_translation_z_coefficients', state=h_tail_translation_z_coefficients)
    # endregion

    # region Fuselage setup
    fuselage_ffd_block = construct_ffd_block_around_entities('fuselage_ffd_block', entities=[fuselage, nose_hub], num_coefficients=(2,2,2), order=(2,2,2))
    fuselage_ffd_block.coefficients.name = 'fuselage_ffd_block_coefficients'
    fuselage_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='fuselage_sectional_parameterization',
                                                                                parameterized_points=fuselage_ffd_block.coefficients,
                                                                                parameterized_points_shape=fuselage_ffd_block.coefficients_shape,
                                                                                principal_parametric_dimension=0)
    fuselage_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_fuselage_stretch', axis=0)

    fuselage_stretch_coefficients = m3l.Variable(name='fuselage_stretch_coefficients', shape=(2,), value=np.array([0., -0.]))
    fuselage_stretch_b_spline = bsp.BSpline(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=fuselage_stretch_coefficients, num_physical_dimensions=1)

    parameterization_solver.declare_state(name='fuselage_stretch_coefficients', state=fuselage_stretch_coefficients)
    # endregion

    # region Lift Rotors setup
    def add_rigid_body_translation(components_name, components, principal_parametric_dimension=0, change_radius=False):
        components_ffd_block = construct_ffd_block_around_entities(f'{components_name}_ffd_block', entities=components, num_coefficients=(2,2,2),
                                                                order=(2,2,2))
        components_ffd_block.coefficients.name = components_name + '_coefficients'
        components_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name=f'{components_name}_sectional_parameterization',
                                                                                    parameterized_points=components_ffd_block.coefficients,
                                                                                    parameterized_points_shape=components_ffd_block.coefficients_shape,
                                                                                    principal_parametric_dimension=principal_parametric_dimension)
        components_ffd_block_sectional_parameterization.add_sectional_translation(name=f'{components_name}_translation_x', axis=0)
        components_ffd_block_sectional_parameterization.add_sectional_translation(name=f'{components_name}_translation_y', axis=1)
        components_ffd_block_sectional_parameterization.add_sectional_translation(name=f'{components_name}_translation_z', axis=2)

        if change_radius is True:
            components_ffd_block_sectional_parameterization.add_sectional_stretch(name=f'{components_name}_stretch_y', axis=1)

            components_translation_x_coefficients = m3l.Variable(name=f'{components_name}_translation_x_coefficients', shape=(2,), value=np.array([0., 0.]))
            components_translation_x_b_spline = bsp.BSpline(name=f'{components_name}_translation_x_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                                    coefficients=components_translation_x_coefficients, num_physical_dimensions=1)
        
            components_stretch_y_coefficients = m3l.Variable(name=f'{components_name}_stretch_y_coefficients', shape=(1, ), value=np.array([0.]))
            components_stretch_y_b_spline = bsp.BSpline(name=f'{components_name}_stretch_y_b_spline', space=constant_b_spline_curve_1_dof_space, 
                                                        coefficients=components_stretch_y_coefficients, num_physical_dimensions=1)
        else:
            components_translation_x_coefficients = m3l.Variable(name=f'{components_name}_translation_x_coefficients', shape=(1,), value=np.array([0.]))
            components_translation_x_b_spline = bsp.BSpline(name=f'{components_name}_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space, 
                                                    coefficients=components_translation_x_coefficients, num_physical_dimensions=1)
        
        components_translation_y_coefficients = m3l.Variable(name=f'{components_name}_translation_y_coefficients', shape=(1,), value=np.array([0.]))
        components_translation_y_b_spline = bsp.BSpline(name=f'{components_name}_translation_y_b_spline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=components_translation_y_coefficients, num_physical_dimensions=1)
        
        components_translation_z_coefficients = m3l.Variable(name=f'{components_name}_translation_z_coefficients', shape=(1,), value=np.array([0.]))
        components_translation_z_b_spline = bsp.BSpline(name=f'{components_name}_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=components_translation_z_coefficients, num_physical_dimensions=1)

        if change_radius is False:
            return [components_translation_x_b_spline, components_translation_y_b_spline, components_translation_z_b_spline], \
                components_ffd_block_sectional_parameterization, components_ffd_block
        else:
            return [components_translation_x_b_spline, components_translation_y_b_spline, components_translation_z_b_spline, components_stretch_y_b_spline], \
                components_ffd_block_sectional_parameterization, components_ffd_block

    lift_rotor_parameterization_objects = {}
    for component_set in lift_rotor_related_components:
        components_name = component_set[0].name[:3] + '_lift_rotor_components'
        component_parameterization_b_splines, component_sectional_parameterization, component_ffd_block = add_rigid_body_translation(
                                                                                                                    components_name, component_set, change_radius=True)
        lift_rotor_parameterization_objects[f'{components_name}_parameterization_b_splines'] = component_parameterization_b_splines
        lift_rotor_parameterization_objects[f'{components_name}_sectional_parameterization'] = component_sectional_parameterization
        lift_rotor_parameterization_objects[f'{components_name}_ffd_block'] = component_ffd_block

        parameterization_solver.declare_state(name=f'{components_name}_translation_x_coefficients', state=component_parameterization_b_splines[0].coefficients)
        parameterization_solver.declare_state(name=f'{components_name}_translation_y_coefficients', state=component_parameterization_b_splines[1].coefficients)
        parameterization_solver.declare_state(name=f'{components_name}_translation_z_coefficients', state=component_parameterization_b_splines[2].coefficients)
        parameterization_solver.declare_state(name=f'{components_name}_stretch_y_coefficients', state=component_parameterization_b_splines[3].coefficients)

    # endregion

    # region booms
    boom_parameterization_objects = {}
    for component in boom_components:
        components_name = component.name
        component_parameterization_b_splines, component_sectional_parameterization, component_ffd_block = add_rigid_body_translation(
                                                                                                                    components_name, component)
        boom_parameterization_objects[f'{components_name}_parameterization_b_splines'] = component_parameterization_b_splines
        boom_parameterization_objects[f'{components_name}_sectional_parameterization'] = component_sectional_parameterization
        boom_parameterization_objects[f'{components_name}_ffd_block'] = component_ffd_block

        parameterization_solver.declare_state(name=f'{components_name}_translation_x_coefficients', state=component_parameterization_b_splines[0].coefficients)
        parameterization_solver.declare_state(name=f'{components_name}_translation_y_coefficients', state=component_parameterization_b_splines[1].coefficients)
        parameterization_solver.declare_state(name=f'{components_name}_translation_z_coefficients', state=component_parameterization_b_splines[2].coefficients)
    # endregion

    # region pusher
    pusher_parameterization_objects = {}
    component_set = pp_components
    components_name = component_set[0].name[:3] + '_pusher_rotor_components'
    component_parameterization_b_splines, component_sectional_parameterization, component_ffd_block = add_rigid_body_translation(
                                                                                                        components_name, component_set)
    pusher_parameterization_objects[f'{components_name}_parameterization_b_splines'] = component_parameterization_b_splines
    pusher_parameterization_objects[f'{components_name}_sectional_parameterization'] = component_sectional_parameterization
    pusher_parameterization_objects[f'{components_name}_ffd_block'] = component_ffd_block

    parameterization_solver.declare_state(name=f'{components_name}_translation_x_coefficients', state=component_parameterization_b_splines[0].coefficients)
    parameterization_solver.declare_state(name=f'{components_name}_translation_y_coefficients', state=component_parameterization_b_splines[1].coefficients)
    parameterization_solver.declare_state(name=f'{components_name}_translation_z_coefficients', state=component_parameterization_b_splines[2].coefficients)
    # endregion

    # region Vertical Stabilizer setup
    vtail_parameterization_objects = {}
    component_set = v_tail
    components_name = v_tail.name
    component_parameterization_b_splines, component_sectional_parameterization, component_ffd_block = add_rigid_body_translation(
                                                                                                        components_name, component_set, principal_parametric_dimension=2)
    vtail_parameterization_objects[f'{components_name}_parameterization_b_splines'] = component_parameterization_b_splines
    vtail_parameterization_objects[f'{components_name}_sectional_parameterization'] = component_sectional_parameterization
    vtail_parameterization_objects[f'{components_name}_ffd_block'] = component_ffd_block

    parameterization_solver.declare_state(name=f'{components_name}_translation_x_coefficients', state=component_parameterization_b_splines[0].coefficients)
    parameterization_solver.declare_state(name=f'{components_name}_translation_y_coefficients', state=component_parameterization_b_splines[1].coefficients)
    parameterization_solver.declare_state(name=f'{components_name}_translation_z_coefficients', state=component_parameterization_b_splines[2].coefficients)
    # endregion

    # endregion

    # region Parameterization Solver Setup Evaluations

    coefficients_list = []
    b_spline_names_list = []
    # region Wing Parameterization Evaluation for Parameterization Solver
    section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
    # sectional_wing_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)
    sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(section_parametric_coordinates)
    sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'sectional_wing_chord_stretch':sectional_wing_chord_stretch,
        'sectional_wingspan_stretch':sectional_wing_wingspan_stretch,
        # 'sectional_wing_twist':sectional_wing_twist,
        'sectional_wing_translation_x' : sectional_wing_translation_x, 
        'sectional_wing_translation_z' : sectional_wing_translation_z,
                            }

    wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
    # coefficients_list.append(wing_coefficients)
    # b_spline_names_list.append(wing.b_spline_names)
    geometry.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)
    # geometry.plot()
    # endregion

    # region Horizontal Stabilizer Parameterization Evaluation for Parameterization Solver
    section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
    # sectional_h_tail_twist = h_tail_twist_b_spline.evaluate(section_parametric_coordinates)
    sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
    sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'sectional_h_tail_chord_stretch':sectional_h_tail_chord_stretch,
        'sectional_h_tail_span_stretch':sectional_h_tail_span_stretch,
        # 'sectional_h_tail_twist':sectional_h_tail_twist,
        'sectional_h_tail_translation_x':sectional_h_tail_translation_x,
        'sectional_h_tail_translation_z':sectional_h_tail_translation_z
                            }

    h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    h_tail_coefficients = h_tail_ffd_block.evaluate(h_tail_ffd_block_coefficients, plot=False)

    # coefficients_list.append(h_tail_coefficients)
    # b_spline_names_list.append(h_tail.b_spline_names)
    geometry.assign_coefficients(coefficients=h_tail_coefficients, b_spline_names=h_tail.b_spline_names)
    # geometry.plot()
    # endregion

    # region Fuselage Parameterization Evaluation for Parameterization Solver
    section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {'sectional_fuselage_stretch':sectional_fuselage_stretch}

    fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    fuselage_and_nose_hub_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
    fuselage_coefficients = fuselage_and_nose_hub_coefficients['fuselage_coefficients']
    nose_hub_coefficients = fuselage_and_nose_hub_coefficients['weird_nose_hub_coefficients']

    # coefficients_list.append(fuselage_coefficients)
    # coefficients_list.append(nose_hub_coefficients)
    # b_spline_names_list.append(fuselage.b_spline_names)
    # b_spline_names_list.append(nose_hub.b_spline_names)
    geometry.assign_coefficients(coefficients=fuselage_coefficients, b_spline_names=fuselage.b_spline_names)
    geometry.assign_coefficients(coefficients=nose_hub_coefficients, b_spline_names=nose_hub.b_spline_names)
    # geometry.plot()
    # endregion


    # region Lift Rotor Evaluation for Parameterization Solver
    for component_set in lift_rotor_related_components:
        components_name = component_set[0].name[:3] + '_lift_rotor_components'
    
        component_parameterization_b_splines = lift_rotor_parameterization_objects[f'{components_name}_parameterization_b_splines']
        component_sectional_parameterization = lift_rotor_parameterization_objects[f'{components_name}_sectional_parameterization']
        component_ffd_block = lift_rotor_parameterization_objects[f'{components_name}_ffd_block']

        section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
        sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
        sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
        sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)
        sectional_stretch_y = component_parameterization_b_splines[3].evaluate(section_parametric_coordinates)

        sectional_parameters = {
            f'{components_name}_translation_x':sectional_translation_x,
            f'{components_name}_translation_y':sectional_translation_y,
            f'{components_name}_translation_z':sectional_translation_z,
            f'{components_name}_stretch_y':sectional_stretch_y,
                                }
        
        component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
        component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)
        disk = component_set[0]
        blade_1 = component_set[1]
        blade_2 = component_set[2]
        hub = component_set[3]

        # coefficients_list.append(component_coefficients[disk.name+'_coefficients'])
        # coefficients_list.append(component_coefficients[blade_1.name+'_coefficients'])
        # coefficients_list.append(component_coefficients[blade_2.name+'_coefficients'])
        # coefficients_list.append(component_coefficients[hub.name+'_coefficients'])
        
        # b_spline_names_list.append(disk.b_spline_names)
        # b_spline_names_list.append(blade_1.b_spline_names)
        # b_spline_names_list.append(blade_2.b_spline_names)
        # b_spline_names_list.append(hub.b_spline_names)

        geometry.assign_coefficients(coefficients=component_coefficients[disk.name+'_coefficients'], b_spline_names=disk.b_spline_names)
        geometry.assign_coefficients(coefficients=component_coefficients[blade_1.name+'_coefficients'], b_spline_names=blade_1.b_spline_names)
        geometry.assign_coefficients(coefficients=component_coefficients[blade_2.name+'_coefficients'], b_spline_names=blade_2.b_spline_names)
        geometry.assign_coefficients(coefficients=component_coefficients[hub.name+'_coefficients'], b_spline_names=hub.b_spline_names)

    # endregion

    # region booms
    for component in boom_components:
        components_name = component.name
        component_parameterization_b_splines = boom_parameterization_objects[f'{components_name}_parameterization_b_splines']
        component_sectional_parameterization = boom_parameterization_objects[f'{components_name}_sectional_parameterization']
        component_ffd_block = boom_parameterization_objects[f'{components_name}_ffd_block']

        section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
        sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
        sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
        sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)

        sectional_parameters = {
            f'{components_name}_translation_x':sectional_translation_x,
            f'{components_name}_translation_y':sectional_translation_y,
            f'{components_name}_translation_z':sectional_translation_z,
                                }
        
        component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
        component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)

        # coefficients_list.append(component_coefficients)
        # b_spline_names_list.append(component.b_spline_names)

        geometry.assign_coefficients(coefficients=component_coefficients, b_spline_names=component.b_spline_names)
    # endregion

    # region pusher
    component_set = pp_components
    components_name = component_set[0].name[:3] + '_pusher_rotor_components'
    component_parameterization_b_splines = pusher_parameterization_objects[f'{components_name}_parameterization_b_splines']
    component_sectional_parameterization = pusher_parameterization_objects[f'{components_name}_sectional_parameterization']
    component_ffd_block = pusher_parameterization_objects[f'{components_name}_ffd_block']

    section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
    sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
    sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)

    sectional_parameters = {
        f'{components_name}_translation_x':sectional_translation_x,
        f'{components_name}_translation_y':sectional_translation_y,
        f'{components_name}_translation_z':sectional_translation_z,
                            }

    component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)
    disk = component_set[0]
    blade_1 = component_set[1]
    blade_2 = component_set[2]
    blade_3 = component_set[3]
    blade_4 = component_set[4]
    hub = component_set[5]

    # coefficients_list.append(component_coefficients[disk.name+'_coefficients'])
    # coefficients_list.append(component_coefficients[blade_1.name+'_coefficients'])
    # coefficients_list.append(component_coefficients[blade_2.name+'_coefficients'])
    # coefficients_list.append(component_coefficients[blade_3.name+'_coefficients'])
    # coefficients_list.append(component_coefficients[blade_4.name+'_coefficients'])
    # coefficients_list.append(component_coefficients[hub.name+'_coefficients'])
    # b_spline_names_list.append(disk.b_spline_names)
    # b_spline_names_list.append(blade_1.b_spline_names)
    # b_spline_names_list.append(blade_2.b_spline_names)
    # b_spline_names_list.append(blade_3.b_spline_names)
    # b_spline_names_list.append(blade_4.b_spline_names)
    # b_spline_names_list.append(hub.b_spline_names)

    geometry.assign_coefficients(coefficients=component_coefficients[disk.name+'_coefficients'], b_spline_names=disk.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_1.name+'_coefficients'], b_spline_names=blade_1.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_2.name+'_coefficients'], b_spline_names=blade_2.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_3.name+'_coefficients'], b_spline_names=blade_3.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_4.name+'_coefficients'], b_spline_names=blade_4.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[hub.name+'_coefficients'], b_spline_names=hub.b_spline_names)
    # endregion

    # region v-tail Parameterization Evaluation for Parameterization Solver
    components_name = v_tail.name
    component_parameterization_b_splines = vtail_parameterization_objects[f'{components_name}_parameterization_b_splines']
    component_sectional_parameterization = vtail_parameterization_objects[f'{components_name}_sectional_parameterization']
    component_ffd_block = vtail_parameterization_objects[f'{components_name}_ffd_block']

    section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
    sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
    sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)

    sectional_parameters = {
        f'{components_name}_translation_x':sectional_translation_x,
        f'{components_name}_translation_y':sectional_translation_y,
        f'{components_name}_translation_z':sectional_translation_z,
                            }

    component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)
    # coefficients_list.append(component_coefficients)
    # b_spline_names_list.append(v_tail.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients, b_spline_names=v_tail.b_spline_names)
    # endregion

    # geometry.assign_coefficients(coefficients=coefficients_list, b_spline_names=b_spline_names_list)

    # endregion

    # region Defining/Declaring Parameterization Solver Inputs
    parameterization_inputs = {}

    # region wing design parameterization inputs
    wingspan = m3l.norm(geometry.evaluate(wing_le_right) - geometry.evaluate(wing_le_left))
    root_chord = m3l.norm(geometry.evaluate(wing_te_center) - geometry.evaluate(wing_le_center))
    tip_chord_left = m3l.norm(geometry.evaluate(wing_te_left) - geometry.evaluate(wing_le_left))
    tip_chord_right = m3l.norm(geometry.evaluate(wing_te_right) - geometry.evaluate(wing_le_right))

    parameterization_solver.declare_input(name='wingspan', input=wingspan)
    parameterization_solver.declare_input(name='root_chord', input=root_chord)
    parameterization_solver.declare_input(name='tip_chord_left', input=tip_chord_left)
    parameterization_solver.declare_input(name='tip_chord_right', input=tip_chord_right)

    parameterization_inputs['wingspan'] = m3l.Variable(name='wingspan', shape=(1,), value=np.array([60.]), dv_flag=True)
    parameterization_inputs['root_chord'] = m3l.Variable(name='root_chord', shape=(1,), value=np.array([5.]), dv_flag=True)
    parameterization_inputs['tip_chord_left'] = m3l.Variable(name='tip_chord_left', shape=(1,), value=np.array([1.]))
    parameterization_inputs['tip_chord_right'] = m3l.Variable(name='tip_chord_right', shape=(1,), value=np.array([1.]))
    # endregion

    # region h_tail design parameterization inputs
    h_tail_span = m3l.norm(geometry.evaluate(tail_le_right) - geometry.evaluate(tail_le_left))
    h_tail_root_chord = m3l.norm(geometry.evaluate(tail_te_center) - geometry.evaluate(tail_le_center))
    h_tail_tip_chord_left = m3l.norm(geometry.evaluate(tail_te_left) - geometry.evaluate(tail_le_left))
    h_tail_tip_chord_right = m3l.norm(geometry.evaluate(tail_te_right) - geometry.evaluate(tail_le_right))

    parameterization_solver.declare_input(name='h_tail_span', input=h_tail_span)
    parameterization_solver.declare_input(name='h_tail_root_chord', input=h_tail_root_chord)
    parameterization_solver.declare_input(name='h_tail_tip_chord_left', input=h_tail_tip_chord_left)
    parameterization_solver.declare_input(name='h_tail_tip_chord_right', input=h_tail_tip_chord_right)

    parameterization_inputs['h_tail_span'] = m3l.Variable(name='h_tail_span', shape=(1,), value=np.array([12.]), dv_flag=True)
    parameterization_inputs['h_tail_root_chord'] = m3l.Variable(name='h_tail_root_chord', shape=(1,), value=np.array([3.]), dv_flag=True)
    parameterization_inputs['h_tail_tip_chord_left'] = m3l.Variable(name='h_tail_tip_chord_left', shape=(1,), value=np.array([2.]))
    parameterization_inputs['h_tail_tip_chord_right'] = m3l.Variable(name='h_tail_tip_chord_right', shape=(1,), value=np.array([2.]))
    # endregion

    # region tail moment arm inputs
    tail_moment_arm = m3l.norm(geometry.evaluate(tail_qc) - geometry.evaluate(wing_qc))
    # tail_moment_arm = m3l.norm(geometry.evaluate(fuselage_tail_te_center) - geometry.evaluate(fuselage_wing_te_center))

    wing_fuselage_connection = geometry.evaluate(wing_te_center) - geometry.evaluate(fuselage_wing_te_center)
    h_tail_fuselage_connection = geometry.evaluate(tail_te_center) - geometry.evaluate(fuselage_tail_te_center)

    parameterization_solver.declare_input(name='tail_moment_arm', input=tail_moment_arm)
    parameterization_solver.declare_input(name='wing_to_fuselage_connection', input=wing_fuselage_connection)
    parameterization_solver.declare_input(name='h_tail_to_fuselage_connection', input=h_tail_fuselage_connection)

    parameterization_inputs['tail_moment_arm'] = m3l.Variable(name='tail_moment_arm', shape=(1,), value=np.array([25.]), dv_flag=True)
    parameterization_inputs['wing_to_fuselage_connection'] = m3l.Variable(name='wing_to_fuselage_connection', shape=(3,), value=wing_fuselage_connection.value)
    parameterization_inputs['h_tail_to_fuselage_connection'] = m3l.Variable(name='h_tail_to_fuselage_connection', shape=(3,), value=h_tail_fuselage_connection.value)
    # endregion

    # region v-tail inputs
    vtail_parametric = geometry.evaluate(v_tail.project(np.array([30.543, 0., 8.231])))
    vtail_fuselage_connection = geometry.evaluate(fueslage_rear_points_parametric) - vtail_parametric
    parameterization_solver.declare_input(name='vtail_fuselage_connection', input=vtail_fuselage_connection)
    parameterization_inputs['vtail_fuselage_connection'] = m3l.Variable(name='vtail_fuselage_connection', shape=(3,), value=vtail_fuselage_connection.value)
    # endregion

    # region lift + pusher rotor parameterization inputs
    pusher_fuselage_connection = geometry.evaluate(fueslage_rear_points_parametric) - geometry.evaluate(fuselage_rear_point_on_pusher_disk_parametric)
    parameterization_solver.declare_input(name='fuselage_pusher_connection', input=pusher_fuselage_connection)
    parameterization_inputs['fuselage_pusher_connection'] = m3l.Variable(name='fuselage_pusher_connection', shape=(3,), value=pusher_fuselage_connection.value)

    flo_radius = fro_radius = front_outer_radius = m3l.Variable(name='front_outer_radius', shape=(1, ), value=10/2, dv_flag=rotor_geom_dv, lower=5/2, upper=15/2, scaler=1e-1)
    fli_radius = fri_radius = front_inner_radius = m3l.Variable(name='front_inner_radius', shape=(1, ), value=10/2, dv_flag=rotor_geom_dv, lower=5/2, upper=15/2, scaler=1e-1)
    rlo_radius = rro_radius = rear_outer_radius = m3l.Variable(name='rear_outer_radius',  shape=(1, ),value=10/2, dv_flag=rotor_geom_dv, lower=5/2, upper=15/2, scaler=1e-1)
    rli_radius = rri_radius = rear_inner_radius = m3l.Variable(name='rear_inner_radius', shape=(1, ), value=10/2, dv_flag=rotor_geom_dv, lower=5/2, upper=15/2, scaler=1e-1)
    dv_radius_list = [rlo_radius, rli_radius, rri_radius, rro_radius, flo_radius, fli_radius, fri_radius, fro_radius]

    disk_centers = [rlo_disk_center, rli_disk_center, rri_disk_center, rro_disk_center, flo_disk_center, fli_disk_center, fri_disk_center, fro_disk_center]
    disk_centers_on_wing = [rlo_disk_center_on_wing, rli_disk_center_on_wing, rri_disk_center_on_wing, rro_disk_center_on_wing, flo_disk_center_on_wing, 
                            fli_disk_center_on_wing, fri_disk_center_on_wing, fro_disk_center_on_wing]
    boom_points = [boom_rlo, boom_rli, boom_rri, boom_rro, boom_flo, boom_fli, boom_fri, boom_fro]
    rotor_prefixes = ['rlo', 'rli', 'rri', 'rro', 'flo', 'fli', 'fri', 'fro']

    rlo_mesh = rlo_mesh.update(geometry=geometry)
    rli_mesh = rli_mesh.update(geometry=geometry)
    rri_mesh = rri_mesh.update(geometry=geometry)
    rro_mesh = rro_mesh.update(geometry=geometry)
    flo_mesh = flo_mesh.update(geometry=geometry)
    fli_mesh = fli_mesh.update(geometry=geometry)
    fri_mesh = fri_mesh.update(geometry=geometry)
    fro_mesh = fro_mesh.update(geometry=geometry)

    radius_1_list = [rlo_mesh.radius, rli_mesh.radius, rri_mesh.radius, rro_mesh.radius,
                    flo_mesh.radius, fli_mesh.radius, fri_mesh.radius, fro_mesh.radius]

    radius_2_list = [rlo_mesh._radius_2, rli_mesh._radius_2, rri_mesh._radius_2, rro_mesh._radius_2,
                    flo_mesh._radius_2, fli_mesh._radius_2, fri_mesh._radius_2, fro_mesh._radius_2]

    for i in range(len(disk_centers)):
        disk_connection = geometry.evaluate(disk_centers[i]) - geometry.evaluate(disk_centers_on_wing[i])
        boom_connection = geometry.evaluate(disk_centers[i]) - geometry.evaluate(boom_points[i])
        
        parameterization_solver.declare_input(name=rotor_prefixes[i]+'_lift_rotor_connection', input=disk_connection)
        parameterization_solver.declare_input(name=rotor_prefixes[i]+'_wing_boom_connection', input=boom_connection)
        parameterization_solver.declare_input(name=rotor_prefixes[i]+'_r1', input=radius_1_list[i])
        parameterization_solver.declare_input(name=rotor_prefixes[i]+'_r2', input=radius_2_list[i])
        
        parameterization_inputs[rotor_prefixes[i]+'_lift_rotor_connection'] = m3l.Variable(name=rotor_prefixes[i]+'_lift_rotor_connection', shape=(3,), value=disk_connection.value)
        parameterization_inputs[rotor_prefixes[i]+'_wing_boom_connection'] = m3l.Variable(name=rotor_prefixes[i]+'_wing_boom_connection', shape=(3,), value=boom_connection.value)
        parameterization_inputs[rotor_prefixes[i]+'_r1'] = m3l.Variable(name=rotor_prefixes[i]+'_r1', shape=(1,), value=3.5, dv_flag=True)
        parameterization_inputs[rotor_prefixes[i]+'_r2'] = m3l.Variable(name=rotor_prefixes[i]+'_r2', shape=(1,), value=3.5, dv_flag=True)
        # parameterization_inputs[rotor_prefixes[i]+'_r1'] = dv_radius_list[i]
        # parameterization_inputs[rotor_prefixes[i]+'_r2'] = dv_radius_list[i]

    # endregion


    # endregion

    # region Parameterization Evaluation

    parameterization_solver_states = parameterization_solver.evaluate(parameterization_inputs)

    # region Wing Parameterization Evaluation for Parameterization Solver
    wing_chord_stretch_b_spline.coefficients = parameterization_solver_states['wing_chord_stretch_coefficients']
    wing_wingspan_stretch_b_spline.coefficients = parameterization_solver_states['wing_wingspan_stretch_coefficients']
    wing_translation_x_b_spline.coefficients = parameterization_solver_states['wing_translation_x_coefficients']
    wing_translation_z_b_spline.coefficients = parameterization_solver_states['wing_translation_z_coefficients']

    section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
    sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(section_parametric_coordinates)
    sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(section_parametric_coordinates)


    sectional_parameters = {
        'sectional_wing_chord_stretch':sectional_wing_chord_stretch,
        'sectional_wingspan_stretch':sectional_wing_wingspan_stretch,
        'sectional_wing_translation_x':sectional_wing_translation_x,
        'sectional_wing_translation_z':sectional_wing_translation_z,
                            }

    wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
    geometry.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)
    # geometry.plot()
    # endregion

    # region Horizontal Stabilizer Parameterization Evaluation for Parameterization Solver
    h_tail_chord_stretch_b_spline.coefficients = parameterization_solver_states['h_tail_chord_stretch_coefficients']
    h_tail_span_stretch_b_spline.coefficients = parameterization_solver_states['h_tail_span_stretch_coefficients']
    h_tail_translation_x_b_spline.coefficients = parameterization_solver_states['h_tail_translation_x_coefficients']
    h_tail_translation_z_b_spline.coefficients = parameterization_solver_states['h_tail_translation_z_coefficients']

    section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
    sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
    sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

    h_tail_twist_coefficients = m3l.Variable(name='h_tail_twist_coefficients', shape=(5,), value=np.array([0., 0., 0., 0., 0.]))
    h_tail_twist_b_spline = bsp.BSpline(name='h_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                            coefficients=h_tail_twist_coefficients, num_physical_dimensions=1)
    sectional_h_tail_twist = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'sectional_h_tail_chord_stretch':sectional_h_tail_chord_stretch,
        'sectional_h_tail_span_stretch':sectional_h_tail_span_stretch,
        'sectional_h_tail_twist':sectional_h_tail_twist,
        'sectional_h_tail_translation_x':sectional_h_tail_translation_x,
        'sectional_h_tail_translation_z':sectional_h_tail_translation_z
                            }

    h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    h_tail_coefficients = h_tail_ffd_block.evaluate(h_tail_ffd_block_coefficients, plot=False)
    geometry.assign_coefficients(coefficients=h_tail_coefficients, b_spline_names=h_tail.b_spline_names)
    # geometry.plot()
    # endregion

    # region Fuselage Parameterization Evaluation for Parameterization Solver
    section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    fuselage_stretch_b_spline.coefficients = parameterization_solver_states['fuselage_stretch_coefficients']
    sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {'sectional_fuselage_stretch':sectional_fuselage_stretch}

    fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    fuselage_and_nose_hub_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
    fuselage_coefficients = fuselage_and_nose_hub_coefficients['fuselage_coefficients']
    nose_hub_coefficients = fuselage_and_nose_hub_coefficients['weird_nose_hub_coefficients']
    geometry.assign_coefficients(coefficients=fuselage_coefficients, b_spline_names=fuselage.b_spline_names)
    geometry.assign_coefficients(coefficients=nose_hub_coefficients, b_spline_names=nose_hub.b_spline_names)
    # geometry.plot()
    # endregion

    # region Lift Rotor Evaluation for Parameterization Solver
    for component_set in lift_rotor_related_components:
        components_name = component_set[0].name[:3] + '_lift_rotor_components'

        component_parameterization_b_splines = lift_rotor_parameterization_objects[f'{components_name}_parameterization_b_splines']
        component_parameterization_b_splines[0].coefficients = parameterization_solver_states[f'{components_name}_translation_x_coefficients']
        component_parameterization_b_splines[1].coefficients = parameterization_solver_states[f'{components_name}_translation_y_coefficients']
        component_parameterization_b_splines[2].coefficients = parameterization_solver_states[f'{components_name}_translation_z_coefficients']
        component_parameterization_b_splines[3].coefficients = parameterization_solver_states[f'{components_name}_stretch_y_coefficients']
        component_sectional_parameterization = lift_rotor_parameterization_objects[f'{components_name}_sectional_parameterization']
        component_ffd_block = lift_rotor_parameterization_objects[f'{components_name}_ffd_block']

        section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
        sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
        sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
        sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)
        sectional_stretch_y = component_parameterization_b_splines[3].evaluate(section_parametric_coordinates)

        sectional_parameters = {
            f'{components_name}_translation_x':sectional_translation_x,
            f'{components_name}_translation_y':sectional_translation_y,
            f'{components_name}_translation_z':sectional_translation_z,
            f'{components_name}_stretch_y':sectional_stretch_y,
                                }
        
        component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
        component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)
        disk = component_set[0]
        blade_1 = component_set[1]
        blade_2 = component_set[2]
        hub = component_set[3]
        # component_names = disk.b_spline_names + blade_1.b_spline_names + blade_2.b_spline_names + hub.b_spline_names
        # component_names = [disk.b_spline_names, blade_1.b_spline_names,  blade_2.b_spline_names,  hub.b_spline_names]
        # component_coefficients_list = list(component_coefficients.values())
        # component_coefficients = [component_coefficients_list[0], component_coefficients_list[1], component_coefficients_list[2], 
        #                           component_coefficients_list[3]]
        # component_coefficients_list = []
        # for i in range(len(component_coefficients)):
        #     component_coefficients_list.extend(list(component_coefficients.values())[i])    
        # geometry.assign_coefficients(coefficients=component_coefficients_list, b_spline_names=component_names)
        geometry.assign_coefficients(coefficients=component_coefficients[disk.name+'_coefficients'], b_spline_names=disk.b_spline_names)
        geometry.assign_coefficients(coefficients=component_coefficients[blade_1.name+'_coefficients'], b_spline_names=blade_1.b_spline_names)
        geometry.assign_coefficients(coefficients=component_coefficients[blade_2.name+'_coefficients'], b_spline_names=blade_2.b_spline_names)
        geometry.assign_coefficients(coefficients=component_coefficients[hub.name+'_coefficients'], b_spline_names=hub.b_spline_names)

    # geometry.plot()
    # endregion

    # region booms
    for component in boom_components:
        components_name = component.name
        component_parameterization_b_splines = boom_parameterization_objects[f'{components_name}_parameterization_b_splines']
        component_parameterization_b_splines[0].coefficients = parameterization_solver_states[f'{components_name}_translation_x_coefficients']
        component_parameterization_b_splines[1].coefficients = parameterization_solver_states[f'{components_name}_translation_y_coefficients']
        component_parameterization_b_splines[2].coefficients = parameterization_solver_states[f'{components_name}_translation_z_coefficients']
        component_sectional_parameterization = boom_parameterization_objects[f'{components_name}_sectional_parameterization']
        component_ffd_block = boom_parameterization_objects[f'{components_name}_ffd_block']

        section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
        sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
        sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
        sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)

        sectional_parameters = {
            f'{components_name}_translation_x':sectional_translation_x,
            f'{components_name}_translation_y':sectional_translation_y,
            f'{components_name}_translation_z':sectional_translation_z,
                                }
        
        component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
        component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)

        geometry.assign_coefficients(coefficients=component_coefficients, b_spline_names=component.b_spline_names)
    # endregion

    # region pusher
    component_set = pp_components
    components_name = component_set[0].name[:3] + '_pusher_rotor_components'
    component_parameterization_b_splines = pusher_parameterization_objects[f'{components_name}_parameterization_b_splines']
    component_parameterization_b_splines[0].coefficients = parameterization_solver_states[f'{components_name}_translation_x_coefficients']
    component_parameterization_b_splines[1].coefficients = parameterization_solver_states[f'{components_name}_translation_y_coefficients']
    component_parameterization_b_splines[2].coefficients = parameterization_solver_states[f'{components_name}_translation_z_coefficients']
    component_sectional_parameterization = pusher_parameterization_objects[f'{components_name}_sectional_parameterization']
    component_ffd_block = pusher_parameterization_objects[f'{components_name}_ffd_block']

    section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
    sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
    sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)

    sectional_parameters = {
        f'{components_name}_translation_x':sectional_translation_x,
        f'{components_name}_translation_y':sectional_translation_y,
        f'{components_name}_translation_z':sectional_translation_z,
                            }

    component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)
    disk = component_set[0]
    blade_1 = component_set[1]
    blade_2 = component_set[2]
    blade_3 = component_set[3]
    blade_4 = component_set[4]
    hub = component_set[5]

    geometry.assign_coefficients(coefficients=component_coefficients[disk.name+'_coefficients'], b_spline_names=disk.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_1.name+'_coefficients'], b_spline_names=blade_1.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_2.name+'_coefficients'], b_spline_names=blade_2.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_3.name+'_coefficients'], b_spline_names=blade_3.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[blade_4.name+'_coefficients'], b_spline_names=blade_4.b_spline_names)
    geometry.assign_coefficients(coefficients=component_coefficients[hub.name+'_coefficients'], b_spline_names=hub.b_spline_names)
    # endregion

    # region v-tail Parameterization Evaluation for Parameterization Solver
    components_name = v_tail.name
    component_parameterization_b_splines = vtail_parameterization_objects[f'{components_name}_parameterization_b_splines']
    component_parameterization_b_splines[0].coefficients = parameterization_solver_states[f'{components_name}_translation_x_coefficients']
    component_parameterization_b_splines[1].coefficients = parameterization_solver_states[f'{components_name}_translation_y_coefficients']
    component_parameterization_b_splines[2].coefficients = parameterization_solver_states[f'{components_name}_translation_z_coefficients']
    component_sectional_parameterization = vtail_parameterization_objects[f'{components_name}_sectional_parameterization']
    component_ffd_block = vtail_parameterization_objects[f'{components_name}_ffd_block']

    section_parametric_coordinates = np.linspace(0., 1., component_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_translation_x = component_parameterization_b_splines[0].evaluate(section_parametric_coordinates)
    sectional_translation_y = component_parameterization_b_splines[1].evaluate(section_parametric_coordinates)
    sectional_translation_z = component_parameterization_b_splines[2].evaluate(section_parametric_coordinates)

    sectional_parameters = {
        f'{components_name}_translation_x':sectional_translation_x,
        f'{components_name}_translation_y':sectional_translation_y,
        f'{components_name}_translation_z':sectional_translation_z,
                            }

    component_ffd_block_coefficients = component_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    component_coefficients = component_ffd_block.evaluate(component_ffd_block_coefficients, plot=False)
    geometry.assign_coefficients(coefficients=component_coefficients, b_spline_names=v_tail.b_spline_names)
    # endregion

    # endregion
    # geometry.plot()

    # region Mesh Evaluation/Update
    wing_meshes = wing_meshes.update(geometry=geometry)
    tail_meshes = tail_meshes.update(geometry=geometry)
    # geometry.plot_meshes([wing_meshes.vlm_mesh, tail_meshes.vlm_mesh])

    pp_mesh = pp_mesh.update(geometry=geometry)
    rlo_mesh = rlo_mesh.update(geometry=geometry)
    rro_mesh = rro_mesh.update(geometry=geometry)
    flo_mesh = flo_mesh.update(geometry=geometry)
    fro_mesh = fro_mesh.update(geometry=geometry)
    rli_mesh = rli_mesh.update(geometry=geometry)
    rri_mesh = rri_mesh.update(geometry=geometry)
    fli_mesh = fli_mesh.update(geometry=geometry)
    fri_mesh = fri_mesh.update(geometry=geometry)


    lift_rotor_mesh_list = [rlo_mesh, rli_mesh, rri_mesh, rro_mesh, flo_mesh, fli_mesh, fri_mesh, fro_mesh]
    lift_rotor_mesh_list_oei_flo = [rlo_mesh, rli_mesh, rri_mesh, rro_mesh, fli_mesh, fri_mesh, fro_mesh]
    lift_rotor_mesh_list_oei_fli = [rlo_mesh, rli_mesh, rri_mesh, rro_mesh, flo_mesh, fri_mesh, fro_mesh]
    lift_rotor_mesh_list_oei_rlo = [rli_mesh, rri_mesh, rro_mesh, flo_mesh, fli_mesh, fri_mesh, fro_mesh]
    lift_rotor_mesh_list_oei_rli = [rlo_mesh, rri_mesh, rro_mesh, flo_mesh, fli_mesh, fri_mesh, fro_mesh]

    lift_rotor_origin_list = [mesh.thrust_origin for mesh in lift_rotor_mesh_list]
    all_rotor_origin_list = lift_rotor_origin_list + [pp_mesh.thrust_origin]

    # dummy_objective = m3l.norm(vlm_mesh.reshape((-1,)))
    # m3l_model.register_output(vlm_mesh)
    # m3l_model.register_output(dummy_objective)
    # endregion

# region assign blade twist and chord coefficients
# chord/ twist profiles
chord_cps_numpy = np.array([0.122222, 0.213889, 0.188426, 0.050926]) * 5
chord_cps_numpy_pusher = np.array([0.122222, 0.213889, 0.188426, 0.050926]) * 4.5

twist_cps_numpy = np.deg2rad(np.linspace(35.000000, 15.000000, 4))
twist_cps_numpy_pusher = np.deg2rad(np.linspace(55.000000, 10.000000, 4))

lower_twist_hover = np.deg2rad(1)
upper_twist_hover = np.deg2rad(60)
lower_twist_pusher = np.deg2rad(5)
upper_twist_pusher = np.deg2rad(85)

flo_blade_chord_bsp_cps = fro_blade_chord_bsp_cps = system_model.create_input('front_outer_blade_chord_cps', val=chord_cps_numpy, dv_flag=rotor_geom_dv, lower=0.1, upper=1.1)
fli_blade_chord_bsp_cps = fri_blade_chord_bsp_cps = system_model.create_input('front_inner_blade_chord_cps', val=chord_cps_numpy, dv_flag=rotor_geom_dv, lower=0.1, upper=1.1)
rlo_blade_chord_bsp_cps = rro_blade_chord_bsp_cps = system_model.create_input('rear_outer_blade_chord_cps', val=chord_cps_numpy, dv_flag=rotor_geom_dv, lower=0.1, upper=1.1)
rli_blade_chord_bsp_cps = rri_blade_chord_bsp_cps = system_model.create_input('rear_inner_blade_chord_cps', val=chord_cps_numpy, dv_flag=rotor_geom_dv, lower=0.1, upper=1.1)

flo_blade_twist_bsp_cps = fro_blade_twist_bsp_cps = system_model.create_input('front_outer_blade_twist_cps', val=twist_cps_numpy, dv_flag=rotor_geom_dv, lower=lower_twist_hover, upper=upper_twist_hover)
fli_blade_twist_bsp_cps = fri_blade_twist_bsp_cps = system_model.create_input('front_inner_blade_twist_cps', val=twist_cps_numpy, dv_flag=rotor_geom_dv, lower=lower_twist_hover, upper=upper_twist_hover)
rlo_blade_twist_bsp_cps = rro_blade_twist_bsp_cps = system_model.create_input('rear_outer_blade_twist_cps', val=twist_cps_numpy, dv_flag=rotor_geom_dv, lower=lower_twist_hover, upper=upper_twist_hover)
rli_blade_twist_bsp_cps = rri_blade_twist_bsp_cps = system_model.create_input('rear_inner_blade_twist_cps', val=twist_cps_numpy, dv_flag=rotor_geom_dv, lower=lower_twist_hover, upper=upper_twist_hover)

pusher_chord_bsp_cps = system_model.create_input('pusher_chord_bsp_cps', val=chord_cps_numpy_pusher, dv_flag=rotor_geom_dv, lower=0.08, upper=1.4)
pusher_twist_bsp_cps = system_model.create_input('pusher_twist_bsp_cps', val=twist_cps_numpy_pusher, dv_flag=rotor_geom_dv, lower=lower_twist_pusher, upper=upper_twist_pusher)

rlo_mesh.chord_cps = rlo_blade_chord_bsp_cps
rlo_mesh.twist_cps = rlo_blade_twist_bsp_cps

rli_mesh.chord_cps = rli_blade_chord_bsp_cps
rli_mesh.twist_cps = rli_blade_twist_bsp_cps

rri_mesh.chord_cps = rri_blade_chord_bsp_cps
rri_mesh.twist_cps = rri_blade_twist_bsp_cps

rro_mesh.chord_cps = rro_blade_chord_bsp_cps
rro_mesh.twist_cps = rro_blade_twist_bsp_cps

flo_mesh.chord_cps = flo_blade_chord_bsp_cps
flo_mesh.twist_cps = flo_blade_twist_bsp_cps

fli_mesh.chord_cps = fli_blade_chord_bsp_cps
fli_mesh.twist_cps = fli_blade_twist_bsp_cps

fri_mesh.chord_cps = fri_blade_chord_bsp_cps
fri_mesh.twist_cps = fri_blade_twist_bsp_cps

fro_mesh.chord_cps = fro_blade_chord_bsp_cps
fro_mesh.twist_cps = fro_blade_twist_bsp_cps

pp_mesh.chord_cps = pusher_chord_bsp_cps
pp_mesh.twist_cps = pusher_twist_bsp_cps
# endregion

# region compute component surface areas and make drag components
component_list = [rlo_boom, rlo_hub, fuselage, wing, h_tail, v_tail, pp_hub, rlo_blade_1]
surface_area_list = compute_component_surface_area(
    component_list=component_list,
    geometry=geometry,
    parametric_mesh_grid_num=20,
    plot=False,
)

# wing reference area (~total wetted area/ 2.1)
S_ref = surface_area_list[3] / 2.1

# Wing aspect ratio
wingspan = m3l.norm(geometry.evaluate(wing_le_right) - geometry.evaluate(wing_le_left))
wing_AR = wingspan**2 / S_ref

# fuselage length
fuselage_l1 = geometry.evaluate(fuselage_l1_parametric).reshape((-1, 3))
fuselage_l2 = geometry.evaluate(fuselage_l2_parametric).reshape((-1, 3))
fuselage_length = m3l.norm(fuselage_l2-fuselage_l1)


fuselage_d1 = geometry.evaluate(fuselage_d1_parametric).reshape((-1, 3))
fuselage_d2=  geometry.evaluate(fuselage_d2_parametric).reshape((-1, 3))
fuselage_diameter = m3l.norm(fuselage_d2-fuselage_d1)

fuselage_drag_comp = DragComponent(
    component_type='fuselage',
    wetted_area=surface_area_list[2],
    characteristic_length=fuselage_length,
    characteristic_diameter=fuselage_diameter,
    Q=2.9,
)


# wing
wing_mid_le = geometry.evaluate(wing_mid_le_parametric).reshape((-1, 3))
wing_mid_te = geometry.evaluate(wing_mid_te_parametric).reshape((-1, 3))
wing_mid_chord_length = m3l.norm(wing_mid_le-wing_mid_te)

wing_le_right_mapped_array = geometry.evaluate(wing_le_right).reshape((-1, 3))
wing_le_left_mapped_array = geometry.evaluate(wing_le_left).reshape((-1, 3))
wing_span = m3l.norm(wing_le_left_mapped_array - wing_le_right_mapped_array)

wing_drag_comp = DragComponent(
    component_type='wing',
    wetted_area=surface_area_list[3],
    characteristic_length=wing_mid_chord_length,
    thickness_to_chord=0.16,
    x_cm=0.45,
    Q=1.4,
)

# h tail
h_tail_mid_le = geometry.evaluate(h_tail_mid_le_parametric).reshape((-1, 3))
h_tail_mid_te = geometry.evaluate(h_tail_mid_te_parametric).reshape((-1, 3))
h_tail_chord_length = m3l.norm(h_tail_mid_le-h_tail_mid_te)
h_tail_area = surface_area_list[4] / 2.1

h_tail_drag_comp = DragComponent(
    component_type='wing',
    wetted_area=surface_area_list[4],
    characteristic_length=h_tail_chord_length,
    thickness_to_chord=0.15,
    x_cm=0.3,
    Q=1.5,

)

# v tail
vtail_mid_le = geometry.evaluate(vtail_mid_le_parametric).reshape((-1, 3))
vtail_mid_te = geometry.evaluate(vtail_mid_te_parametric).reshape((-1, 3))
vtail_chord_length = m3l.norm(vtail_mid_le-vtail_mid_te)
v_tail_area = surface_area_list[5] / 2.

vtail_drag_comp = DragComponent(
    component_type='wing',
    wetted_area=surface_area_list[5],
    characteristic_length=vtail_chord_length,
    thickness_to_chord=0.115,
    x_cm=0.2,
    Q=1.5,
)

# boom
boom_l1 = geometry.evaluate(boom_l1_parametric).reshape((-1, 3))
boom_l2 = geometry.evaluate(boom_l2_parametric).reshape((-1, 3))
boom_length = m3l.norm(boom_l1-boom_l2)

boom_d1 = geometry.evaluate(boom_d1_parametric).reshape((-1, 3))
boom_d2 = geometry.evaluate(boom_d2_parametric).reshape((-1, 3))
boom_diameter = m3l.norm(boom_d1-boom_d2)

boom_drag_comp = DragComponent(
    component_type='boom',
    wetted_area=surface_area_list[0],
    characteristic_diameter=boom_diameter,
    characteristic_length=boom_length,
    multiplicity=8,
    Q=2.,
)

# lift hubs
hub_l1 = geometry.evaluate(hub_l1_parametric)
hub_l2 = geometry.evaluate(hub_l2_parametric)
hub_length = m3l.norm(hub_l1-hub_l2)

hub_drag_comp = DragComponent(
    component_type='nacelle',
    wetted_area=surface_area_list[1],
    characteristic_diameter=hub_length,
    characteristic_length=hub_length,
    multiplicity=8,
    Q=1,

)


# blade 
blade_tip = geometry.evaluate(blade_tip_parametric)
blade_hub = geometry.evaluate(blade_hub_parametric)

blade_length = m3l.norm(blade_tip-blade_hub)
blade_drag_comp = DragComponent(
    component_type='flat_plate',
    characteristic_length=blade_length,
    wetted_area=surface_area_list[-1],
    multiplicity=16,
    Q=2,

)

drag_comp_list = [wing_drag_comp, fuselage_drag_comp, h_tail_drag_comp, vtail_drag_comp,
                  blade_drag_comp, boom_drag_comp, hub_drag_comp]



# endregion



# m3l_model.register_output(geometry.coefficients)
# m3l_model.add_objective(dummy_objective)

# csdl_model = m3l_model.assemble()
# sim = Simulator(csdl_model)
# sim.run()
# print(geometry.coefficients.operation.name)
# print(geometry.coefficients.name)

# print(vlm_mesh.operation.name)
# print(vlm_mesh.name)
# geometry.coefficients = sim['10548_plus_10550_operation.10551']
# camber_surface = sim['10562_reshape_operation_Hryi2.10563']
# geometry.plot_meshes([camber_surface])
# geometry.plot()
# sim.check_totals(of=[vlm_mesh.operation.name + '.' + vlm_mesh.name],
#                  wrt=['root_chord', 'wingspan', 'tip_chord_left', 'tip_chord_right'])