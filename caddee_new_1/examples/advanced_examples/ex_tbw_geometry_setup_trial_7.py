"Tbw_images --> Output file is on Output folder"

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
geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_40.stp', parallelize=True)
# geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_45.stp', parallelize=True)
# geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_51.stp', parallelize=True)
# geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_63.stp', parallelize=True)
# geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_69.stp', parallelize=True)
# geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_76.stp', parallelize=True)
# geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_80.stp', parallelize=True)
# geometry.refit(parallelize=True, order=(4, 4))
geometry_plot = True
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
# components_plot = True
if components_plot:
    strut.plot()

component_list = [wing, htail, strut]
surface_area_list = compute_component_surface_area(
    component_list=component_list,
    geometry=geometry,
    parametric_mesh_grid_num=20,
    plot=False,
)

strut_area = surface_area_list[2] / 2.07
print('strut area', strut_area.value)

# exit()
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


component_list = [Fuselage, wing, htail, vtail, jury, gearpod, strut]
surface_area_list = compute_component_surface_area(
    component_list=component_list,
    geometry=geometry,
    parametric_mesh_grid_num=20,
    plot=False,
)

S_ref_area = surface_area_list[1] / 1.969576502