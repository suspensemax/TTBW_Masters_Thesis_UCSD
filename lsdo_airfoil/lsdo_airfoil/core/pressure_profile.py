import m3l
import csdl
import numpy as np
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL, CpModelCSDL
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


class PressureProfile(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='pressure_operation')
        self.parameters.declare('component', default=None)
        self.parameters.declare('airfoil_name')
        self.parameters.declare('compute_control_points', default=False)
        self.parameters.declare('num_nodes', types=int, default=1)
        self.parameters.declare('use_inverse_cl_map', types=bool, default=False)
        self.m3l_var_list_cl = None
        self.m3l_var_list_re = None
    
    def compute(self) -> csdl.Model:
        num_nodes = self.parameters['num_nodes']
        airfoil_name = self.parameters['airfoil_name']
        compute_control_points = self.parameters['compute_control_points']
        use_inverse_cl_map = self.parameters['use_inverse_cl_map']
        
        csdl_model = CpModelCSDL(
            num_nodes=num_nodes,
            airfoil_name=airfoil_name,
            compute_control_points=compute_control_points,
            use_inverse_cl_map=use_inverse_cl_map,
            m3l_var_list_cl=self.m3l_var_list_cl,
            m3l_var_list_re=self.m3l_var_list_re,
        )

        return csdl_model
        
        # return super().compute()
    
    def evaluate(self, cl_list:list=[], re_list:list=[], design_condition=None) -> tuple:
        if design_condition:
            dc_name = design_condition.parameters['name']
            self.name = f'{dc_name}_airfoil_ml_model'
        else:
            self.name = 'airfoil_ml_model'
        self.arguments = {}
        self.m3l_var_list_cl = cl_list
        self.m3l_var_list_re = re_list
   

        cp_upper_list = []
        cp_lower_list = []
        cd_list = []

        counter = 0
        for cl in cl_list:
            self.arguments[cl.name] = cl
            
            Re = re_list[counter]
            self.arguments[Re.name] = Re
            counter += 1
   
            shape = cl.shape
            num_eval = shape[0] * shape[1]
            
            cp_upper = m3l.Variable(name=f'{cl.name.split("_")[0]}_cp_upper', shape=(num_eval, 100), operation=self)
            cp_lower = m3l.Variable(name=f'{cl.name.split("_")[0]}_cp_lower', shape=(num_eval, 100), operation=self)

            cd = m3l.Variable(name=f'{cl.name.split("_")[0]}_cd', shape=(num_eval, 100), operation=self)

            cp_upper_list.append(cp_upper)
            cp_lower_list.append(cp_lower)
            cd_list.append(cd)

        return cp_upper_list, cp_lower_list, cd_list
    
class NodalPressureProfile(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
    
    def compute(self):
        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']

        csdl_module = ModuleCSDL()

        for i in range(len(surface_names)):
            name = surface_names[i]
            if 'tail' in name:
                num_nodes = surface_shapes[i][1] - 1
            else:
                num_nodes = surface_shapes[i][1]
            ml_pressures_upper = csdl_module.register_module_input(f'{name.split("_")[0]}_cp_upper', shape=(num_nodes, 100))
            ml_pressures_lower = csdl_module.register_module_input(f'{name.split("_")[0]}_cp_lower', shape=(num_nodes, 100))
    
            # Identity map
            csdl_module.register_module_output(f'{name.split("_")[0]}_oml_cp_upper', ml_pressures_upper * 1)
            csdl_module.register_module_output(f'{name.split("_")[0]}_oml_cp_lower', ml_pressures_lower * 1)

        return csdl_module

    def evaluate(self, ml_pressure_upper, ml_pressure_lower, nodal_pressure_mesh, design_condition=None) -> tuple:
        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']
        
        self.nodal_forces_meshes = nodal_pressure_mesh
        if design_condition:
            dc_name = design_condition.parameters['name']
            self.name = f"{dc_name}_{''.join(surface_names)}_ml_pressure_mapping_model"
        else:
            self.name = f"{''.join(surface_names)}_ml_pressure_mapping_model"
        
        self.arguments = {}

        for i in range(len(surface_names)):
            surface_name = surface_names[i].split("_")[0]
            self.arguments[surface_name + '_cp_upper'] = ml_pressure_upper[i]
            self.arguments[surface_name + '_cp_lower'] = ml_pressure_lower[i]

        oml_pressures_upper = []
        oml_pressures_lower = []
        shapes = [(100, 24), (100, 8)]
        for i in range(len(surface_names)):
            surface_name = surface_names[i].split("_")[0]
            # shape = (surface_shapes[i][0], surface_shapes[i][1])
            shape = shapes[i]
            print(shape)
            oml_pressure_upper = m3l.Variable(name=f'{surface_name}_oml_cp_upper', shape=shape, operation=self)
            oml_pressure_lower = m3l.Variable(name=f'{surface_name}_oml_cp_lower', shape=shape, operation=self)

            oml_pressures_upper.append(oml_pressure_upper)
            oml_pressures_lower.append(oml_pressure_lower)
        
        # exit()
        return oml_pressures_upper, oml_pressures_lower