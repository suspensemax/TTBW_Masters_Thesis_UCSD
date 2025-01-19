import csdl
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp
from VAST.core.fluid_problem import FluidProblem
import m3l
from typing import List
import numpy as np


class ViscousCorrectionModel(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int, default=1)

    def compute(self):
        '''
        Creates a CSDL model to compute the solver outputs.

        Returns
        -------
        csdl_model : csdl.Model
            The csdl model which computes the outputs (the normal solver)
        '''


        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]

        csdl_model = csdl.Model()

        self.F = []
        self.M = []

        submodule = ViscousCorrectionCSDL(
            module=self,
            surface_names=surface_names,  
            surface_shapes=surface_shapes)

        csdl_model.add(submodule,'viscous_correction_ml')
    
        return csdl_model      

    def compute_derivates(self,inputs,derivatives):
        pass

    def evaluate(self, ac_states, forces, cd_v, panel_area, moment_pt, evaluation_pt, design_condition=None):
        '''
        Evaluates the vast model.
        
        Parameters
        ----------
        displacements : list of m3l.Variable = None
            The forces on the mesh nodes.

        Returns
        -------
        panel_forces : m3l.Variable
            The displacements of the mesh nodes.

        '''
        # Gets information for naming/shapes
        # beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']
        
        self.arguments = {}
        if design_condition:
            self.name = f"{design_condition.parameters['name']}_{''.join(surface_names)}_viscous_correction_model"

        else:
            self.name = f"{''.join(surface_names)}_viscous_correction_model"
        # print(displacements)

        
        # print(arguments)
        # new_arguments = {**arguments, **ac_states}
        self.arguments['u'] = ac_states['u']
        self.arguments['v'] = ac_states['v']
        self.arguments['w'] = ac_states['w']
        # self.arguments['p'] = ac_states['p']
        # self.arguments['q'] = ac_states['q']
        # self.arguments['r'] = ac_states['r']
        self.arguments['theta'] = ac_states['theta']
        self.arguments['psi'] = ac_states['psi']
        self.arguments['gamma'] = ac_states['gamma']
        # self.arguments['psiw'] = ac_states['psi_w']
        if forces is not None:
            for i in range(len(surface_names)):
                surface_name = surface_names[i]
                self.arguments[f'{surface_name}_total_forces'] = forces[i]
        if cd_v is not None:
            for i in range(len(surface_names)):
                surface_name = surface_names[i]
                self.arguments[f'{surface_name}_cd_v_span'] = cd_v[i]
        if panel_area is not None:
            for i in range(len(surface_names)):
                surface_name = surface_names[i]
                self.arguments[f'{surface_name}_s_panel'] = panel_area[i]
        if moment_pt is not None:
            self.arguments['evaluation_pt'] = moment_pt[0]
        if evaluation_pt is not None:
            for i in range(len(surface_names)):
                surface_name = surface_names[i]
                self.arguments[f'{surface_name}_eval_pts_coords'] = evaluation_pt[i]

        total_force = m3l.Variable(name='F', shape=(num_nodes, 3), operation=self)
        total_moment = m3l.Variable(name='M', shape=(num_nodes, 3), operation=self)
        # return spanwise cl, forces on panels with vlm internal correction for cl0 and cdv, total force and total moment for trim
        return total_force, total_moment




class ViscousCorrectionCSDL(csdl.Model):
    """
    Computes the viscous correction to the forces and moments.
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('eval_pts_shapes',default=None)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0] 

        submodel = AdapterComp(surface_names=surface_names, surface_shapes=surface_shapes)
        self.add(submodel, 'adapter_comp')

        rho = self.declare_variable('density', shape=(num_nodes,1))
        v_inf_sq = self.declare_variable('v_inf_sq', shape=(num_nodes,1))
        alpha = self.declare_variable('alpha', shape=(num_nodes, 1))

        F_total_surface_list = []
        M_total_surface_list = []

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[1]
            ny = surface_shape[2]

            panel_size = int((surface_shape[1]-1) * (surface_shape[2]-1))
            num_span = surface_shape[2]-1

            surface_cd_v = self.declare_variable(surface_name + '_cd_v_span', shape=(num_nodes,num_span,1),val=np.random.random((num_nodes,num_span,1))*1e-2)
            panel_area = self.declare_variable(surface_name + '_s_panel', shape=(num_nodes,nx-1,ny-1))

            rho_v_sq_expand = csdl.expand(rho * v_inf_sq, (num_nodes,nx-1,num_span,1), 'il->ijkl')

            panel_drag = csdl.expand(surface_cd_v, (num_nodes,nx-1,num_span,1), 'ikl->ijkl') * csdl.reshape(panel_area,(num_nodes,nx-1,ny-1,1)) * 0.5 * rho_v_sq_expand
            panel_drag_flatten = csdl.reshape(panel_drag, (num_nodes,panel_size,1))

            self.register_output(surface_name + '_panel_drag', panel_drag_flatten)


            panel_forces_total = self.declare_variable(surface_name + '_total_forces', shape=(num_nodes,panel_size,3))

            cosa = csdl.expand(csdl.cos(alpha),(num_nodes,panel_size,1), 'il->ijl')
            sina = csdl.expand(csdl.sin(alpha),(num_nodes,panel_size,1), 'il->ijl')


            panel_forces_viscous = self.create_output(surface_name + '_total_forces_viscous', shape=(num_nodes,panel_size,3))
            panel_forces_viscous[:,:,0] = panel_forces_total[:,:,0]-panel_drag_flatten*cosa
            panel_forces_viscous[:,:,1] = panel_forces_total[:,:,1]
            panel_forces_viscous[:,:,2] = panel_forces_total[:,:,2]+panel_drag_flatten*sina

            F_total_surface = csdl.sum(panel_forces_viscous,axes=(1,))

            self.register_output(surface_name + '_F', F_total_surface)
            moment_pt = self.declare_variable('evaluation_pt',
                                                  val=np.zeros(3, ))

            evaluation_pt = self.declare_variable(surface_name+'_eval_pts_coords',shape=(num_nodes,nx-1,ny-1,3))
            evaluation_pt_exp = csdl.expand(moment_pt,(evaluation_pt.shape),'l->ijkl')          
            r_M = evaluation_pt - evaluation_pt_exp

            total_moments_surface = self.create_output(surface_names[i]+'_total_moments_surface_visvous',shape=(num_nodes,panel_size,3))
            print('panel_forces_viscous',panel_forces_viscous.shape)
            print('r_M',r_M.shape)
            total_moments_surface_temp = csdl.cross(csdl.reshape(r_M,(num_nodes,panel_size,3)), panel_forces_viscous, axis=2)
            total_moments_surface[:,:,0] = total_moments_surface_temp[:,:,0] * 0 # NOTE: hard coded
            total_moments_surface[:,:,1] = total_moments_surface_temp[:,:,1]
            total_moments_surface[:,:,2] = total_moments_surface_temp[:,:,2] * 0

            total_moments_surface_visvous = csdl.sum(total_moments_surface,axes=(1,))

            F_total_surface_list.append(F_total_surface)
            M_total_surface_list.append(total_moments_surface_visvous)
        
        F = sum(F_total_surface_list)
        M = sum(M_total_surface_list)
        self.register_output('F', F)
        self.register_output('M', M)

if __name__ == "__main__":

    # import numpy as np
    # from VAST.utils.generate_mesh import *
    # from python_csdl_backend import Simulator
    # import caddee.api as cd 

    # fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    # num_nodes=1; nx=3; ny=11

    # v_inf = np.ones((num_nodes,1))*248.136
    # theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles


    # surface_names = ['wing']
    # surface_shapes = [(num_nodes, nx, ny, 3)]
    # mesh_dict = {
    #     "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
    #     "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    # }
    # # Generate mesh of a rectangular wing
    # mesh = generate_mesh(mesh_dict)

    # ###########################################
    # # 1. Create a dummy m3l.Model()
    # ###########################################
    # dummy_model = m3l.Model()
    # # fluid_model = VASTFluidSover(fluid_problem=fluid_problem, surface_names=surface_names, surface_shapes=surface_shapes, mesh_unit='m', cl0=0.0)


    # # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    # # model_1.add(submodel, 'InputsModule')
    # fluid_model = VASTFluidSover(fluid_problem=fluid_problem,
    #                              surface_names=surface_names,
    #                              surface_shapes=surface_shapes,
    #                              input_dicts=None,)


    # ###########################################
    # # 3. set fluid_model inputs 
    # ###########################################
    # fluid_model.set_module_input('u',val=v_inf)
    # fluid_model.set_module_input('v',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('w',val=np.ones((num_nodes,1))*0)
    # fluid_model.set_module_input('p',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('q',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('r',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('phi',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('theta',val=theta)
    # fluid_model.set_module_input('psi',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('x',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('y',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('z',val=np.ones((num_nodes, 1))*1000)
    # fluid_model.set_module_input('phiw',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('gamma',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('psiw',val=np.zeros((num_nodes, 1)))
 
    # fluid_model.set_module_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    # input_dicts = {}
    # # input_dicts['v_inf'] = v_inf
    # # input_dicts['theta'] = theta
    # # input_dicts['undef_mesh'] = [np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh)]
    # # input_dicts['displacements'] = [np.zeros((num_nodes, nx, ny, 3))]

    # ###########################################
    # # 2. Create fluid_model as VASTFluidSover 
    # # (msl.explicit operation)
    # ###########################################


    # displacements = []
    # for i in range(len(surface_names)):
    #     surface_name = surface_names[i]
    #     surface_shape = surface_shapes[i]
    #     displacement = m3l.Variable(f'{surface_name}_displacements',shape=surface_shape,value=np.ones(surface_shape)*10)
    #     fluid_model.set_module_input(f'{surface_name}_displacements', val=np.ones(surface_shape)*100)
    #     displacements.append(displacement)

    # ###########################################
    # # 4. call fluid_model.evaluate to get
    # # surface panel forces
    # ###########################################
    # forces = fluid_model.evaluate(displacements)

    # ###########################################
    # # 5. register outputs to dummy_model
    # ###########################################
    # for i in range(len(surface_names)):
    #     surface_name = surface_names[i]
    #     dummy_model.register_output(forces[i])
        
    # ###########################################
    # # 6. call _assemble_csdl to get dummy_model_csdl
    # ###########################################
    # dummy_model_csdl = dummy_model._assemble_csdl()
    # ###########################################
    # # 7. use sim.run to run the csdl model
    # ###########################################    

    # sim = Simulator(dummy_model_csdl,analytics=False) # add simulator
    # sim.run()




    import numpy as np
    from VAST.utils.generate_mesh import *
    from python_csdl_backend import Simulator

    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    num_nodes=1; nx=5; ny=11

    v_inf = np.ones((num_nodes,1))*57
    theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles

    model_1 = ModuleCSDL()

    # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    # model_1.add(submodel, 'InputsModule')
    # model_1.add_design_variable('InputsModule.u')
    model_1.create_input('u', val=70, shape=(num_nodes, 1))
    model_1.add_design_variable('u', lower=50, upper=100, scaler=1e-2)

    surface_names = ['wing','tail']
    surface_shapes = [(num_nodes, nx, ny, 3),(num_nodes, nx-2, ny-2, 3)]

    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }

    mesh_dict_1 = {
        "num_y": ny-2, "num_x": nx-2, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }

    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict) 
    mesh_1 = generate_mesh(mesh_dict_1) 
    wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
    wing = model_1.create_input('tail', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh_1))

    # add VAST fluid solver

    submodel = VASTCSDL(
        fluid_problem=fluid_problem,
        surface_names=surface_names,
        surface_shapes=surface_shapes,
    )
    model_1.add(submodel, 'VASTSolverModule')
    sim = Simulator(model_1, analytics=True) # add simulator

    
    model_1.add_objective('VASTSolverModule.VLMSolverModel.VLM_outputs.LiftDrag.total_drag')
    sim.run()
    sim.check_totals()

