from VAST.core.vlm_llt.vlm_system import VLMSystem
# from VLM_package.VLM_system.vlm_system import VLMSystem

from VAST.core.submodels.output_submodels.vlm_post_processing.compute_outputs_group import Outputs
import numpy as np
import csdl


class VLM(MechanicsModel):
    def initialize(self, kwargs):
        self.num_nodes = 3
        self.parameters.declare('mesh')
        self.parameters.declare('component')
        self.model_selection = None

    def _assemble_csdl(self):
        mesh = self.parameters['mesh']
        surface_names = mesh.parameters['surface_names']
        surface_shapes = mesh.parameters['surface_shapes']

        component = self.parameters['component']
        prefix = component.parameters['name']

        num_nodes = len(self.model_selection)
        num_active_nodes = int(sum(self.model_selection))
        
        
        surface_shapes = [(num_active_nodes, ) + surface_shape for surface_shape in surface_shapes]
        eval_pts_shapes = [(num_active_nodes, x[1] - 1, x[2] - 1, 3)
                           for x in surface_shapes]

        coeffs_aoa = [(0.535, 0.091), (0.535, 0.091), (0.535, 0.091),
                      (0.535, 0.091)]
        coeffs_cd = [(0.00695, 1.297e-4, 1.466e-4),
                     (0.00695, 1.297e-4, 1.466e-4),
                     (0.00695, 1.297e-4, 1.466e-4),
                     (0.00695, 1.297e-4, 1.466e-4)]
        
        # print(self.num_nodes)
        # exit()
        csdl_model = VLMSolverModel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=self.num_nodes,
            eval_pts_shapes=eval_pts_shapes,
            # coeffs_aoa=coeffs_aoa,
            # coeffs_cd=coeffs_cd,
            mesh_unit='ft',
            cl0=[0, 0 ,0 ,0],
            model_selection=self.model_selection,
            module_opt=True,
        )

        return csdl_model

class VLMMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('surface_names')
        self.parameters.declare('surface_shapes')
        self.parameters.declare('meshes')
        self.parameters.declare('mesh_units', default='m')


class VLMSolverModel(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('prefix')


        self.parameters.declare('free_stream_velocities', default=None)

        self.parameters.declare('eval_pts_location', default=0.25)
        self.parameters.declare('eval_pts_names', default=None)

        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('sprs', default=None)
        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('TE_idx', default='last')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=[0])
        self.parameters.declare('model_selection', types=np.ndarray)
        self.parameters.declare('module_opt',default=True)

    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        num_nodes = self.parameters['num_nodes']
        surface_shapes = self.parameters['surface_shapes']
        model_selection = self.parameters['model_selection']
        num_nodes = len(model_selection)
        num_active_nodes = int(sum(model_selection))
        active_nodes = np.where(model_selection==1)[0]
        # surface_shapes = [(num_active_nodes, ) + surface_shape for surface_shape in surface_shapes]
        
        cl0 = self.parameters['cl0']
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_location = self.parameters['eval_pts_location']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        sprs = self.parameters['sprs']
        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']
        mesh_unit = self.parameters['mesh_unit']
        

        u_all = self.declare_variable('u', shape=(num_nodes, 1))
        v_all = self.declare_variable('v', shape=(num_nodes, 1))
        w_all = self.declare_variable('w', shape=(num_nodes, 1))
        theta_all = self.declare_variable('theta', shape=(num_nodes, 1))
        gamma_all = self.declare_variable('gamma', shape=(num_nodes, 1))
        psi_all = self.declare_variable('psi', shape=(num_nodes, 1))
        p_all = self.declare_variable('p', shape=(num_nodes, 1))
        q_all = self.declare_variable('q', shape=(num_nodes, 1))
        r_all = self.declare_variable('r', shape=(num_nodes, 1))
        x_all = self.declare_variable('x', shape=(num_nodes, 1))
        y_all = self.declare_variable('y', shape=(num_nodes, 1))
        z_all = self.declare_variable('z', shape=(num_nodes, 1))
        rho_all = self.declare_variable('density', shape=(num_nodes, 1))

        u = self.create_output('u_active_nodes', shape=(num_active_nodes, 1), val=0)
        v = self.create_output('v_active_nodes', shape=(num_active_nodes, 1), val=0)
        w = self.create_output('w_active_nodes', shape=(num_active_nodes, 1), val=0)
        theta = self.create_output('theta_active_nodes', shape=(num_active_nodes, 1), val=0)
        gamma = self.create_output('gamma_active_nodes', shape=(num_active_nodes, 1), val=0)
        psi  = self.create_output('psi_active_nodes', shape=(num_active_nodes, 1), val=0)
        p = self.create_output('p_active_nodes', shape=(num_active_nodes, 1), val=0)
        q = self.create_output('q_active_nodes', shape=(num_active_nodes, 1), val=0)
        r = self.create_output('r_active_nodes', shape=(num_active_nodes, 1), val=0)
        x = self.create_output('x_active_nodes', shape=(num_active_nodes, 1), val=0)
        y = self.create_output('y_active_nodes', shape=(num_active_nodes, 1), val=0)
        z = self.create_output('z_active_nodes', shape=(num_active_nodes, 1), val=0)
        rho = self.create_output('density_active_nodes', shape=(num_active_nodes, 1), val=0)

        for i in range(len(active_nodes)):
            index = int(active_nodes[i])
            u[i, 0] = u_all[index, 0]
            v[i, 0] = v_all[index, 0]
            w[i, 0] = w_all[index, 0]
            theta[i, 0] = theta_all[index, 0]
            gamma[i, 0] = gamma_all[index, 0]
            psi[i, 0] = psi_all[index, 0]
            p[i, 0] = p_all[index, 0]
            q[i, 0] = q_all[index, 0]
            r[i, 0] = r_all[index, 0]
            x[i, 0] = x_all[index, 0]
            y[i, 0] = y_all[index, 0]
            z[i, 0] = z_all[index, 0]
            rho[i, 0] = rho_all[index, 0]


        self.add(
            VLMSystem(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_active_nodes,
                solve_option=self.parameters['solve_option'],
                TE_idx=self.parameters['TE_idx'],
                mesh_unit=mesh_unit,
                module_opt=self.parameters['module_opt'],
                # eval_pts_option=eval_pts_option,
            ), 'VLM_system')
        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        else:
            eval_pts_names=self.parameters['eval_pts_names']

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_location=eval_pts_location,
            sprs=sprs,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
            mesh_unit=mesh_unit,
            cl0=cl0,
            # num_nodes=num_nodes,
            # active_nodes=active_nodes,
        )
        self.add(sub, name='VLM_outputs')


if __name__ == "__main__":

    pass