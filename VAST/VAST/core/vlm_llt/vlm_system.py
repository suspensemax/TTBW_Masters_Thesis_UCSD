import numpy as np
import csdl

from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp
from VAST.core.submodels.geometric_submodels.wake_coords_comp import WakeCoords
from VAST.core.submodels.geometric_submodels.mesh_preprocessing_comp import MeshPreprocessingComp
from VAST.core.submodels.aerodynamic_submodels.seperate_gamma_b import SeperateGammab
from VAST.core.submodels.implicit_submodels.solve_group import SolveMatrix
from VAST.core.submodels.implicit_submodels.compute_residual import ComputeResidual


class VLMSystem(csdl.Model):
    '''
    contains
    1. MeshPreprocessing_comp
    2. WakeCoords_comp
    3. solve_gamma_b_group
    3. seperate_gamma_b_comp
    4. extract_gamma_w_comp
    '''
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t', default=100)
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_location', default=0.25)

        self.parameters.declare('AcStates', default=None)

        self.parameters.declare('n_wake_pts_chord', default=2)

        self.parameters.declare('solve_option',
                                default='direct',
                                values=['direct', 'optimization'])
        self.parameters.declare('TE_idx', default='last')

    def define(self):
        # rename parameters
        num_nodes = self.parameters['num_nodes']
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        mesh_unit = self.parameters['mesh_unit']
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_location = self.parameters['eval_pts_location']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        bd_vortex_shapes = surface_shapes
        delta_t = self.parameters['delta_t']
        gamma_b_shape = sum((i[1] - 1) * (i[2] - 1) for i in bd_vortex_shapes)

        # frame_vel = self.declare_variable('frame_vel', shape=(3, ))
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((n_wake_pts_chord, item[1], 3)) for item in surface_shapes
        ]
        wake_vel_shapes = [(x[0] * x[1], 3) for x in wake_vortex_pts_shapes]

        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=surface_shapes,
                                       mesh_unit=mesh_unit,
                                       eval_pts_option=eval_pts_option,
                                       eval_pts_location=eval_pts_location,
                                    #    problem_type='fixed_wake',
                                       ),
                 name='MeshPreprocessing_comp')
        AcStates = self.parameters["AcStates"]
        if AcStates != None:
            add_adapter = True
        else:
            add_adapter = False

        if add_adapter == True:
            m = AdapterComp(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
            )
            # m.optimize_ir(False)
            self.add(m, name='adapter_comp')

        m = WakeCoords(surface_names=surface_names,
                       surface_shapes=surface_shapes,
                       n_wake_pts_chord=n_wake_pts_chord,
                       delta_t=delta_t,
                       TE_idx=self.parameters['TE_idx'])
        # m.optimize_ir(False)
        self.add(m, name='WakeCoords_comp')

        if self.parameters['solve_option'] == 'direct':
            self.add(SolveMatrix(n_wake_pts_chord=n_wake_pts_chord,
                                 surface_names=surface_names,
                                 bd_vortex_shapes=bd_vortex_shapes,
                                 delta_t=delta_t),
                     name='solve_gamma_b_group')
        elif self.parameters['solve_option'] == 'optimization':
            self.add(ComputeResidual(n_wake_pts_chord=n_wake_pts_chord,
                                     surface_names=surface_names,
                                     bd_vortex_shapes=bd_vortex_shapes,
                                     delta_t=delta_t),
                     name='solve_gamma_b_group')
        gamma_b = self.declare_variable('gamma_b',
                                        shape=(num_nodes, gamma_b_shape))

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=surface_shapes),
                 name='seperate_gamma_b_comp')


if __name__ == "__main__":

    pass