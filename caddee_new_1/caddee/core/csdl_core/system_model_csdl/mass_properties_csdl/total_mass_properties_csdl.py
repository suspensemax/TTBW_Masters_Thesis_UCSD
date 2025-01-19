from csdl import Model
from caddee.utils.base_model_csdl import BaseModelCSDL
from caddee.core.caddee_core.system_model.sizing_group.sizing_group import SizingGroup
import numpy as np


class TotalMassPropertiesCSDL(BaseModelCSDL):
    """
    Computes total mass properties of the system
    """
    def initialize(self):
        self.parameters.declare('sizing_group', default=None, types=SizingGroup, allow_none=True)

    def define(self):
        sizing_group = self.parameters['sizing_group']
        
        ref_pt = self.declare_variable('ref_pt', shape=(3,), val=np.array([0, 0, 0]))
        
        # Initialize mass proporties as CSDL variables with zero value
        # Total mass
        m = self.create_input('m_compute_total', val=0)
        
        # CG in the global reference frame
        cgx = self.create_input('cgx_compute_total', val=0)
        cgy = self.create_input('cgy_compute_total', val=0)
        cgz = self.create_input('cgz_compute_total', val=0)
        
        # Elements of the inertia tensor in the global reference frame
        ixx = self.create_input('ixx_compute_total', val=0)
        iyy = self.create_input('iyy_compute_total', val=0)
        izz = self.create_input('izz_compute_total', val=0)
        ixz = self.create_input('ixz_compute_total', val=0)

        # Loop over all sizing models and compute total mass properties 
        # of the system (using parallel axis theorem)

        if sizing_group:
            m_sc = self.declare_variable(f"m_total_constant", shape=(1, ))
            cgx_sc = self.declare_variable(f"cgx_total_constant", shape=(1, ))
            cgy_sc = self.declare_variable(f"cgy_total_constant", shape=(1, ))
            cgz_sc = self.declare_variable(f"cgz_total_constant", shape=(1, ))
            ixx_sc = self.declare_variable(f"ixx_total_constant", shape=(1, ))
            iyy_sc = self.declare_variable(f"iyy_total_constant", shape=(1, ))
            izz_sc = self.declare_variable(f"izz_total_constant", shape=(1, ))
            ixz_sc = self.declare_variable(f"ixz_total_constant", shape=(1, ))

            # Compute total cg
            cgx = (m * cgx + m_sc * cgx_sc) / (m + m_sc)
            cgy = (m * cgy + m_sc * cgy_sc) / (m + m_sc)
            cgz = (m * cgz + m_sc * cgz_sc) / (m + m_sc)

            # Position vector elements
            pos_x = cgx_sc - ref_pt[0]
            pos_y = cgy_sc - ref_pt[1]
            pos_z = cgz_sc - ref_pt[2]

            # Compute total inertia tensor
            ixx = ixx + ixx_sc + m_sc * (pos_y**2 + pos_z**2)
            iyy = iyy + iyy_sc + m_sc * (pos_x**2 + pos_z**2)
            izz = izz + izz_sc + m_sc * (pos_x**2 + pos_y**2)
            ixz = ixz + ixz_sc + m_sc * (pos_x * pos_z)

            # Compute total mass
            m = m + m_sc
        
        # Varying MPs
        m_v = self.declare_variable('m_total_varying', shape=(1, ), val=0.)
        cgx_v = self.declare_variable('cgx_total_varying', shape=(1, ), val=0.)
        cgy_v = self.declare_variable('cgy_total_varying', shape=(1, ), val=0.)
        cgz_v = self.declare_variable('cgz_total_varying', shape=(1, ), val=0.)
        ixx_v = self.declare_variable('ixx_total_varying', shape=(1, ), val=0.)
        iyy_v = self.declare_variable('iyy_total_varying', shape=(1, ), val=0.)
        izz_v = self.declare_variable('izz_total_varying', shape=(1, ), val=0.)
        ixz_v = self.declare_variable('ixz_total_varying', shape=(1, ), val=0.)

        # Compute total cg
        cgx = (m * cgx + m_v * cgx_v) / (m + m_v)
        cgy = (m * cgy + m_v * cgy_v) / (m + m_v)
        cgz = (m * cgz + m_v * cgz_v) / (m + m_v)

        # Position vector elements
        pos_x = cgx_v - ref_pt[0]
        pos_y = cgy_v - ref_pt[1]
        pos_z = cgz_v - ref_pt[2]

        # Compute total inertia tensor
        ixx = ixx + ixx_v + m_v * (pos_y**2 + pos_z**2)
        iyy = iyy + iyy_v + m_v * (pos_x**2 + pos_z**2)
        izz = izz + izz_v + m_v * (pos_x**2 + pos_y**2)
        ixz = ixz + ixz_v + m_v * (pos_x * pos_z)


        m_fudge = self.declare_variable('m_fudge', shape=(1, ), val=210)
        # Compute total mass
        m = m + m_v #+ m_fudge

        # Register total system mass properties 
        self.print_var(m)
        self.register_output('m_total', m)
        self.register_output('cgx_total', cgx)
        self.register_output('cgy_total', cgy)
        self.register_output('cgz_total', cgz)
        self.register_output('ixx_total', ixx)
        self.register_output('iyy_total', iyy)
        self.register_output('izz_total', izz)
        self.register_output('ixz_total', ixz)
