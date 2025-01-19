from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import csdl
import numpy as np
import lsdo_geo as lg
from caddee.core.caddee_core.system_model.design_scenario.design_condition.design_condition import AcStates, AtmosphericProperties
from VAST.core.vast_solver import VLMOutputs
# from caddee.utils.aircraft_models.tbw.Tbw_Viscous_Drag_Model_new import Tbwviscousdrag
from caddee.utils.aircraft_models.tbw.tbw_viscous_drag_model_sweep_3 import Tbwviscousdrag
import m3l
from typing import Union, Tuple
from dataclasses import dataclass
from caddee.utils.aircraft_models.tbw.tbw_area_ffd import tbwArea, geometryOutputs
from caddee.utils.aircraft_models.tbw.tbw_propulsion import TbwpropulsionProperties
from caddee.utils.helper_classes import MassProperties


@dataclass
class tbwconstranaly_outputs:
    W_S: m3l.Variable
    T_W: m3l.Variable
    Constraint_Stall: m3l.Variable
    Constraint_Climb: m3l.Variable
    Constraint_Maneuver: m3l.Variable
    # Constr_Obj: m3l.Variable

class TBW_Constraints_Analysis_Model(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        # parameters
        self.parameters.declare('name', types=str)
        self.parameters.declare('geometry_units', default='m', values=['m', 'ft'])
        # self.parameters.declare('counter', types = str)
        self.parameters.declare('num_nodes', default=1)

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.geometry_units = self.parameters['geometry_units']

    def compute(self) -> csdl.Model:
        return TBW_Constraints_Analysis_Model_CSDL(
            module=self,
            geometry_units=self.parameters['geometry_units'],
            num_nodes = self.parameters['num_nodes'],
        )

    def evaluate(self, atmos: AtmosphericProperties, 
                 ac_states : AcStates, 
                 tbw_mass: MassProperties, tbw_prop: TbwpropulsionProperties,
                 vlm_outputs: VLMOutputs, tbw_random_drag: Tbwviscousdrag, 
                 tbwarea: geometryOutputs,  
                 wing_span_dv: Union[m3l.Variable, None], 
                 wing_sweep_angle: Union[m3l.Variable, None], 
                 ) -> tbwconstranaly_outputs:
        
        num_nodes = self.parameters['num_nodes']
        # self.name = f"{self.counter}_tbw_constraints_analysis_model"

        self.arguments = {}
        self.arguments['span'] = wing_span_dv
        self.arguments['area'] = tbwarea.wing_area
        self.arguments['rho'] = atmos.density
        self.arguments['thrust'] = tbw_prop.thrust
        self.arguments['weight'] = tbw_mass.mass
        self.arguments['vlm_drag'] = vlm_outputs.Total_drag
        self.arguments['ran_drag'] = tbw_random_drag.D
        self.arguments['wing_sweep_angle'] = wing_sweep_angle
        self.arguments['u'] = ac_states.u
        self.arguments['v'] = ac_states.v
        self.arguments['w'] = ac_states.w
        # self.arguments['V'] = tbw_random_drag.V_inf
        self.arguments['cf_wave'] = tbw_random_drag.cf_wave_1
        self.arguments['Cf_interference'] = tbw_random_drag.Cf_interference

        W_S = m3l.Variable(name='W_S', shape=(num_nodes, 1), operation=self)
        T_W = m3l.Variable(name='T_W', shape=(num_nodes, 1), operation=self)
        Constraint_Stall = m3l.Variable(name='Constraint_Stall', shape=(num_nodes, 1), operation=self)
        Constraint_Climb = m3l.Variable(name='Constraint_Climb', shape=(num_nodes, 1), operation=self)
        Constraint_Maneuver = m3l.Variable(name='Constraint_Maneuver', shape=(num_nodes, 1), operation=self)
        # Constr_Obj = m3l.Variable(name='Constr_Obj', shape=(num_nodes, 1), operation=self)
 
        outputs = tbwconstranaly_outputs(
            W_S = W_S,
            T_W = T_W,
            Constraint_Stall = Constraint_Stall,
            Constraint_Climb = Constraint_Climb,
            Constraint_Maneuver = Constraint_Maneuver,
            # Constr_Obj = Constr_Obj,
        )
        return outputs

class TBW_Constraints_Analysis_Model_CSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare(name='name', default='constraint_analysis_climb')
        self.parameters.declare('num_nodes', default=100)
        self.parameters.declare('geometry_units', default='m')
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        geometry_units = self.parameters['geometry_units']

        span = self.declare_variable('span', shape=(num_nodes, 1))
        S = self.declare_variable('area', shape=(num_nodes, 1))
        T = self.declare_variable('thrust', shape=(num_nodes, 1))
        W = self.declare_variable('weight', shape=(num_nodes, 1))
        rho = self.declare_variable('rho', shape=(num_nodes, 1))
        vlm_drag = self.declare_variable('vlm_drag', shape=(num_nodes, 1))
        ran_drag = self.declare_variable('ran_drag', shape=(num_nodes, 1))
        cf_wave = self.declare_variable('cf_wave', shape=(num_nodes, 1))
        Cf_interference = self.declare_variable('Cf_interference', shape=(num_nodes, 1))
        u = self.declare_variable('u', shape=(num_nodes, 1))
        v = self.declare_variable('v', shape=(num_nodes, 1))
        w = self.declare_variable('w', shape=(num_nodes, 1))
        V_Me = (u**2 + v**2 + w**2)**0.5
        # V_Me = self.declare_variable('V', shape=(num_nodes, 1))
        lam = self.declare_variable('wing_sweep_angle', shape=(num_nodes,1))
        CD0 = cf_wave + Cf_interference
        self.register_output('CD0', CD0)

        n = 2.5

        span_Me = span * 0.3048
        S_Me = S * 0.092903
        # span_Me = span * 1.
        # S_Me = S * 1.
        # V_Me = V * 343.
        D = vlm_drag + ran_drag

        AR = span_Me**2. / S_Me

        e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64 # straight-wing
        # e = 0.75 # oswald efficiency

        q = 0.5 * rho * V_Me**2.
        pi_e_AR_q = 3.1415 * e * AR * q
        G = (T - D) / (W*9.8)

        W_S = W / S_Me
        T_W = T / (W*9.8)

        # use equation --------------
        # lam1 = 0. * lam
        lam1 = 1 * lam
        Clmax = 1.7
        # Clmax = 1.615
        CLmax = 0.9 * Clmax * csdl.cos(lam1)
        Vstall = (2 / (rho*CLmax) * W_S)**0.5
        # e = 4.61 * (1 - 0.045 * A**0.68) * (csdl.cos(wingLambda))**0.15 - 3.1 # swept-wing
        # # use constants --------------
        # CLmax = 1.5 # 1.53 # 1.8
        # Vstall = 45.6741
        # # Vstall = 81.5 * 0.51444

        C1 = CLmax * 0.5 * rho * Vstall**2.
        A2 = CD0 * q
        B2 = 1/pi_e_AR_q
        C2 = G * 1.
        A3 = A2 * 1.
        B3 = n**2 * B2
        Constraint_Stall = W_S - C1

        T_W_Climb = (A2/W_S + B2*W_S + C2)
        T_W_Maneuver = (A3/W_S + B3*W_S)

        Constraint_Climb = T_W - T_W_Climb
        Constraint_Maneuver = T_W - T_W_Maneuver
        # Constraint_Stall = W_S - (CLmax * 0.5 * rho * Vstall**2.)
        # Constraint_Climb = T_W - (G + W_S/pi_e_AR_q + CD0_q_W_S)
        # Constraint_Maneuver = T_W - (CD0_q_W_S + n**2 / pi_e_AR_q)

        self.register_output('AR', AR)
        self.register_output('S_Me', S_Me)

        self.register_output('C1', C1)
        self.register_output('A2', A2)
        self.register_output('B2', B2)
        self.register_output('C2', C2)
        self.register_output('A3', A3)
        self.register_output('B3', B3)
        self.register_output('CLmax', CLmax)

        self.register_output('T_W_Climb', T_W_Climb)
        self.register_output('T_W_Maneuver', T_W_Maneuver)

        self.register_output('W_S', W_S)
        self.register_output('T_W', T_W)
        self.register_output('Constraint_Stall', Constraint_Stall)
        self.register_output('Constraint_Climb', Constraint_Climb)
        self.register_output('Constraint_Maneuver', Constraint_Maneuver)

        # # # old objective: T_W -----------------------------
        # # # a = 0.5
        # # # Constr_Obj = a * (-1.) * W_S + (1-a) * T_W
        # # Constr_Obj = 1. * T_W
        # # # Constr_Obj = (-1.) * W_S
        # # self.register_output('Constr_Obj', Constr_Obj)

        # # new objective: fuel consumption -----------------------------
        # abc_Drag = vlm_drag + ran_drag
        # abc_Lift_over_Drag = vlm_lift / abc_Drag
        # self.register_output(name = 'Lift_over_Drag', var = abc_Lift_over_Drag)

        # TSFC = self.declare_variable('TSFC', shape=(num_nodes, 1))
        # V = self.declare_variable('V', shape=(num_nodes, 1))
        # # g = 9.8
        # # Range = 6482000.

        # # # Wf
        # # Constr_Obj = W * (1. - 1. / csdl.exp(Range / abc_Lift_over_Drag * (TSFC * 0.454/(60*60*4.448)) * g/(V*343)))
        # # # Constr_Obj = 1. * T_W + (TSFC + V) * 0.
        # # self.register_output('Constr_Obj', Constr_Obj)


        # # SFC = self.declare_variable('SFC', shape=(num_nodes, 1), val = 1.30139 * 10**(-4)) #TSFC
        # # velocity_1g = self.declare_variable('velocity_1g', shape=(num_nodes, 1), val = 207.02897322)
        # Range_cruise = self.declare_variable('Range_cruise', shape=(num_nodes, 1), val = 5556000) #3000 nautical miles
        # fuel_burn_a = (Range_cruise * TSFC)/(V * abc_Lift_over_Drag)
        # fuel_burn_b = 1 + fuel_burn_a + (fuel_burn_a**2)/2 + (fuel_burn_a**3)/6 + (fuel_burn_a**4)/24 + (fuel_burn_a**5)/120
        # fuel_burn = W*(1-(1/fuel_burn_b))
        # W_f = W - fuel_burn
        # self.register_output(name='final_weight',var=W_f)
        # self.register_output(name = 'fuel_burn_aaa', var = fuel_burn)
        # self.print_var(fuel_burn)
        # Constr_Obj = self.create_output(name='Constr_Obj', shape=(num_nodes,1), val=0)
        # Constr_Obj[:,0] = fuel_burn
