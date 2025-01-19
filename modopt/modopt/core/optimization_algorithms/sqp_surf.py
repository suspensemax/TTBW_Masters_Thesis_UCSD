import numpy as np
from scipy.sparse.csc import csc_matrix
from modopt.core.merit_functions.modified_lagrangian_ineq import ModifiedLagrangianIneq
import osqp
import scipy.sparse as sp
import time

from modopt.api import Optimizer
from modopt.line_search_algorithms import ScipyLS, BacktrackingArmijo, Minpack2LS
from modopt.merit_functions import AugmentedLagrangianIneq, ModifiedLagrangianIneq
# from modopt.approximate_hessians import BFGS as BFGS

# from modopt.approximate_hessians import BFGSM1 as BFGS
from modopt.approximate_hessians import BFGSScipy as BFGS

import pickle


# This optimizer takes constraints in all-inequality form, C(x) >= 0
class SQP_SURF(Optimizer):
    def initialize(self):
        self.solver_name = 'sqp_surf'

        self.obj_in = self.problem.evaluate_objective
        self.grad_in = self.problem.evaluate_objective_gradient

        self.con_in = self.problem.evaluate_constraints_and_residuals
        self.jac_in = self.problem.evaluate_constraint_and_residual_jacobian

        # self.residual = self.problem.evaluate_residuals
        # self.res_jac = self.problem.evaluate_residual_jacobian
        # self.state = self.problem.solve_residual_equations
        self.states_in = self.problem.solve_residual_equations
        self.adj_in = self.problem.compute_adjoint_vector

        self.options.declare('max_itr', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-7, types=float)
        self.options.declare('feas_tol', default=1e-7, types=float)

        self.options.declare('callback',
                             default=lambda: None,
                             types=type(lambda x: None))

        self.default_outputs_format = {
            'major': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            'y': (float, (self.problem.ny, )),

            # Note that the number of constraints will
            # be updated after constraints are setup
            'lag_mult': (float, (self.problem.nc, )),
            'slacks': (float, (self.problem.nc, )),
            'constraints': (float, (self.problem.nc, )),
            # 'residuals': (float, (self.problem.nr, )),
            'adjoints': (float, (self.problem.nr, )),
            'opt': float,
            'feas': float,
            'time': float,
            'num_f_evals': int,
            'num_g_evals': int,
            'step': float,
            'rho': float,
            'merit': float,
        }

        self.options.declare(
            'outputs',
            types=list,
            default=[
                'major',
                'obj',
                'x',
                'y',
                'opt',
                'feas',
                'slacks',
                'lag_mult',
                'constraints',
                'time',
                'rho',
                'merit',
                'num_f_evals',
                'num_g_evals',
                'step',
                #  'lam',
                #  'psi',
                #  'residuals',
            ])

    # FOR CHECK PARTIALS
    def obj(self, x):
        return self.obj_in(x[:self.nx], x[self.nx:])

    def grad(self, x):
        return self.grad_in(x[:self.nx], x[self.nx:])

    def adj(self, x):
        return self.adj_in(x[:self.nx], x[self.nx:])

    def con(self, x):
        return self.con_in(x[:self.nx], x[self.nx:])

    def jac(self, x):
        return self.jac_in(x[:self.nx], x[self.nx:])

    def setup(self):
        # self.setup_constraints()
        nx = self.nx = self.problem.nx
        ny = self.ny = self.problem.ny
        nr = self.nr = self.problem.ny
        nc = self.nc = self.problem.nc

        # Total num. of design and state vars.
        n = self.n = nx + ny
        # Total num. of constraints including residuals
        m = self.m = nc + 2 * nr

        self.default_outputs_format['lag_mult'] = (float, (m, ))
        # self.default_outputs_format['lam'] = (float, (nc, ))
        # self.default_outputs_format['psi'] = (float, (nr, ))

        self.default_outputs_format['rho'] = (float, (m, ))
        self.default_outputs_format['slacks'] = (float, (m, ))
        self.default_outputs_format['constraints'] = (float, (m, ))
        # self.default_outputs_format['constraints'] = (float, (nc, ))
        # self.default_outputs_format['residuals'] = (float, (nr, ))
        # Damping parameters
        self.delta_rho = 1.
        self.num_rho_changes = 0

        # self.QN = BFGS(nx=n)
        self.QN = BFGS(nx=n,
                       exception_strategy='damp_update')
        self.MF = AugmentedLagrangianIneq(nx=n,
                                          nc=m,
                                          f=self.obj,
                                          c=self.con,
                                          g=self.grad,
                                          j=self.jac)
        self.ML = ModifiedLagrangianIneq(nx=n,
                                         nc=m,
                                         f=self.obj,
                                         c=self.con,
                                         g=self.grad,
                                         j=self.jac)

        self.LSS = ScipyLS(f=self.MF.compute_function,
                           g=self.MF.compute_gradient,
                           max_step=2.0)
        # self.LS = Minpack2LS(f=self.MF.compute_function,
        #                      g=self.MF.compute_gradient)
        self.LSB = BacktrackingArmijo(f=self.MF.compute_function,
                                      g=self.MF.compute_gradient)

        # For u_k of QP solver
        self.u_k = np.full((m, ), np.inf)

    # Adapt constraints to C(x) >= 0
    def setup_constraints(self, ):

        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper

        x_lower = self.problem.x_lower
        x_upper = self.problem.x_upper

        self.nc = 0
        self.lower_bounds = False
        self.upper_bounds = False
        self.lower_constraint_bounds = False
        self.upper_constraint_bounds = False

        # Adapt bounds as ineq constraints C(x) >= 0
        # Remove bounds with -np.inf as lower bound
        lbi = self.lower_bound_indices = np.where(x_lower != -np.inf)[0]
        # Remove bounds with np.inf as upper bound
        ubi = self.upper_bound_indices = np.where(x_upper != np.inf)[0]

        if len(lbi) > 0:
            self.lower_bounds = True
            self.nc += len(lbi)

        if len(ubi) > 0:
            self.upper_bounds = True
            self.nc += len(ubi)

        # Adapt eq/ineq constraints as constraints >= 0
        # Remove constraints with -np.inf as lower bound
        lci = self.lower_constraint_indices = np.where(
            c_lower != -np.inf)[0]
        # Remove constraints with np.inf as upper bound
        uci = self.upper_constraint_indices = np.where(
            c_upper != np.inf)[0]

        if len(lci) > 0:
            self.lower_constraint_bounds = True
            self.nc += len(lci)

        if len(uci) > 0:
            self.upper_constraint_bounds = True
            self.nc += len(uci)

    # def con(self, x, y):
    #     # Compute problem constraints
    #     c_in = self.con_in(x, y)

    #     lbi = self.lower_bound_indices
    #     ubi = self.upper_bound_indices
    #     lci = self.lower_constraint_indices
    #     uci = self.upper_constraint_indices

    #     c_out = np.array([])

    #     if self.lower_bounds:
    #         c_out = np.append(c_out, x[lbi] - self.problem.x_lower[lbi])
    #     if self.upper_bounds:
    #         c_out = np.append(c_out, self.problem.x_upper[ubi] - x[ubi])
    #     if self.lower_constraint_bounds:
    #         c_out = np.append(c_out,
    #                           c_in[lci] - self.problem.c_lower[lci])
    #     if self.upper_constraint_bounds:
    #         c_out = np.append(c_out,
    #                           self.problem.c_upper[uci] - c_in[uci])

    #     res = self.residuals(x, y)
    #     c_out = np.concatenate((c_out, res, -res))

    #     return c_out

    # def jac(self, x, y):
    #     # Compute problem constraint Jacobian
    #     j_in = self.jac_in(x, y)

    #     n = self.n
    #     lbi = self.lower_bound_indices
    #     ubi = self.upper_bound_indices
    #     lci = self.lower_constraint_indices
    #     uci = self.upper_constraint_indices

    #     j_out = np.empty((1, n), dtype=float)

    #     if self.lower_bounds:
    #         j_out = np.append(j_out, np.identity(n)[lbi], axis=0)
    #     if self.upper_bounds:
    #         j_out = np.append(j_out, -np.identity(n)[ubi], axis=0)
    #     if self.lower_constraint_bounds:
    #         j_out = np.append(j_out, j_in[lci], axis=0)
    #     if self.upper_constraint_bounds:
    #         j_out = np.append(j_out, -j_in[uci], axis=0)

    #     res_jac = self.res_jac(x, y)
    #     j_out = np.concatenate((j_out, res_jac, -res_jac))

    #     return j_out[1:]

    # Minimize A.L. w.r.t. slacks s.t. s >= 0
    def reset_slacks(self, c, pi, rho):
        rho_zero_indices = np.where(rho == 0)[0]
        rho_nonzero_indices = np.where(rho > 0)[0]
        s = np.zeros((len(rho), ))

        if len(rho_zero_indices) > 0:
            s[rho_zero_indices] = np.maximum(0, c[rho_zero_indices])

        if len(rho_nonzero_indices) > 0:
            c_rnz = c[rho_nonzero_indices]
            pi_rnz = pi[rho_nonzero_indices]
            rho_rnz = rho[rho_nonzero_indices]
            s[rho_nonzero_indices] = np.maximum(
                # 0, (c_rnz - pi_rnz /rho_rnz)
                0,
                (c_rnz - pi_rnz / np.maximum(rho_rnz, 1e-2)))

        return s

    def update_scalar_rho(self, rho_k, dir_deriv_al, pTHp, p_pi, c_k,
                          s_k):
        m = self.m
        delta_rho = self.delta_rho
        num_rho_changes = self.num_rho_changes

        # rho_ref = np.linalg.norm(rho_k)
        rho_ref = rho_k[0] * 1.

        if dir_deriv_al <= -0.5 * pTHp:
            rho_computed = rho_k[0]
        else:
            # note: scalar rho_min here
            if np.linalg.norm(c_k - s_k) != 0:
                rho_min = 2 * np.linalg.norm(p_pi) / np.linalg.norm(
                    c_k - s_k)
            else:  # Just a heuristic
                rho_min = 2.e3 * np.linalg.norm(p_pi)
            rho_computed = max(rho_min, 2 * rho_k[0])
            # rho[:] = np.max(rho_min, 2 * rho[0]) * np.ones((m,))

        # Damping for rho (Note: vector rho is still not updated)
        if rho_k[0] < 4 * (rho_computed + delta_rho):
            rho_damped = rho_k[0]
        else:
            rho_damped = np.sqrt(rho_k[0] * (rho_computed + delta_rho))

        rho_new = max(rho_computed, rho_damped)
        rho_k[:] = rho_new * np.ones((m, ))

        # Increasing rho
        if rho_k[0] > rho_ref:
            if num_rho_changes >= 0:
                num_rho_changes += 1
            else:
                delta_rho *= 2.
                num_rho_changes = 0

        # Decreasing rho
        elif rho_k[0] < rho_ref:
            if num_rho_changes <= 0:
                num_rho_changes -= 1
            else:
                delta_rho *= 2.
                num_rho_changes = 0

        return rho_k

    def update_vector_rho(self, rho_k, dir_deriv_al, pTHp, p_pi, c_k,
                          s_k):
        delta_rho = self.delta_rho
        num_rho_changes = self.num_rho_changes

        rho_ref = np.linalg.norm(rho_k)
        # rho_ref = rho_k[0] * 1.

        # ck_sk =
        a = -np.power(c_k - s_k, 2)
        b = -0.5 * pTHp - dir_deriv_al + np.inner(a, rho_k)

        well_defined_a = True
        # print('a*sgn(b)', a * np.sign(b))

        if b > 0:
            a_plus = np.maximum(a, 0)
            a_plus_norm = np.linalg.norm(a_plus)
            if a_plus_norm == 0:
                well_defined_a = False
            else:
                rho_computed = (b / (a_plus_norm**2)) * a_plus
        elif b < 0:
            a_minus = -np.minimum(a, 0)
            a_minus_norm = np.linalg.norm(a_minus)
            if a_minus_norm == 0:
                well_defined_a = False
            else:
                rho_computed = -(b / (a_minus_norm**2)) * a_minus

        # TODO: check the condition below
        else:  # When b = 0
            rho_computed = np.zeros((len(rho_k), ))

        if not (well_defined_a):  # Perform a scalar rho update if a = 0
            if dir_deriv_al <= -0.5 * pTHp:
                rho_computed = rho_k * 1.
            else:
                # note: scalar rho_min here
                if np.linalg.norm(c_k - s_k) != 0:
                    rho_min = 2 * np.linalg.norm(p_pi) / np.linalg.norm(
                        c_k - s_k)
                else:  # Just a heuristic
                    rho_min = 2.e3 * np.linalg.norm(p_pi)
                rho_computed = np.maximum(2 * rho_k, rho_min)

        # Damping for rho (Note: vector rho is still not updated)
        rho_damped = np.where(
            rho_k < 4 * (rho_computed + delta_rho), rho_k,
            np.sqrt(rho_k * (rho_computed + delta_rho)))

        rho_new = np.maximum(rho_computed, rho_damped)
        rho_k[:] = rho_new

        rho_norm = np.linalg.norm(rho_k)
        # Increasing rho
        if rho_norm > rho_ref:
            if num_rho_changes >= 0:
                num_rho_changes += 1
            else:
                delta_rho *= 2.
                num_rho_changes = 0

        # Decreasing rho
        elif rho_norm < rho_ref:
            if num_rho_changes <= 0:
                num_rho_changes -= 1
            else:
                delta_rho *= 2.
                num_rho_changes = 0

        return rho_k

    def opt_check(self, pi, c, g, j):
        opt_tol = self.options['opt_tol']

        opt_tol_factor = (1 + np.linalg.norm(pi, np.inf))
        scaled_opt_tol = opt_tol * opt_tol_factor

        nonneg_violation = max(0., -np.min(pi))
        opt_check1 = (nonneg_violation <= scaled_opt_tol)

        # Note: compl_violation can be negative(Question: Ask Prof. Gill);
        #       Other 2 violations are always nonnegative
        compl_violation = np.max(c * pi)
        opt_check2 = (compl_violation <= scaled_opt_tol)

        stat_violation = np.linalg.norm(g - j.T @ pi, np.inf)
        opt_check3 = (stat_violation <= scaled_opt_tol)

        # opt is always nonnegative
        opt = max(nonneg_violation, compl_violation,
                  stat_violation) / opt_tol_factor
        opt_satisfied = (opt_check1 and opt_check2 and opt_check3)

        return opt_satisfied, opt

    def feas_check(self, x, c):
        feas_tol = self.options['feas_tol']

        feas_tol_factor = (1 + np.linalg.norm(x, np.inf))
        scaled_feas_tol = feas_tol * feas_tol_factor

        # Violation is positive
        max_con_violation = max(0., -np.min(c))
        feas_check = (max_con_violation <= scaled_feas_tol)

        feas = max_con_violation / feas_tol_factor
        feas_satisfied = feas_check
        return feas_satisfied, feas

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.nx
        ny = self.ny
        n = self.n
        nc = self.nc
        nr = self.nr
        m = self.m

        # x0 = self.problem.x.get_data()
        x0 = self.problem.x0
        # y0 = self.problem.y.get_data()
        y0 = self.problem.y0

        opt_tol = self.options['opt_tol']
        feas_tol = self.options['feas_tol']
        max_itr = self.options['max_itr']
        method = self.prob_options['formulation']

        obj = self.obj
        grad = self.grad
        con = self.con
        jac = self.jac

        # ''' ADD SURF
        update_states = self.states_in
        adj = self.adj
        # '''

        # qp_con = self.qp_con

        LSS = self.LSS
        LSB = self.LSB
        QN = self.QN
        MF = self.MF
        ML = self.ML

        start_time = time.time()

        # Set intial values for current iterates
        # z = (x, y)
        # pi = (lam, psi)
        # s = (s_con, s_res)
        # v = (z, pi, s)

        z_k = np.concatenate((x0, y0))
        x_k = z_k[:nx]
        y_k = z_k[nx:]

        pi_k = np.full((m, ), 1.)
        lam_k = pi_k[:nc]
        psi_k = pi_k[nc:]

        # ''' ADD SURF (FOR CFS and SURF)
        if method != 'fs':
            p_state = update_states(x_k) - y_k
            z_k[nx:] += p_state

            # psi = adj(x_k, y_k)
            psi = adj(z_k)
            # TODO: Check if psi is same for both ineq. constraints of an eq. constraint
            # pi_k[nc:] = np.append(-psi, -psi)
            pi_k[nc:] = np.append(psi, psi)

        # '''

        # f_k = obj(x_k, y_k)
        f_k = obj(z_k)
        # g_k = grad(x_k, y_k)
        g_k = grad(z_k)
        # c_k = con(x_k, y_k)
        c_k = con(z_k)
        # J_k = jac(x_k, y_k)
        J_k = jac(z_k)
        B_k = np.identity(n)

        # Vector of penalty parameters
        # rho_k = np.full((nc, ), 1.)
        rho_k = np.full((m, ), 0.)

        # Compute slacks by minimizing A.L. s.t. s_k >= 0
        # s_k = np.full((nc, ), 1.)
        s_k = self.reset_slacks(c_k, pi_k, rho_k)

        # s_k = np.maximum(0, c_k) # Use this if rho_k = 0. at start

        # Vector of design vars., lag. mults., and slack vars.
        v_k = np.concatenate((z_k, pi_k, s_k))
        z_k = v_k[:n]
        pi_k = v_k[n:(n + m)]
        s_k = v_k[(n + m):]

        x_k = v_k[:nx]
        y_k = v_k[nx:n]

        p_k = np.zeros((len(v_k), ))
        p_z = p_k[:n]
        p_pi = p_k[n:(n + m)]
        p_s = p_k[(n + m):]

        # QP parameters
        l_k = -c_k
        u_k = self.u_k
        A_k = J_k

        # Iteration counter
        itr = 0

        opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
        feas_satisfied, feas = self.feas_check(z_k, c_k)
        tol_satisfied = (opt_satisfied and feas_satisfied)
        num_f_evals = 1
        num_g_evals = 1

        # Evaluate merit function value
        MF.set_rho(rho_k)
        mf_k = MF.evaluate_function(z_k, pi_k, s_k, f_k, c_k)
        mfg_k = MF.evaluate_gradient(z_k, pi_k, s_k, f_k, c_k, g_k, J_k)

        # TODO: Test this to see if this makes any change
        ML.set_x_k(z_k, c_k=c_k, J_k=J_k)
        ML.set_pi_k(pi_k)
        mlg_k = g_k * 1.
        # mlg_k = ML.evaluate_gradient(z_k, f_k, c_k, g_k, J_k)
        # print('g_k:', g_k)
        # print('mlg_k:', mlg_k)

        # Initializing declared outputs
        self.update_outputs(major=0,
                            x=x_k,
                            y=y_k,
                            lag_mult=pi_k,
                            # lam=lam_k,
                            # psi_k=psi_k,
                            slacks=s_k,
                            # residuals=c_k[nc:nc + nr],
                            obj=f_k,
                            constraints=c_k,
                            # constraints=c_k[:nc],
                            opt=opt,
                            feas=feas,
                            time=time.time() - start_time,
                            num_f_evals=num_f_evals,
                            num_g_evals=num_g_evals,
                            step=0.,
                            rho=rho_k,
                            merit=mf_k)

        # Create scipy csc_matrix for Hk
        Bk_rows = np.triu(
            np.outer(np.arange(1, n + 1),
                     np.ones(n, dtype='int'))).flatten('f')
        Bk_rows = Bk_rows[Bk_rows != 0] - 1
        Bk_cols = np.repeat(np.arange(n), np.arange(1, n + 1))
        Bk_ind_ptr = np.insert(np.cumsum(np.arange(1, n + 1)), 0, 0)
        Bk_data = B_k[Bk_rows, Bk_cols]

        csc_Bk = sp.csc_matrix((Bk_data, Bk_rows, Bk_ind_ptr),
                               shape=(n, n))

        # QP problem setup
        qp_prob = osqp.OSQP()

        if isinstance(A_k, np.ndarray):
            # Setting up QP problem with fully dense dummy_A ensures that
            # change in density of A_k will not affect the 'update()'s
            dummy_A = np.full(A_k.shape, 1.)
            qp_prob.setup(csc_Bk, g_k, sp.csc_matrix(dummy_A), l_k, u_k)
            qp_prob.update(Ax=A_k.flatten('F'))

            dummy_A = None
            del dummy_A
        elif isinstance(A_k, sp.csc_matrix):
            # We assume that the sparsity structure of A_k will not
            # change during iterartions if a csc_matrix is given as input
            qp_prob.setup(
                csc_Bk,
                g_k,
                A_k,
                l_k,
                u_k,
                # Solver settings
                warm_start=False,
                # warm_start=True,
                verbose=True,
                # Polishing regularization parameter
                delta=1e-6,
                # polish=False,
                polish=True,
                polish_refine_iter=3,
                linsys_solver='qdldl',
                # ADMM rho step
                rho=0.1,
                # ADMM sigma step
                sigma=1e-6,
                max_iter=5000,
                # max_iter=3000,

                # Absolute tolerance
                # eps_abs=1e-3,
                # eps_abs=1e-4,
                eps_abs=min(opt_tol, feas_tol),
                # eps_abs=
                # 1e-6,  # To converge to tighter opt/feas tol for the overall problem,
                # need to tune this depending on problem scaling
                # Relative tolerance
                # eps_rel=1e-3,
                # eps_rel=1e-4,
                eps_rel=min(opt_tol, feas_tol),
                # eps_rel=
                # 1e-6,  # To converge to tighter opt/feas tol for the overall problem,
                # need to tune this depending on problem scaling
                # Primal infeasibility tolerance
                # eps_prim_inf=1e-4,
                eps_prim_inf=feas_tol,
                # Dual infeasibility tolerance
                # eps_dual_inf=1e-4,
                eps_dual_inf=opt_tol,
                # ADMM overrelaxation parameter
                alpha=1.6,
                # Scaled termination conditions
                scaled_termination=False,
                # Check termination interval
                check_termination=25,
                # Number of scaling iterations
                scaling=10,
                adaptive_rho=True,
                adaptive_rho_interval=0,
                adaptive_rho_tolerance=5,
                adaptive_rho_fraction=0.4,
                time_limit=0,
            )
        else:
            raise TypeError(
                'A_k must be a numpy matrix or scipy csc_matrix')

        self.consecutive_ls_failures = 0

        while (not (tol_satisfied) and itr < max_itr):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate

            # Solve QP problem
            qp_sol = qp_prob.solve()

            # Search direction for z_k:
            # (qp_sol.x) is the direction toward the next iterate for z
            p_k[:n] = qp_sol.x

            # Search direction for pi_k:
            # (-qp_sol.y) is the new estimate for pi
            # TODO: p_k[nx:(nx + nc)] = (-qp_sol.y) - pi_k
            p_k[n:(n + m)] = (-qp_sol.y) - v_k[n:n + m]

            # Search direction for s_k :
            # (c_k + J_k @ p_x) is the new estimate for s
            # TODO: p_k[(nx + nc):] = c_k + J_k @ p_x - s_k
            p_k[(n + m):] = c_k + J_k @ p_z - v_k[n + m:]

            dir_deriv_al = np.dot(mfg_k, p_k)
            pTHp = p_z.T @ (B_k @ p_z)

            # Update penalty parameters
            # rho_k = self.update_scalar_rho(rho_k, dir_deriv_al, pTHp,
            #                                p_pi, c_k, s_k)
            rho_k = self.update_vector_rho(rho_k, dir_deriv_al, pTHp,
                                           p_pi, c_k, s_k)

            MF.set_rho(rho_k)
            mf_k = MF.evaluate_function(z_k, pi_k, s_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(z_k, pi_k, s_k, f_k, c_k, g_k,
                                         J_k)

            # Compute the step length along the search direction via a line search
            alpha, mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = LSS.search(
                x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            # LSB.options['gamma_c'] = 0.1
            # LSB.options['max_itr'] = 2
            # alpha, mf_new, new_f_evals, new_g_evals, converged = LSB.search(
            #     x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            num_f_evals += new_f_evals
            num_g_evals += new_g_evals

            # if not converged:  # Fallback: Backtracking LS
            #     if self.consecutive_ls_failures <= 1:
            #         LSB.options['gamma_c'] = 0.1
            #         LSB.options['max_itr'] = 4
            #         alpha, mf_new, new_f_evals, new_g_evals, converged = LSB.search(
            #             x=v_k, p=p_k, f0=mf_k, g0=mfg_k)

            #         num_f_evals += new_f_evals
            #         num_g_evals += new_g_evals

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                self.consecutive_ls_failures += 1
                "Compute this factor heuristically"
                print('not converged, px_norm=', np.linalg.norm(p_z))
                print('not converged, ppi_norm=', np.linalg.norm(p_pi))
                print('not converged, ps_norm=', np.linalg.norm(p_s))
                alpha = None

                if self.consecutive_ls_failures == 1:
                    d_k = p_k * 1.e-1
                elif self.consecutive_ls_failures == 2:
                    d_k = p_k * 5.e-1
                # elif self.consecutive_ls_failures == 3:
                #     d_k = p_k * 1.e-1
                # elif self.consecutive_ls_failures == 4:
                #     d_k = p_k * 2.e-1
                # elif self.consecutive_ls_failures == 5:
                #     d_k = p_k * 5.e-1
                # elif self.consecutive_ls_failures == 6:
                #     d_k = p_k * 5.e-1
                # elif self.consecutive_ls_failures == 7:
                #     d_k = p_k * 5.e-1
                else:
                    d_k = p_k * 10.e-1

            else:
                self.consecutive_ls_failures = 0
                # print('converged, px_norm=', np.linalg.norm(p_z))
                # print('converged, ppi_norm=', np.linalg.norm(p_pi))
                # print('converged, ps_norm=', np.linalg.norm(p_s))
                d_k = alpha * p_k

            v_k += d_k
            # print('d_k', d_k)
            # print('p_x', d_k[:nx])
            # print('p_pi', d_k[nx:(nx + nc)])
            # print('p_s', d_k[(nx + nc):])

            # print('v_k', v_k)
            # print('v_k[:nx]', v_k[:nx])
            # print('v_k[nx:(nx + nc)]', v_k[nx:(nx + nc)])
            # print('v_k[(nx + nc):]', v_k[(nx + nc):])

            # print('x_k', x_k)
            # print('pi_k', pi_k)
            # print('s_k', s_k)

            # ''' ADD SURF (FOR SURF AND CFS)
            if method != 'fs':

                # Update the step for states
                p_state = update_states(x_k) - y_k
                v_k[nx:n] += p_state
                d_k[nx:n] += p_state

                # Update the multipliers
                # psi = adj(x_k, y_k)
                psi = adj(z_k)
                # p_psi = np.append(-psi, -psi) - v_k[n + nc:n + m]
                p_psi = np.append(psi, psi) - v_k[n + nc:n + m]
                # TODO: Check if psi is same for both ineq. constraints of an eq. constraint
                v_k[n + nc:n + m] += p_psi
                d_k[n + nc:n + m] += p_psi

            # '''

            # f_k = obj(x_k, y_k)
            f_k = obj(z_k)
            temp = -g_k

            # g_k = grad(x_k, y_k)
            g_k = grad(z_k)
            temp += g_k
            # c_k = con(x_k, y_k)
            c_k = con(z_k)
            # J_k = jac(x_k, y_k)
            J_k = jac(z_k)

            # Slack reset
            # if rho_k[0] == 0:
            #     v_k[nx + nc:] = np.maximum(0, c_k)
            # # When rho[0] > 0
            # else:
            #     v_k[nx + nc:] = np.maximum(
            #         0, c_k - v_k[nx:nx + nc] / rho_k)

            v_k[(n + m):] = self.reset_slacks(c_k, pi_k, rho_k)

            # Note: MF changes (decreases) after slack reset
            mf_k = MF.evaluate_function(z_k, pi_k, s_k, f_k, c_k)
            mfg_k = MF.evaluate_gradient(z_k, pi_k, s_k, f_k, c_k, g_k,
                                         J_k)

            # Update QP parameters
            l_k = -c_k
            A_k = J_k

            # Evaluate the Mod. Lag. gradient at the updated point x_(k+1)
            ML.set_pi_k(pi_k)
            mlg_new = ML.evaluate_gradient(z_k, f_k, c_k, g_k, J_k)
            w_k = mlg_new - mlg_k
            ML.set_x_k(z_k, c_k, J_k)
            mlg_k = g_k * 1.

            # if g_new == 'Unavailable':
            #     g_new = grad(x_k, y_k)
            #     g_new = grad(z_k)

            # Update the Hessian approximation
            print('temp-g_k:', np.linalg.norm(temp - g_k))
            QN.update(d_k[:n], w_k)

            # New Hessian approximation
            B_k = QN.B_k
            Bk_data[:] = B_k[Bk_rows, Bk_cols]

            # Update the QP problem
            if isinstance(A_k, np.ndarray):
                # print(np.linalg.norm(g_k), np.linalg.norm(l_k),
                #       np.linalg.norm(Bk_data), np.linalg.norm(A_k))
                # print(g_k[0])
                # print(l_k)
                # print(Bk_data)
                # print(A_k)
                qp_prob.update(q=g_k,
                               l=l_k,
                               Px=Bk_data,
                               Ax=A_k.flatten('F'))
            else:
                qp_prob.update(q=g_k, l=l_k, Px=Bk_data, Ax=A_k.data)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            opt_satisfied, opt = self.opt_check(pi_k, c_k, g_k, J_k)
            feas_satisfied, feas = self.feas_check(z_k, c_k)
            tol_satisfied = (opt_satisfied and feas_satisfied)

            # Update arrays inside outputs dict with new values from the current iteration

            # print(itr)
            # print('opt', opt)
            # print('feas', feas)
            # print('x', x_k)
            # print('obj', f_k)
            # print('cons', c_k)
            self.update_outputs(
                major=itr,
                x=x_k,
                y=y_k,
                lag_mult=pi_k,
                # lam=lam_k,
                # psi=psi_k,
                slacks=s_k,
                obj=f_k,
                constraints=c_k,
                # constraints=c_k[:nc],
                # residuals=c_k[nc:nc + nr],
                opt=opt,
                feas=feas,
                time=time.time() - start_time,
                num_f_evals=num_f_evals,
                num_g_evals=num_g_evals,
                rho=rho_k,
                step=alpha,
                # merit=mf_new)
                merit=mf_k)

            if self.options['callback']() is not None:
                with open('callback_{}.pkl'.format(itr), 'wb') as outp:
                    pickle.dump(self.options['callback'](), outp,
                                pickle.HIGHEST_PROTOCOL)

            # print('rho_k', rho_k[0])
            # print('alpha_k', alpha)
            # print('x_k', x_k[0])
            # print('y_k', y_k[0])
            # print('pi_k', pi_k[0])
            # print('c_k', c_k[0])
            # print('J_k_norm', sp.linalg.norm(J_k))
            # print('g_k_norm', np.linalg.norm(g_k))
            # print('s_k', s_k[0])
            # print('B_k_norm', np.linalg.norm(B_k))

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time

        # print(self.outputs)
