from typing import Any, Dict, List, Set, Tuple, Union
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.output import Output
from csdl.lang.input import Input
from csdl.lang.concatenation import Concatenation
from csdl.operations import *
from csdl.utils.check_default_val_type import check_default_val_type
from csdl.utils.check_constraint_value_type import check_constraint_value_type
from csdl.lang.implicit_operation_factory import ImplicitOperationFactory
from csdl.lang.variable import Variable
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.linear_solver import LinearSolver
from csdl.rep.graph_representation import GraphRepresentation
from csdl.lang.implicit_operation import ImplicitOperation
from csdl.lang.bracketed_search_operation import BracketedSearchOperation
from csdl.utils.collect_terminals import collect_terminals

# from lsdo_modules.module.implicit_module import ImplicitModule
from lsdo_modules.utils.parameters import Parameters
from csdl import Model
import numpy as np
from copy import copy

from lsdo_modules.utils.unpack_module import unpack_module
from json2html import *
import os 
import webbrowser

def _to_array(
    x: Union[int, float, np.ndarray, Variable]
) -> Union[np.ndarray, Variable]:
    if not isinstance(x, (np.ndarray, Variable)):
        x = np.array(x)
    return x

class ModuleMaker:
    def __init__(self, module=None, **kwargs) -> None:
        self.declared_variables = list()
        self.inputs = list()
        self.registered_outputs = list()
        self.created_outputs = list()
        
        self.module_info = list()
        self.module_inputs = list()
        self.module_outputs = list()
        self.promoted_vars = list()
        self.design_variables = dict()
        self.objective = dict()
        self.constraints = dict()
        self.module = module #kwargs['module']

        # NOTE 
        self.parameters: Parameters = Parameters()
        self.initialize_module()
        self.parameters.update(kwargs)
        # print(kwargs)
        # for key in kwargs.keys():
            # self.parameters.update(key)
        # self.__dict__.update(kwargs)

    def initialize_module(self):
        """
        User-defined method similar to `initialize` of the CSDL `Model` class.
        """
        pass

    def define_module(self): 
        """
        User-define method to define the module with similar similar 
        functionality to the `define` method of the CSDL `Model` class.
        """
        pass

    def register_module_input(
        self,
        name: str,
        val=1.0,
        shape=(1, ),
        src_indices=None,
        flat_src_indices=None,
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ):
        """
        General method for registering inputs that mirrors the following 
        methods of CSDL model class:
            - `declare_variable`
            - `create_input`
        
        
        By default, `register_module_input()` will return a CSDL object
        of type `Variable` (i.e., `declare_variable` will be called ).
        
        If a `module` is provided to the instance of `ModuleMaker`, 
        `register_module_input()` will return CSDL object of type
        `Input` and also declare a design variable if specified by 
        the user.
        """  
        if self.module is None:
            # print('NO MODULE')
            v = DeclaredVariable(
                name,
                val=check_default_val_type(val),
                shape=shape,
                src_indices=src_indices,
                flat_src_indices=flat_src_indices,
                units=units,
                desc=desc,
                tags=tags,
                shape_by_conn=shape_by_conn,
                copy_shape=copy_shape,
                distributed=distributed,
            )
            self.module_info.append(v)
            self.module_inputs.append(name)
            return v
        
        else:
            # print('YES MODULE')
            if name not in self.module.inputs:
                v = DeclaredVariable(
                    name,
                    val=check_default_val_type(val),
                    shape=shape,
                    src_indices=src_indices,
                    flat_src_indices=flat_src_indices,
                    units=units,
                    desc=desc,
                    tags=tags,
                    shape_by_conn=shape_by_conn,
                    copy_shape=copy_shape,
                    distributed=distributed,
                )
                self.module_info.append(v)
                self.module_inputs.append(name)
                return v
                # raise Exception(f"CSDL variable '{name}' is not found within the set module inputs: {list(self.module.inputs.keys())}. When calling 'set_module_input()', make sure the string matches '{name}'.")
            else:
                mod_var = self.module.inputs[name]
                mod_var_units = mod_var['units']
                mod_var_val = mod_var['val']
                if isinstance(mod_var_val, (int, float)):
                    mod_var_shape = (1, )
                else:
                    mod_var_shape = mod_var_val.shape

                

            if mod_var['computed_upstream'] is False and mod_var['dv_flag'] is False:
                i = Input(
                    name,
                    val=check_default_val_type(mod_var_val),
                    shape=mod_var_shape,
                    units=mod_var_units,
                    desc=desc,
                    tags=tags,
                    shape_by_conn=shape_by_conn,
                    copy_shape=copy_shape,
                    distributed=distributed,
                )
                self.module_info.append(i)
                self.module_inputs.append(name)
                return i

            elif mod_var['computed_upstream'] is False and mod_var['dv_flag'] is True:
                i = Input(
                    name,
                    val=check_default_val_type(mod_var_val),
                    shape=mod_var_shape,
                    units=mod_var_units,
                    desc=desc,
                    tags=tags,
                    shape_by_conn=shape_by_conn,
                    copy_shape=copy_shape,
                    distributed=distributed,
                )
                self.design_variables[name] = {
                    'lower': mod_var['lower'],
                    'upper': mod_var['upper'],
                    'scaler': mod_var['scaler']
                }
                self.module_info.append(i)
                self.module_inputs.append(name)
                return i 

            elif mod_var['computed_upstream'] is True:
                v = DeclaredVariable(
                    name,
                    val=check_default_val_type(mod_var_val),
                    shape=mod_var_shape,
                    src_indices=src_indices,
                    flat_src_indices=flat_src_indices,
                    units=mod_var_units,
                    desc=desc,
                    tags=tags,
                    shape_by_conn=shape_by_conn,
                    copy_shape=copy_shape,
                    distributed=distributed,
                )
                self.module_info.append(v)
                self.module_inputs.append(name)
                return v
            
            else:
                raise NotImplementedError
                
        
    def register_module_output(self, 
        name: str, 
        var: Output=None, 
        val=1.0, 
        shape=None, 
        units=None,
        res_units=None,
        desc='',
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
        promotes: bool = False
    ):  
        """
        General method for registering outputs that mirrors the following 
        methods of CSDL model class:
            - `register_output`
            - `create_output`

        If the argument `shape` is not None, `register_module_output` will 
        return a CSDL variable of type `Concatenation` (i.e., it will call
        CSDL's `create_outpu`). Otherwise it will return a CSDL variable of 
        type `Output`.
        """ 
        
        if shape and var:
            raise Exception(f'Attempting to create output with shape {shape}, while providing a variable to register as an output.')
    

        if shape:
            c = Concatenation(
            name,
            val=check_default_val_type(val),
            shape=shape,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            res_units=res_units,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
            distributed=distributed,
        )
            # self.register_output(name, c)
            self.module_info.append(c)
            self.module_outputs.append(name)
            if promotes is True:
                self.promoted_vars.append(name)

            return c
        else:
            if not isinstance(var, Output):
                raise TypeError(
                    'Can only register Output object as an output. Received type {}.'
                    .format(type(var)))
            else:
                if var in self.registered_outputs:
                    raise ValueError(
                        "Cannot register output twice; attempting to register "
                        "{} as {}.".format(var.name, name))
                if name in [r.name for r in self.registered_outputs]:
                    raise ValueError(
                        "Cannot register two outputs with the same name; attempting to register two outputs with name {}."
                        .format(name))
                if name in [r.name for r in self.inputs]:
                    raise ValueError(
                        "Cannot register output with the same name as an input; attempting to register output named {} with same name as an input."
                        .format(name))
                if name in [r.name for r in self.declared_variables]:
                    raise ValueError(
                        "Cannot register output with the same name as a declared variable; attempting to register output named {} with same name as a declared variable."
                        .format(name))

            var.name = name
            self.module_info.append(var)
            self.module_outputs.append(name)
            if promotes is True:
                self.promoted_vars.append(name)

            return var 

    def register_objective(
        self,
        name,
        ref=None,
        ref0=None,
        index=None,
        units=None,
        adder=None,
        scaler=None,
        parallel_deriv_color=None,
        cache_linear_solution=False,
    ):
        """
        Declare the objective for the optimization problem. Objective
        must be a scalar variable.
        """
        if len(self.objective) > 0:
            raise ValueError(
                "Cannot add more than one objective. More than one objective was added to the same model."
            )
        self.objective[name] = dict(
            ref=ref,
            ref0=ref0,
            index=index,
            units=units,
            adder=adder,
            scaler=scaler,
            parallel_deriv_color=parallel_deriv_color,
            cache_linear_solution=cache_linear_solution,
        )
        print('objective', self.objective)

    def register_constraint(
        self,
        name: str,
        lower=None,
        upper=None,
        equals=None,
        ref=None,
        ref0=None,
        adder=None,
        scaler=None,
        units=None,
        indices=None,
        linear=False,
        parallel_deriv_color=None,
        cache_linear_solution=False,
    ):  
        """
        Add a constraint to the optimization problem.
        """
        if name in self.constraints.keys():
            raise ValueError(
                "Constraint already defined for {}".format(name))
        else:
            check_constraint_value_type(lower)
            check_constraint_value_type(upper)
            check_constraint_value_type(equals)

            if lower is not None and upper is not None:
                if np.greater(lower, upper).any():
                    raise ValueError(
                        "Lower bound is greater than upper bound:\n lower bound: {}\n upper bound: {}"
                        .format(lower, upper))
            self.constraints[name] = dict(
                lower=lower,
                upper=upper,
                equals=equals,
                ref=ref,
                ref0=ref0,
                adder=adder,
                scaler=scaler,
                units=units,
                indices=indices,
                linear=linear,
                parallel_deriv_color=parallel_deriv_color,
                cache_linear_solution=cache_linear_solution,
            )

    def add_module(
        self,
        submodule,
        name=None,
        # promote_all=False
        promote=None
    ):
        csdl_model = submodule.assemble_csdl()
        self.promoted_vars += submodule.promoted_vars
        
        sub_module_info = {
            'csdl_model': csdl_model,
            'sub_module': submodule,
            'name': name,
            'promote': promote,
            # 'all_promoted_vars': submodule.promoted_vars
        }
        self.module_info.append(sub_module_info)
        
        return sub_module_info

    def generate_html(self, sim=None):
        # self.define_module()
        module_info = self.module_info
        module_dict = unpack_module(module_info)
        def recursive_items(dictionary):
            for key, value in dictionary.items():
                if type(value) is dict:
                    yield (key, value)
                    yield from recursive_items(value)
                else:
                    yield (key, value)
        
        def append_value_to_key(nested_dict, key_value_pair, new_value):
            # Loop through the dictionary to find the key-value pair
            for key, value in nested_dict.items():
                if isinstance(value, dict):
                    # If the current value is a dictionary, call the function recursively
                    nested_dict[key] = append_value_to_key(value, key_value_pair, new_value)
                elif key == key_value_pair[0] and value == key_value_pair[1]:
                    # If we find the key-value pair, modify the key by appending the new value
                    
                    # Update the dictionary with the new key and the same value
                    nested_dict['value'] = new_value
                else:
                    print(key_value_pair)
            return nested_dict
        
        
        # def append_to_dict(d, key1, key2, value):
        #     """
        #     Appends a value to a key in an arbitrarily nested dictionary.
        #     """
        #     for k, v in d.items():
        #         if k == key:
        #             d[k]['Inputs']['value'] = value
        #             d[k]['Outputs']['value'] = value
        #             # if isinstance(v, list):
        #             #     print(d[k])
        #             #     d[k]['value'] = value
        #             # else:
        #             #     d[k] = [v, value]
        #         elif isinstance(v, dict):
        #             append_to_dict(v, key, value)
        #     return d
        
        if sim:            
            # inputs = []
            # outputs = []
            # for key, value in recursive_items(module_dict):
            #     if key is 'Inputs':
            #         inputs += list(value.keys())
            #     elif key is 'Outputs':
            #         outputs += list(value.keys())
            # for input in inputs:
            #     val = sim[input]
            #     append_value_to_key(module_dict, ('Inputs', input), val)
            # for output in outputs:
            #     val = sim[output]
            #     append_value_to_key(module_dict, ('Outputs', output), val)
            pass
        # exit()
        # print('dict_keys', module_dict.keys())
        html = json2html.convert(json=module_dict)
        open('module_test.html', 'w').write(html)
        filename = 'file:///' + os.getcwd()+'/' + 'module_test.html'
        webbrowser.open_new_tab(filename)
        
    
    def assemble_csdl(self): 
        self.define_module()
        all_promoted_vars = self.promoted_vars
        design_variables = self.design_variables
        objective = self.objective
        module_info = self.module_info
        constraints = self.constraints
        
        class CSDLModel(Model):
            def initialize(self): pass

            def define(self):
                vars = []
                for entry in module_info: # self.module_info:
                    
                    # Inputs
                    if isinstance(entry, DeclaredVariable):
                        name = entry.name
                        val = entry.val
                        shape = entry.shape
                        self.declare_variable(name=name, val=val, shape=shape)
                        vars.append(name)

                    elif isinstance(entry, Input):
                        name = entry.name
                        val = entry.val
                        shape = entry.shape
                        self.create_input(name=name, val=val, shape=shape)
                        if name in design_variables:
                            dv = design_variables[entry.name]
                            self.add_design_variable(
                                dv_name=entry.name,
                                lower=dv['lower'],
                                upper=dv['upper'],
                                scaler=dv['scaler'],
                            )
                        vars.append(name)
                        
                    # Outputs
                    elif isinstance(entry, Output):
                        name =  entry.name
                        self.register_output(name=name, var=entry)
                        if name in objective.keys():
                            self.add_objective(
                                name=name,
                                # ref=objective[name]['ref'],
                                ref0=objective[name]['ref0'],
                                index=objective[name]['index'],
                                units=objective[name]['units'],
                                adder=objective[name]['adder'],
                                scaler=objective[name]['scaler'],
                                parallel_deriv_color=objective[name]['parallel_deriv_color'],
                                cache_linear_solution=objective[name]['cache_linear_solution'],
                            )
                        vars.append(name)

                    elif isinstance(entry, Concatenation):
                        name = entry.name
                        self.register_output(name=name, var=entry)
                        if name in objective.keys():
                            self.add_objective(
                                name=name,
                                # ref=objective[name]['ref'],
                                ref0=objective[name]['ref0'],
                                index=objective[name]['index'],
                                units=objective[name]['units'],
                                adder=objective[name]['adder'],
                                scaler=objective[name]['scaler'],
                                parallel_deriv_color=objective[name]['parallel_deriv_color'],
                                cache_linear_solution=objective[name]['cache_linear_solution'],
                            )
                        vars.append(name)

                    # Adding submodel
                    elif isinstance(entry, dict):
                        csdl_submodel = entry['csdl_model']
                        submodule = entry['sub_module']
                        module_inputs = submodule.module_inputs
                        name = entry['name']
                        promote = entry['promote']
                        if promote is None:
                            promotes=None
                        else:
                            promotes = promote + submodule.promoted_vars + [e for e in all_promoted_vars if e in module_inputs]
                            print('PROMOTES=', promotes)
                        self.add(csdl_submodel, name, promotes)
                    
                    # Implicit operation
                    elif isinstance(entry, ImplicitOperationFactory):
                        print('IMPLICIT')
                        csdl_model = entry.model
                        self.create_implicit_operation(csdl_model)
                        pass
                    
                    else:
                        raise NotImplementedError

                    # Adding constraints
                
                constraint_vars = list(set(vars).intersection(list(constraints.keys())))
                
        
        print(print('CONSTRAINTS', constraints))
        csdl_model = CSDLModel()
        for name in constraints.keys():
            csdl_model.add_constraint(
                name=name,
                lower=constraints[name]['lower'],
                upper=constraints[name]['upper'],
                equals=constraints[name]['equals'],
                ref=constraints[name]['ref'],
                ref0=constraints[name]['ref0'],
                adder=constraints[name]['adder'],
                scaler=constraints[name]['scaler'],
                units=constraints[name]['units'],
                indices=constraints[name]['indices'],
                linear=constraints[name]['linear'],
                parallel_deriv_color=constraints[name]['parallel_deriv_color'],
                cache_linear_solution=constraints[name]['cache_linear_solution'],
            )

        return csdl_model

    def _bracketed_search(
        self,
        states: Dict[str, Dict[str, Any]],
        residuals: List[str],
        implicit_model: 'Model',
        brackets: Dict[str,
                       Tuple[Union[int, float, np.ndarray, Variable],
                             Union[int, float, np.ndarray, Variable]]],
        *arguments: Variable,
        expose: List[str] = [],
    ):
        """
        Create an implicit operation whose residuals are defined by a
        `Model`.
        An implicit operation is an operation that solves an equation
        $f(x,y)=0$ for $y$, given some value of $x$.
        CSDL solves $f(x,y)=0$ by defining a residual $r=f(x,y)$ and
        updating $y$ until $r$ converges to zero.

        **Parameters**

        `arguments: List[Variable]`

        > List of variables to use as arguments for the implicit
        > operation.
        > Variables must have the same name as a declared variable
        > within the `model`'s class definition.

        :::note
        The declared variable _must_ be declared within `model`
        _and not_ promoted from a child submodel.
        :::

        `states: List[str]`

        > Names of states to compute using the implicit operation.
        > The order of the names of these states corresponds to the
        > order of the output variables returned by
        > `implicit_operation`.
        > The order of the names in `states` must also match the order
        > of the names of the residuals associated with each state in
        > `residuals`.

        :::note
        The declared variable _must_ be declared within `model`
        _and not_ promoted from a child submodel.
        :::

        `residuals: List[str]`

        > The residuals associated with the states.
        > The name of each residual must match the name of a
        > registered output in `model`.

        :::note
        The registered output _must_ be registered within `model`
        _and not_ promoted from a child submodel.
        :::

        `model: Model`

        > The `Model` object to use to define the residuals.
        > Residuals may be defined via functional composition and/or
        > hierarchical composition.

        :::note
        _Any_ `Model` object may be used to define residuals for an
        implicit operation
        :::

        `nonlinear_solver: NonlinearSolver`

        > The nonlinear solver to use to converge the residuals

        `linear_solver: LinearSolver`

        > The linear solver to use to solve the linear system

        `expose: List[str]`

        > List of intermediate variables inside `model` that are
        > required for computing residuals to which it is desirable
        > to have access outside of the implicit operation.
        > For example, if a trajectory is computed using time marching
        > and a residual is computed from the final state of the
        > trajectory, it may be desirable to plot that trajectory
        > after the conclusion of a simulation, e.g. after an
        > iteration during an optimization process.

        :::note
        The variable names in `expose` may be any name within the
        model hierarchy defined in `model`, but the variable names
        in `expose` are neither declared variables, nor registered
        outputs in `model`, although they may be declared
        variables/registered outputs in a submodel (i.e. they are
        neither states nor residuals in the, implicit operation).
        :::

        **Returns**

        `Tuple[Ouput]`

        > Variables to use in this `Model`.
        > The variables are named according to `states` and `expose`,
        > and are returned in the same order in which they are
        > declared.
        > For example, if `states=['a', 'b', 'c']` and
        > `expose=['d', 'e', 'f']`, then the outputs
        > `a, b, c, d, e, f` in
        > `a, b, c, d, e, f = self.implcit_output(...)`
        > will be named
        > `'a', 'b', 'c', 'd', 'e', 'f'`, respectively.
        > This enables use of exposed intermediate variables (in
        > addition to the states computed by converging the
        > residuals) from `model` in this `Model`.
        > Unused outputs will be ignored, so
        > `a, b, c = self.implcit_output(...)`
        > will make the variables declared in `expose` available for
        > recording/analysis and promotion/connection, but they will
        > be unused by this `Model`.
        > Note that these variables are already registered as outputs
        > in this `Model`, so there is no need to call
        > `Model.register_output` for any of these variables.
        """
        state_names = list(states.keys())
        (
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_residuals,
            rep,
            exposed_variables,
        ) = self._generate_maps_for_implicit_operation(
            implicit_model,
            arguments,
            state_names,
            residuals,
            expose,
        )

        # store brackets that are not CSDL variables as numpy arrays
        new_brackets: Dict[str, Tuple[Union[np.ndarray, Variable],
                                      Union[np.ndarray,
                                            Variable]]] = dict()
        # use this to check which states the user has failed to assign a
        # bracket
        states_without_brackets = copy(state_names)
        for k, v in brackets.items():
            if k not in state_names:
                raise ValueError(
                    "No state {} for specified bracket {}".format(k, v))
            if k in states_without_brackets:
                states_without_brackets.remove(k)

            if len(v) != 2:
                raise ValueError(
                    "Bracket {} for state {} is not a tuple of two values or Variable objects."
                    .format(v, k))

            (a, b) = v
            a = _to_array(a)
            b = _to_array(b)
            if a.shape != b.shape:
                raise ValueError(
                    "Bracket values for {} are not the same shape; {} != {}"
                    .format(k, a.shape, b.shape))
            new_brackets[k] = (a, b)

        if len(states_without_brackets) > 0:
            raise ValueError(
                "The following states are missing brackets: {}".format(
                    states_without_brackets))

        op = BracketedSearchOperation(
            implicit_model,
            rep,
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_variables,
            exposed_residuals,
            *arguments,
            expose=expose,
            brackets=new_brackets,
            # TODO: add tol
        )

        return self._return_implicit_outputs(
            implicit_model,
            op,
            residuals,
            expose,
            states,
        )
    
    def _implicit_operation(
        self,
        states: Dict[str, Dict[str, Any]],
        *arguments: Variable,
        residuals: List[str],
        model: 'Model',
        nonlinear_solver: NonlinearSolver,
        linear_solver: Union[LinearSolver, None] = None,
        expose: List[str] = [],
        defaults: Dict[str, Union[int, float, np.ndarray]] = dict(),
    ) -> Union[Output, Tuple[Output, ...]]:
        """
        Create an implicit operation whose residuals are defined by a
        `Model`.
        An implicit operation is an operation that solves an equation
        $f(x,y)=0$ for $y$, given some value of $x$.
        CSDL solves $f(x,y)=0$ by defining a residual $r=f(x,y)$ and
        updating $y$ until $r$ converges to zero.

        **Parameters**

        `arguments: List[Variable]`

            List of variables to use as arguments for the implicit
            operation.
            Variables must have the same name as a declared variable
            within the `model`'s class definition.

            :::note
            The declared variable _must_ be declared within `model`
            _and not_ promoted from a child submodel.
            :::

        `states: List[str]`

            Names of states to compute using the implicit operation.
            The order of the names of these states corresponds to the
            order of the output variables returned by
            `implicit_operation`.
            The order of the names in `states` must also match the order
            of the names of the residuals associated with each state in
            `residuals`.

            :::note
            The declared variable _must_ be declared within `model`
            _and not_ promoted from a child submodel.
            :::

        `residuals: List[str]`

            The residuals associated with the states.
            The name of each residual must match the name of a
            registered output in `model`.

            :::note
            The registered output _must_ be registered within `model`
            _and not_ promoted from a child submodel.
            :::

        `model: Model`

            The `Model` object to use to define the residuals.
            Residuals may be defined via functional composition and/or
            hierarchical composition.

            :::note
            _Any_ `Model` object may be used to define residuals for an
            implicit operation
            :::

        `nonlinear_solver: NonlinearSolver`

            The nonlinear solver to use to converge the residuals

        `linear_solver: LinearSolver`

            The linear solver to use to solve the linear system

        `expose: List[str]`

            List of intermediate variables inside `model` that are
            required for computing residuals to which it is desirable
            to have access outside of the implicit operation.

            For example, if a trajectory is computed using time marching
            and a residual is computed from the final state of the
            trajectory, it may be desirable to plot that trajectory
            after the conclusion of a simulation, e.g. after an
            iteration during an optimization process.

            :::note
            The variable names in `expose` may be any name within the
            model hierarchy defined in `model`, but the variable names
            in `expose` are neither declared variables, nor registered
            outputs in `model`, although they may be declared
            variables/registered outputs in a submodel (i.e. they are
            neither states nor residuals in the, implicit operation).
            :::

        **Returns**

        `Tuple[Ouput]`

            Variables to use in this `Model`.
            The variables are named according to `states` and `expose`,
            and are returned in the same order in which they are
            declared.
            For example, if `states=['a', 'b', 'c']` and
            `expose=['d', 'e', 'f']`, then the outputs
            `a, b, c, d, e, f` in
            `a, b, c, d, e, f = self.implcit_output(...)`
            will be named
            `'a', 'b', 'c', 'd', 'e', 'f'`, respectively.
            This enables use of exposed intermediate variables (in
            addition to the states computed by converging the
            residuals) from `model` in this `Model`.
            Unused outputs will be ignored, so
            `a, b, c = self.implcit_output(...)`
            will make the variables declared in `expose` available for
            recording/analysis and promotion/connection, but they will
            be unused by this `Model`.
            Note that these variables are already registered as outputs
            in this `Model`, so there is no need to call
            `Model.register_output` for any of these variables.
        """
        state_names = list(states.keys())
        (
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_residuals,
            rep,
            exposed_variables,
        ) = self._generate_maps_for_implicit_operation(
            model,
            arguments,
            state_names,
            residuals,
            expose,
        )

        # store default values as numpy arrays
        new_default_values: Dict[str, np.ndarray] = dict()
        for k, v in defaults.items():
            if k not in state_names:
                raise ValueError(
                    "No state {} for specified default value {}".format(
                        k, v))
            if not isinstance(v, (int, float, np.ndarray)):
                raise ValueError(
                    "Default value for state {} is not an int, float, or ndarray"
                    .format(k))
            if isinstance(v, np.ndarray):
                f = list(
                    filter(lambda x: x.name == k,
                           model.registered_outputs))
                if len(f) > 0:
                    if f[0].shape != v.shape:
                        raise ValueError(
                            "Shape of value must match shape of state {}; {} != {}"
                            .format(k, f[0].shape, v.shape))
                new_default_values[k] = np.array(v) * np.ones(
                    f[0].shape)

        # create operation, establish dependencies on arguments
        op = ImplicitOperation(
            model,
            rep,
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_variables,
            exposed_residuals,
            *arguments,
            expose=expose,
            defaults=new_default_values,
            nonlinear_solver=nonlinear_solver,
            linear_solver=linear_solver,
        )

        return self._return_implicit_outputs(
            model,
            op,
            residuals,
            expose,
            states,
        )
    
    def _generate_maps_for_implicit_operation(
        self,
        model: 'Model',
        arguments: Tuple[Variable, ...],
        state_names: List[str],
        residual_names: List[str],
        expose: List[str] = [],
    ) -> Tuple[Dict[str, Output], Dict[str, DeclaredVariable], Dict[
            str, List[DeclaredVariable]], Dict[
                str, List[DeclaredVariable]], Set[str],
               GraphRepresentation, Dict[str, Output]]:
        if not isinstance(model, Model):
            raise TypeError("{} is not a Model".format(model))

        rep = GraphRepresentation(model)

        # top level inputs will be unused, so we don't allow them
        if len(model.inputs) > 0:
            raise ValueError(
                "The model that defines residuals is not allowed to"
                "define top level inputs (i.e. calls to"
                "`Model.create_input`).")

        # check for duplicate arguments, states, and residuals
        arg_names: List[str] = [var.name for var in arguments]
        if len(set(arg_names)) < len(arg_names):
            raise ValueError("Duplicate arguments found")
        if len(set(state_names)) < len(state_names):
            raise ValueError("Duplicate names for states found")
        if len(set(residual_names)) < len(residual_names):
            raise ValueError("Duplicate names for residuals found")

        # check that each declared state has an associated residual
        if len(state_names) != len(residual_names):
            raise ValueError(
                "Number of states and residuals must be equal")

        for name in expose:
            if '.' in name:
                KeyError(
                    "Invalid name {} for exposing an intermediate variable in composite residual. Exposing intermediate variables with unpromoted names is not supported."
                    .format(name))

        # check that name and shape of each argument matches name and
        # shape of a declared variable in internal model, and transfer
        # value from argument to declared variable in model
        declared_variables_map: Dict[str, DeclaredVariable] = {
            x.name: x
            for x in model.declared_variables
        }
        for arg in arguments:
            if arg.name not in declared_variables_map.keys():
                raise ValueError(
                    "The argument {} is not a declared variable of the model used to define an implicit operation"
                    .format(arg.name))
            var = declared_variables_map[arg.name]
            if arg.shape != var.shape:
                raise ValueError(
                    "The argumet {} has shape {}, which does not match the shape {} of the declared variable of the model used to define an implicit operation"
                    .format(arg.name, arg.shape, var.shape))
            var.val = arg.val

        # check that name of each state matches name of a declared
        # variable in internal model
        for state_name in state_names:
            if state_name not in declared_variables_map.keys():
                raise ValueError(
                    "The state {} is not a declared variable of the model used to define an implicit operation"
                    .format(state_name))

        # check that name of each residual matches name of a registered
        # output in internal model
        registered_outputs_map: Dict[str, Output] = {
            x.name: x
            for x in model.registered_outputs
        }
        for residual_name in residual_names:
            if residual_name not in registered_outputs_map.keys():
                raise ValueError(
                    "The residual {} is not a registered output of the model used to define an implicit operation"
                    .format(residual_name))
        exposed_variables: Dict[str, Output] = {
            x.name: x
            for x in model.registered_outputs if x.name in set(expose)
        }

        # check that name of each exposed intermediate output matches
        # name of a registered output in internal model
        for exposed_name in expose:
            if exposed_name not in registered_outputs_map.keys():
                raise ValueError(
                    "The exposed output {} is not a registered output of the model used to define an implicit operation"
                    .format(exposed_name))

        # create two-way mapping between state objects and residual
        # objects so that back end can define derivatives of residuals
        # wrt states and arguments
        out_res_map: Dict[str, Output] = dict()
        res_out_map: Dict[str, DeclaredVariable] = dict()
        for s, r in zip(state_names, residual_names):
            if registered_outputs_map[
                    r].shape != declared_variables_map[s].shape:
                raise ValueError(
                    "Shape of state {} and residual {} do not match.".
                    format(s, r))
            out_res_map[s] = registered_outputs_map[r]
            res_out_map[r] = declared_variables_map[s]

        # TODO: (?) keep track of which exposed variables depend on
        # residuals; necessary for computing derivatives of residuals
        # associated with exposed variables wrt exposed variables, but
        # only those exposed variables that do not depend on a stata

        argument_names = [x.name for x in arguments]

        # Associate states with the arguments and states they depend on;
        out_in_map: Dict[str, List[DeclaredVariable]] = dict()
        for state_name, residual in out_res_map.items():
            # Collect inputs (terminal nodes) for this residual only; no
            # duplicates
            in_vars = list(
                set(collect_terminals(
                    [],
                    residual,
                    residual,
                )))

            if state_name in state_names and state_name not in [
                    var.name for var in in_vars
            ]:
                raise ValueError(
                    "Residual {} does not depend on state {}".format(
                        residual.name, state_name))

            # Only the arguments specified in the parent model and
            # declared states will be
            # inputs to the internal model
            out_in_map[state_name] = [
                v for v in in_vars
                if v.name in set(argument_names + state_names)
            ]

        # Associate exposed outputs with the inputs they depend on;
        exp_in_map: Dict[str, List[DeclaredVariable]] = dict()
        for exposed_name in expose:
            # Collect inputs (terminal nodes) for this residual only; no
            # duplicates
            in_vars = list(
                set(
                    collect_terminals(
                        [],
                        registered_outputs_map[exposed_name],
                        registered_outputs_map[exposed_name],
                    )))

            # Only the arguments specified in the parent model and
            # declared states will be
            # inputs to the internal model
            exp_in_map[exposed_name] = [
                v for v in in_vars
                if v.name in set(argument_names + state_names)
            ]

        # collect exposed variables that are residuals so that we don't
        # assume residuals are zero for these variables
        exposed_residuals: Set[str] = {
            exposed_name
            for exposed_name in expose
            if exposed_name in set(residual_names)
        }

        return (
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_residuals,
            rep,
            exposed_variables,
        )
    
    def _return_implicit_outputs(
        self,
        model: 'Model',
        op: Union[ImplicitOperation, BracketedSearchOperation],
        residuals: List[str],
        expose: List[str],
        states: Dict[str, Dict[str, Any]],
    ) -> Union[Output, Tuple[Output, ...]]:
        # create outputs of operation, establish dependencies on
        # operation, and register outputs
        outs: List[Output] = []

        # TODO: loop over exposed
        state_names = list(states.keys())
        for s, r in zip(state_names, residuals):

            internal_var = list(
                filter(lambda x: x.name == s,
                       model.declared_variables))[0]

            out = Output(
                s,
                shape=internal_var.shape,
                **states[s],
                op=op,
            )
            self.register_module_output(s, out)
            outs.append(out)

        # include exposed intermediate outputs in GraphRepresentation
        se = set(expose)
        ex = filter(lambda x: x.name in se, model.registered_outputs)
        for e in ex:
            out = Output(
                e.name,
                val=e.val,
                shape=e.shape,
                units=e.units,
                desc=e.desc,
                tags=e.tags,
                shape_by_conn=e.shape_by_conn,
                copy_shape=e.copy_shape,
                distributed=e.distributed,
                op=op,
            )
            self.register_output(e.name, out)
            outs.append(out)

        # ensure operation has knowledge of outputs so that back end can
        # generate code from operation
        op.outs = tuple(outs)

        # return outputs
        if len(outs) > 1:
            return tuple(outs)
        else:
            return outs[0]
        
    def create_implicit_operation(self, module): 
        # self_arg = self.assemble_csdl()
        csdl_model = module.assemble_csdl()
        return ImplicitOperationFactory(self, csdl_model)
        # return ImplicitModule(module)

