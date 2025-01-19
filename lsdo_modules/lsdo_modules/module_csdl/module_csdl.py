from csdl import Model
import numpy as np
from csdl import GraphRepresentation
import warnings
# from lsdo_modules.utils.make_xdsm import make_xdsm
from itertools import count


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return 'UserWarning: ' + str(msg) + '\n'
warnings.formatwarning = custom_formatwarning


class ModuleCSDL(Model):
    """
    Class acting as a liason between CADDEE and CSDL. 
    The API mirrors that of the CSDL Model class. 
    """ 
    _ids = count(0)
    def __init__(
            self, 
            module=None, 
            sub_modules=None,
            prepend=None,
            name='parent_module', 
            **kwargs
        ):
        self.id = next(self._ids)
        self.module = module
        self.prepend = prepend
        self.sub_modules_csdl = sub_modules
        self.name = name
        self.promoted_vars = list()
        
        self.module_inputs = dict()
        self.module_declared_vars = dict()
        self.module_outputs = dict()
        self.sub_modules = dict()
        self._module_output_names = list()
        self._auto_iv = list()
        
        super().__init__(**kwargs)

        self.objective = dict()
        self.design_variables = dict()
        self.design_constraints = dict()
        self.connections = list()
        

    
    def register_module_input(
            self, 
            name: str,
            val=1.0,
            shape=(1, ),
            units=None,
            desc='',
            importance=0,
            computed_upstream=True,
            vectorized=False,
            promotes=False,
        ):
        # When calling 'add_module()', we promote everything by default if 
        # the promotes keyword is not set. In that case the 'promotes=False' 
        # is never considered. However, when calling 'add_module(..., promotes=['...']),'
        # only a subset of variables are promoted. When a developer calls 
        # 'register_module_input(..., promotes=True)' they can simply call 
        # 'add_module(..., promotes=[])' and modules will append that list with 
        # variables where 'promotes=True'. The capability of passing in strings 
        # directly into 'promotes=['']' is reatained
        # TODO: see if the lines below need to take into account whether there is 
        # a prepend or not. Example 'cruise_condition' + 'mach_number'. 
        if promotes is True:
            self.promoted_vars.append(name)
        
        # If there is a module associated with the ModuleCSDL instance
        # The created CSDL variable will be of type Input
        if self.module:
            # Check if the variable set by the user when calling 'set_module_input()' 
            # matches what the variable defined by the (solver) developer when calling
            # 'self.regsiter_module_input' If not, check if the variable is an output 
            # of an upstreams model. If not raise a warning and make the variable
            # an instance of DeclaredVariable
            
            if self.prepend:
                name = name.removeprefix(f'{self.prepend}_')
            else: pass

            if computed_upstream is True:
                if self.prepend:
                    var_name = f'{self.prepend}_{name}'
                    input_variable = self.declare_variable(name=var_name, shape=shape)
                    self.module_declared_vars[name] = dict(
                        shape=shape, 
                        importance=importance,
                        vectorized=vectorized,         
                    )
                else:
                    input_variable = self.declare_variable(name=name, shape=shape)
                    self.module_declared_vars[name] = dict(
                        shape=shape, 
                        importance=importance,
                        vectorized=vectorized,
                    )

            else:
                if name not in self.module.inputs:
                    # print('self.sub_modules.values()', self.sub_modules.values())
                    # print(self.sub_modules)
                    # print(self.sub_modules_csdl)
                    # print(self.module_outputs)
                    # Get all module outputs of upstream modules and append to a list
                    if self.sub_modules_csdl is not None:
                        for sub_module in {**self.sub_modules_csdl, **self.sub_modules}.values():
                            module_outputs = sub_module['outputs']
                            self._module_output_names += [name for name in module_outputs]
                    else:
                        for sub_module in self.sub_modules.values():
                            module_outputs = sub_module['outputs']
                            self._module_output_names += [name for name in module_outputs]
                    # Check if the variable is computed in an upstream module
                    if name in self._module_output_names:
                        if self.prepend:
                            var_name = f'{self.prepend}_{name}'
                            input_variable = self.declare_variable(name=var_name, shape=shape)
                            self.module_declared_vars[name] = dict(
                                shape=shape, 
                                importance=importance,
                                vectorized=vectorized,
                            )
                        else:
                            input_variable = self.declare_variable(name=name, shape=shape)
                            self.module_declared_vars[name] = dict(
                                shape=shape, 
                                importance=importance,
                                vectorized=vectorized,
                            )
                    
                    # Raise warning if not and store variable name, shape, val in auto_iv
                    # For CADDEE purposes this will need to be an exception 
                    else:
                        # input_variable = self.declare_variable(name=name, val=val, shape=shape)
                        # self._auto_iv.append((name, val, shape))
                        # self.module_declared_vars[name] = dict(
                        #     shape=shape, 
                        #     importance=importance)
                        # warnings.warn((f"CSDL variable '{name}' is neither a user-defined input (specified with the 'set_module_input' method)")
                        #               (f"nor an output that is computed upstream (all upstream outputs: {self._module_output_names}).")
                        #               (f"This variable will by of type 'DeclaredVariable' with shape {shape} and value {val}"))

                        print(self.module_inputs.keys())
                        print('\n')
                        error_message = f"One or more unknown or missing user-defined variable(s) {list(self.module.inputs.keys())}. "\
                                        f"The developer of module '{type(self)}' has specified variable '{name}' as an input to their model, "\
                                        "which requires the user to set this variable with 'set_module_input' or it needs to "\
                                        f"be computed (and connected) from an upstream model {self._module_output_names}."
                        raise Exception(error_message)
                # else: the variable is set by the user via 'set_module_input'
                else:
                    mod_var = self.module.inputs[name]
                    mod_var_val = mod_var['val']
                    
                    # Check whether the sizes of the set module input and the to be 
                    # created/declares CSDL variable match in size
                    if not isinstance(mod_var_val, (float, int)) and np.size(mod_var_val) != np.prod(shape):
                        raise Exception(f'Size mismatch- module input {name} has size {np.size(mod_var)} but the corresponding csdl variable has shape {shape}')
                    elif np.size(mod_var_val) == 1:
                        pass
                    else:
                        mod_var_val = mod_var_val.reshape(shape)
                    
                    # Check whether variable is a float, int or array and assign 
                    # shape accordingly
                    if isinstance(mod_var_val, (float, int)) and shape == (1, ):
                        mod_var_shape = (1, )
                    elif isinstance(mod_var_val, (float, int)) and shape != (1, ): # corresponds to expanding a scaler to another scaler
                        mod_var_shape = shape
                    else:
                        mod_var_shape = mod_var_val.shape
                            
                    mod_var_units = mod_var['units']    

                    if mod_var['dv_flag'] is False:
                        if self.prepend:
                            var_name = f'{self.prepend}_{name}'
                            input_variable = self.create_input(
                                name=var_name,
                                val=mod_var_val,
                                shape=mod_var_shape,
                                units=mod_var_units,
                                desc=desc,
                            )
                            self.module_inputs[var_name] = dict(
                                shape=shape, 
                                importance=importance,
                                vectorized=vectorized,
                            )
                        else:
                            input_variable = self.create_input(
                                name=name,
                                val=mod_var_val,
                                shape=mod_var_shape,
                                units=mod_var_units,
                                desc=desc,
                            )
                            self.module_inputs[name] = dict(
                                shape=shape, 
                                importance=importance,
                                vectorized=vectorized,
                            )


                    elif mod_var['dv_flag'] is True:
                        
                        if self.prepend:
                            if 'rpm' in name:
                                print('prepend')
                                print(name)
                                print(f'{self.prepend}_{name}')
                                # exit()
                            var_name = f'{self.prepend}_{name}'
                            input_variable = self.create_input(
                                name=var_name,
                                val=mod_var_val,
                                shape=mod_var_shape,
                                units=mod_var_units,
                                desc=desc,
                            )
                            lower = mod_var['lower']
                            upper = mod_var['upper']
                            scaler = mod_var['scaler']
                            self.add_design_variable(var_name, lower=lower, upper=upper, scaler=scaler)
                            self.module_inputs[var_name] = dict(
                                shape=shape, 
                                importance=importance,
                                vectorized=vectorized,
                            )
                        else:
                            input_variable = self.create_input(
                                name=name,
                                val=mod_var_val,
                                shape=mod_var_shape,
                                units=mod_var_units,
                                desc=desc,
                            )
                            lower = mod_var['lower']
                            upper = mod_var['upper']
                            scaler = mod_var['scaler']
                            self.add_design_variable(name, lower=lower, upper=upper, scaler=scaler)
                            self.module_inputs[name] = dict(
                                shape=shape, 
                                importance=importance,
                                vectorized=vectorized,
                            )

                    else:
                        raise NotImplementedError
        
        # else: if no module is provided
        # In this case, all variables will be declared variables 
        else:
            # print('exit')
            # print(name)
            # exit()
            # if promotes is True:
            #     self.promoted_vars.append(name)
                
            if self.prepend:
                var_name = f'{self.prepend}_{name}'
                input_variable = self.declare_variable(
                    name=var_name, 
                    val=val, 
                    shape=shape, 
                    units=units, 
                    desc=desc,
                )
                self.module_declared_vars[var_name] = dict(
                    shape=shape, 
                    importance=importance,
                    vectorized=vectorized,
                )
            else: 
                input_variable = self.declare_variable(
                    name=name, 
                    val=val, 
                    shape=shape, 
                    units=units, 
                    desc=desc,
                )
                self.module_declared_vars[name] = dict(
                    shape=shape, 
                    importance=importance,
                    vectorized=vectorized,
                )

        return input_variable
    
    def register_module_output(
            self, 
            name: str, 
            var=None, 
            shape=None,
            importance=0,
            val=1.0,
            promotes=True,
        ):
        """
        Register a module variable as a csdl output variable.
        Calls the method `register_output` of the csdl `Model` class or 
        the `create_ouput` method. 

        Parameters
        ----------
        `name : str`
            String name of the module and csdl variable

        `var : Output`
            csdl output variable

        `promotes : bool`
            If true, the registered output will be promoted if the model that 
            contains the output is added as a submodel. The default is `False`.
        
        `shape : Tuple[int]`
            The shape of the output. Can only be `not None` if the user wants to
            create the csdl variable with the `create_output` method of the csdl
            `Model` class. Otherwise the shape of the output variable will be 
            determined by the variables used in the operations that compute the 
            output. The default is `None`
        """

        # Check if the method key words set by the user make sense
        if var is None and shape is None:
            raise Exception(f"Set 'mod_var' key word or specify a shape to create an output that is indexable")
        elif var is not None and shape is not None:
            raise Exception(f"Attempting to register {name} as an output while specifying a shape. A shape can only be specified if 'mod_var' is 'None'.")
        elif var is None and shape is not None:
            output_variable = self.create_output(name=name, val=val, shape=shape)
            self.module_outputs[name] = dict(
                shape=shape,
                importance=importance,
            )
        else: 
            output_variable = self.register_output(name=name, var=var)
            self.module_outputs[name] = dict(
                shape=var.shape,
                importance=importance,
            )
        
        # if promotes is True:
        #     print('PROMOTES is TRUE')
        #     self.promoted_vars.append(name)
        # else:
        #     pass
        
        return output_variable
    
    def add_module(
            self,
            submodule,
            name,
            promotes=None,
            increment : int = 1
        ):

        if self.id == 0:
            pass
        elif self.id == 1:
            increment += self.id
            # raise Exception('Weird nesting of ModuleCSDL sub-classes.')
        elif self.id > 0:
            increment += (self.id - 1)
        else:
            raise NotImplementedError
        """
        Add a submodule to a parent module.

        Calls the `add` method of the csld `Model` class.
        """
        GraphRepresentation(submodule)
        
        # Need to increment each input and output of the sub_module 
        for input in submodule.module_inputs.values():
            if input['importance'] != 0:
                input['importance'] += increment
        
        for declared_var in submodule.module_declared_vars.values():
            if declared_var['importance'] != 0:
                declared_var['importance'] += increment 

        for output in submodule.module_outputs.values():
            # print('output', output['importance'])
            if output['importance'] != 0:
                output['importance'] += increment 
        
        # 1) Only promote a subset of user-defined variables
        if promotes is not None:
            # if name == 'vast_fluid_model':
            #     print('wing_vlm promoted_vars', submodule.promoted_vars)
            #     print('self.promoted_vars', self.promoted_vars)
            #     print('sub_modules', submodule.sub_modules)
            #     exit()
            self.add(submodule, name, promotes=promotes+submodule.promoted_vars)
            self.promoted_vars += promotes + submodule.promoted_vars
            self.sub_modules[name] = dict(
                inputs=submodule.module_inputs,
                declared_vars=submodule.module_declared_vars,
                outputs=submodule.module_outputs,
                promoted_vars=promotes+submodule.promoted_vars,
                submodules=submodule.sub_modules,
                auto_iv=submodule._auto_iv,
            )
        
        # 2) Promote the entire submodel
        else:
            # if name == 'VLM_system':
            #     print('VLM_system promoted_vars', submodule.promoted_vars)
            #     print('self.promoted_vars', self.promoted_vars+ list(submodule.module_inputs.keys()))
                # exit()
            self.add(submodule, name)
            self.promoted_vars += submodule.promoted_vars
            # self.promoted_vars +=  list(submodule.module_inputs.keys()) + list(submodule.module_outputs.keys())
            self.sub_modules[name] = dict(
                inputs=submodule.module_inputs,
                declared_vars=submodule.module_declared_vars,
                outputs=submodule.module_outputs,
                promoted_vars=list(submodule.module_declared_vars.keys()) + list(submodule.module_inputs.keys()) + list(submodule.module_outputs.keys()),
                submodules=submodule.sub_modules,
                auto_iv=submodule._auto_iv,
            )
        # print('sub_module', self.sub_modules)

    def connect_modules(self, a: str, b: str):
        """
        Connect variables between modules. 

        Calls the `connect` method of the csdl `Model` class.
        """
        self.connect(a, b)


    def visualize_implementation(self, importance=0, show_outputs=True):
        from  pyxdsm.XDSM import (
            XDSM,
            OPT,
            SUBOPT,
            SOLVER,
            DOE,
            IFUNC,
            FUNC,
            GROUP,
            IGROUP,
            METAMODEL,
            LEFT,
            RIGHT,
        )
        def find_up_stream_module(list_of_tuples, inputs_from_upstream):
            modules = []
            for input_from_upstream in inputs_from_upstream:
                for module, output in list_of_tuples:
                    if input_from_upstream == output:
                        modules.append(module)
            return modules

        def find_inputs_outputs(dictionary, inp_out_list=[], inp_out='inputs'):
            for k, v in dictionary.items():
                if k == inp_out:
                    for k2, v2 in v.items():
                        if v2['importance'] <= (importance+1):
                            inp_out_list.append(k2)
                        else:
                            pass
                    else:
                        pass
                elif isinstance(v, dict):
                    if inp_out_list:
                        find_inputs_outputs(v, inp_out_list, inp_out=inp_out)
                    else:
                        find_inputs_outputs(v, inp_out_list, inp_out=inp_out)
                else:
                    pass
            
            return inp_out_list
        # TODO: dsm connections for nesting 
        def unpack_sub_modules(dictionary, importance_outputs = [], sub_modules = []):
            for module, module_values in dictionary.items():
                # Collect all module outputs with importance less than
                # or equal to the importance specified by the user
                importance_outputs_copy = importance_outputs.copy()
                for sub_mod_out, sub_mod_out_par in module_values['outputs'].items():
                    if sub_mod_out_par['importance'] > 0 and sub_mod_out_par['importance']  <= importance:
                        importance_outputs.append((module, sub_mod_out))
                # If there are "important" outputs call 'add_system'
                if set(importance_outputs) != set(importance_outputs_copy):
                    x.add_system(module, FUNC, generate_dsm_text(module))
                    sub_modules.append((module, module_values))
                    if show_outputs is True:
                        x.add_output(module, [generate_dsm_text(output) for output in  list(set([t[1] for t in importance_outputs]) & set(list(module_values['outputs'].keys())))], side=RIGHT)
                    
                    # Add any inputs defined by the user
                    inputs_from_user = list(set(user_module_inputs) & set(list(module_values['inputs'].keys()) + list(module_values['declared_vars'].keys())))
                    if inputs_from_user:
                        x.add_input(module, [generate_dsm_text(connection) for connection in inputs_from_user], label_width=2)
                    # inputs_auto_iv = list(set(auto_iv_inputs) & set(list(module_values['inputs'].keys())))
                    # if inputs_auto_iv:
                    #     x.add_input(module, [generate_dsm_text(connection) for connection in inputs_auto_iv], faded=True)
                    
                    # Check if there are any inputs from upstream modules
                    inputs_from_upstream = list(set(list(module_values['declared_vars'].keys())) & set([t[1] for t in importance_outputs]))
                    # print('inputs_from_upstream', inputs_from_upstream)
                    # print('module', module)
                    # print('module_outputs', list(module_values['outputs'].keys()))
                    # print('\n')
                    if inputs_from_upstream:
                        upstream_modules = find_up_stream_module(importance_outputs, inputs_from_upstream)
                        for upstream_module in upstream_modules:
                            x.connect(upstream_module, module, [generate_dsm_text(connection) for connection in inputs_from_upstream])
                if module_values['submodules']:
                    unpack_sub_modules(module_values['submodules'], importance_outputs=importance_outputs)
            
            # print('importance_outputs', importance_outputs)
            if not importance_outputs:
                raise Exception('All registered outputs have zero importance or have higher importance than specified by user')

            return module, module_values


        x = XDSM()
        from examples.viz_test import generate_dsm_text
        
        # print(find_inputs_outputs(self.sub_modules, inp_out_list=[], inp_out='inputs'))
        # print(find_inputs_outputs(self.sub_modules, inp_out_list=[], inp_out='outputs'))
        # exit()
        
        user_module_inputs =  find_inputs_outputs(self.sub_modules, inp_out_list=[], inp_out='inputs') + list(self.module_inputs.keys())
        # auto_iv_inputs = find_inputs_outputs(self.sub_modules,inp_out_list=[], inp_out='auto_iv')
        # print('user_module_inputs', user_module_inputs)
        # exit()
                                           
        if importance == 0:
            # print('NAME', self.module_inputs)
            x.add_system(self.name, OPT, generate_dsm_text(self.name))
            parent_module_inputs = list(self.module_inputs.keys())
            parent_module_outputs = list(self.module_outputs.keys())
            if not parent_module_inputs:
                parent_module_inputs = user_module_inputs
            if not parent_module_outputs:
                parent_module_outputs = find_inputs_outputs(self.sub_modules[list(self.sub_modules)[-1]], inp_out_list=[],inp_out='outputs')

            # print('inputs_from_user', ",\,".join([generate_dsm_text(connection) for connection in parent_module_inputs]))
            x.add_input(self.name, [generate_dsm_text(input) for input in parent_module_inputs], label_width=2)
            x.add_output(self.name, [generate_dsm_text(input) for input in parent_module_outputs], side=RIGHT)
        
        else:
            module, module_values = unpack_sub_modules(self.sub_modules)
            print(module_values)
            # for module, module_values in self.sub_modules.items():
                # pass

            # Lastly, show outputs of last module
            # x.add_output(module, [generate_dsm_text(output) for output in list(module_values['outputs'].keys())], side=RIGHT)
            # print(list(module_values['outputs'].keys()))
        # exit()
        x.write(f'{self.name}_xdsm')


    


