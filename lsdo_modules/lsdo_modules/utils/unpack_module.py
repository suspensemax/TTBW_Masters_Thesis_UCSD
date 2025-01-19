import pandas as pd
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.output import Output
from csdl.lang.input import Input
from csdl.lang.concatenation import Concatenation
from json2html import *
import os 
import webbrowser

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def unpack_module(lst, compact_print=False, internal_call=False):
    module_dict_1 = {'module' : {
        'Inputs': {},
        'Outputs': {},
        'Submodules' : {},
    }}
    module_dict_2 = {}
    
    for item in lst:
        if isinstance(item, dict):
            submodule = item['sub_module']
            submodule_name = item['name']
            if internal_call is True:
                module_dict_2[submodule_name] = {
                    'Inputs': {},
                    'Outputs': {},
                    'Submodules' : {},
                }
            else:
                module_dict_1['module']['Submodules'][submodule_name] = {
                    'Inputs': {},
                    'Outputs': {},
                    'Submodules' : {},
                }
            inputs = submodule.module_inputs
            outputs = submodule.module_outputs
            for sub_item in submodule.module_info:
                # print(type(sub_item))
                if isinstance(sub_item, dict):
                    if compact_print == True: 
                        pass
                    else: 
                        sub_list = [sub_item]
                        if internal_call is True:
                            module_dict_2[submodule_name]['Submodules'].update(unpack_module(sub_list, internal_call=True))
                        else:   
                            module_dict_1['module']['Submodules'][submodule_name]['Submodules'].update(unpack_module(sub_list, internal_call=True))
                    #module_dict.update(unpack_list(item, key_func, value_func))
                else:
                    var_name = sub_item.name
                    var_shape = sub_item.shape
                    if internal_call is True: 
                        if var_name in inputs:
                            module_dict_2[submodule_name]['Inputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
                        elif var_name in outputs:
                            module_dict_2[submodule_name]['Outputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
                    else:
                        if var_name in inputs:
                            module_dict_1['module']['Submodules'][submodule_name]['Inputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
                        elif var_name in outputs:
                            module_dict_1['module']['Submodules'][submodule_name]['Outputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
        else:
            print('ITEM', item)
            if isinstance(item, DeclaredVariable) or isinstance(item, Input):
                var_name = item.name
                var_shape = item.shape
                module_dict_1['module']['Inputs'][var_name] = {'type' : type(item).__name__, 'shape' : f"{var_shape}"}
           
            elif isinstance(item, Output) or isinstance(item, Concatenation):
                var_name = item.name
                var_shape = item.shape
                module_dict_1['module']['Outputs'][var_name] = {'type' : type(item).__name__, 'shape' : f"{var_shape}"}

    return Merge(module_dict_2, module_dict_1)


# exit()
# def unpack_list(lst, compact_print=False):
#     module_dict = {'parent_module' : {
#               'Inputs': {},
#               'Outputs' : {},
#               'Submodules' : {},
#               }
#     }
#     submodule_name=None
#     for item in lst:
#         if isinstance(item, dict):
#             submodule = item['sub_module']
#             submodule_name = item['name']
#             module_dict['parent_module']['Submodules'][submodule_name] = {
#                 'Inputs': {},
#                 'Outputs': {},
#                 'Submodules' : {},
#             }
#             inputs = submodule.module_inputs
#             outputs = submodule.module_outputs
#             for sub_item in submodule.module_info:
#                 # print(type(sub_item))
#                 if isinstance(sub_item, dict):
#                     if compact_print == True: 
#                         pass
#                     else: 
#                         sub_list = [sub_item]
#                         module_dict['parent_module']['Submodules'][submodule_name]['Submodules'].update(unpack_list(sub_list))
#                     #module_dict.update(unpack_list(item, key_func, value_func))
#                 else:
#                     var_name = sub_item.name
#                     var_shape = sub_item.shape
#                     if var_name in inputs:
#                         module_dict['parent_module']['Submodules'][submodule_name]['Inputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
#                     elif var_name in outputs:
#                         module_dict['parent_module']['Submodules'][submodule_name]['Outputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
#         else:
#             print(item)
#             if isinstance(item, DeclaredVariable) or isinstance(item, Input):
#                 var_name = item.name
#                 var_shape = item.shape
#                 if submodule_name:
#                     module_dict['parent_module']['Submodules'][submodule_name]['Inputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
#                 else:
#                     module_dict['parent_module']['Inputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}

#             elif isinstance(item, Output) or isinstance(item, Concatenation):
#                 var_name = item.name
#                 var_shape = item.shape
#                 if submodule_name:
#                     module_dict['parent_module']['Submodules'][submodule_name]['Outputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}
#                 else:
#                     module_dict['parent_module']['Outputs'][var_name] = {'type' : type(sub_item).__name__, 'shape' : f"{var_shape}"}

#                 pass
#                 # input_types = 
#                 # print('not dict')
#                 # print('name', sub_item.name)
#                 # print('shape', sub_item.shape)
#                 # print('type', type(sub_item).__name__)
#     #             key = key_func(item)
#     #             value = value_func(item)
#     #             if key is not None and value is not None:
#     #                 module_dict[key] = value
#     return module_dict #pd.DataFrame(module_dict)

