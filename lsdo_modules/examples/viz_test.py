greek_letters = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", 
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", 
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", 
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", 
    "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
]

special_characters = ['max', 'min']

def rlo(string, old_substring, new_substring):
    last_occurrence_index = string.rfind(old_substring)
    if last_occurrence_index == -1:
        return string  # Sub-string not found in string
    else:
        return string[:last_occurrence_index] + new_substring + string[last_occurrence_index + len(old_substring):]


def generate_dsm_text(s):
    print(s)
    space_split = s.split(" ")
    return_string = r""
    for string in space_split:
        new_string = string.split("_") # Non underscore 
        if len(new_string) == 1:
            if string in greek_letters:
                return_string += f" \{string}"
            else: 
                return_string += r" \text{ " + string + r"}"
        else:
            counter = 0
            for math_string in new_string:
                if counter == len(new_string)-1:
                    if math_string in greek_letters:
                        return_string += f"\{math_string}" + r"}"*counter
                    elif len(math_string) == 1 or math_string.isnumeric() is True or math_string in special_characters: 
                        print('return string', [return_string, math_string])
                        return_string += math_string + r"}"*counter
                    else:
                        print('return string', [return_string, math_string])
                        return_string = rlo(return_string, "_{", " ") +   math_string
                    break
                else:
                    if math_string in greek_letters:
                        return_string += f"\{math_string}" + r"_{"
                    elif counter == 0:
                        return_string += math_string + r"_{"
                    elif len(math_string) == 1 or math_string.isnumeric() is True:
                        return_string += math_string + r"_{"
                    else:
                        return_string += math_string + r" "
                counter += 1

    if return_string.count('{') != return_string.count('}'):
        return_string = return_string.replace('{', '')
        return_string = return_string.replace('}', '')
        return_string = return_string.replace('_', ' ')
        print(return_string)
    return return_string.replace(" ", "\,")

# print(generate_dsm_text("dummy_output_2"))

# exit()

# def rlo(string, old_substring, new_substring):
#     last_occurrence_index = string.rfind(old_substring)
#     if last_occurrence_index == -1:
#         return string  # Sub-string not found in string
#     else:
#         return string[:last_occurrence_index] + new_substring + string[last_occurrence_index + len(old_substring):]


# def count_characters(s):
#     space_split = s.split(" ")
#     return_string = r""
#     for string in space_split:
#         new_string = string.split("_") # Non underscore 
#         if len(new_string) == 1:
#             if string in greek_letters:
#                 return_string += f" \{string}"
#             else: 
#                 return_string += r" \text{ " + string + r"}"
#         else:
#             counter = 0
#             for math_string in new_string:
#                 if counter == len(new_string)-1:
#                     if math_string in greek_letters:
#                         return_string += f"\{math_string}" + r"}"*counter
#                     elif len(math_string) == 1 or math_string.isnumeric() is True:
#                         return_string += math_string + r"}"*counter
#                     else:
#                         num_brackets = return_string.count("_{")
#                         print('num_brackets', num_brackets)
#                         if num_brackets > 1:
#                             return_string =  rlo(return_string, "_{", "}"*(counter-1)) + " " + math_string
#                         else:
#                             return_string += "} " + math_string
#                             # return_string.replace(r"_{", " ",-2)  + math_string 
#                     break
#                 else:
#                     if math_string in greek_letters:
#                         return_string += f"\{math_string}" + r"_{"
#                     elif len(math_string) == 1 or math_string.isnumeric is True:
#                         print('elif', [return_string, math_string])
#                         if return_string[-1] != "{":
#                             return_string += r"_{" + math_string
#                         else:    
#                             return_string += math_string + r"_{" 
#                     else:
#                         print('else', [return_string, math_string])
#                         return_string = rlo(return_string, "_{", " ") + math_string 
#                 counter += 1
 
#         # print(new_string==string)
#     # if underscore_split:
#     #     print(underscore_split)
#     #     pass
#     # elif space_split:
#     #     string = r"\text{" + space_split + "}"

#     # for i in range(len(parts)-1):
#     #     left_count = len(parts[i])
#     #     right_count = len(parts[i+1])
#     #     counts.append((left_count, right_count))
#     return return_string

# print(count_characters("x_alpha_module"))

# def convert_snake_case(s):
#     parts = s.split("_")
#     new_parts = []
#     for part in parts:
#         if len(part) > 1:
#             if part in greek_letters:
#                 new_parts.append(f"\{part}")
#             new_parts.append(part)
#         else:
#             new_parts.append(f"\\mathrm{{{part}}}")
#     return " ".join(new_parts)

# # 'a $\\mathrm{1}$'
# exit()

# from pyxdsm.XDSM import (
#     XDSM,
#     OPT,
#     SUBOPT,
#     SOLVER,
#     DOE,
#     IFUNC,
#     FUNC,
#     GROUP,
#     IGROUP,
#     METAMODEL,
#     LEFT,
#     RIGHT,
# )

# x = XDSM(
#     # auto_fade={
#     #     # "inputs": "none",
#     #     "outputs": "connected",
#     #     "connections": "outgoing",
#     #     # "processes": "none",
#     # }
# )

# x.add_system("opt", OPT, r"\text{Optimizer}")
# x.add_system("DOE", DOE, r"\text{DOE}")
# x.add_system("MDA", SOLVER, r"\text{Newton}")
# x.add_system("D1", FUNC, "D_1")

# # can fade out blocks to allow for emphasis on sub-sections of XDSM
# x.add_system("D2", IFUNC, "D_2", faded=True)

# x.add_system("D3", IFUNC, "D_3")
# x.add_system("subopt", SUBOPT, "SubOpt", faded=True)
# x.add_system("G1", GROUP, "G_1")
# x.add_system("G2", IGROUP, "G_2")
# x.add_system("MM", METAMODEL, "MM")

# # if you give the label as a list or tuple, it splits it onto multiple lines
# x.add_system("F", FUNC, ("F", r"\text{Functional}"))

# # stacked can be used to represent multiple instances that can be run in parallel
# x.add_system("H", FUNC, "H", stack=True)

# x.add_process(
#     ["opt", "DOE", "MDA", "D1", "D2", "subopt", "G1", "G2", "MM", "F", "H", "opt"],
#     arrow=True,
# )

# x.connect("opt", "D1", ["x", "z", "y_2"], label_width=2)
# x.connect("opt", "D2", ["z", "y_1"])
# x.connect("opt", "D3", "z, y_1")
# x.connect("opt", "subopt", "z, y_1")
# x.connect("D3", "G1", "y_3")
# x.connect("subopt", "G1", "z_2")
# x.connect("subopt", "G2", "z_2")
# x.connect("subopt", "MM", "z_2")
# x.connect("subopt", "F", "f")
# x.connect("MM", "subopt", "f")
# x.connect("opt", "G2", "z")
# x.connect("opt", "F", "x, z")
# x.connect("opt", "F", "y_1, y_2")

# # you can also stack variables
# x.connect("opt", "H", "y_1, y_2", stack=True)

# x.connect("D1", "opt", r"\mathcal{R}(y_1)")
# x.connect("D2", "opt", r"\mathcal{R}(y_2)")

# x.connect("F", "opt", "f")
# x.connect("H", "opt", "h", stack=True)

# # can specify inputs to represent external information coming into the XDSM
# x.add_input("D1", "P_1")
# x.add_input("D2", "P_2")
# x.add_input("opt", r"x_0", stack=True)

# # can put outputs on the left or right sides
# x.add_output("opt", r"x^*, z^*", side=RIGHT)
# x.add_output("D1", r"y_1^*", side=LEFT)
# x.add_output("D2", r"y_2^*", side=LEFT)
# x.add_output("F", r"f^*", side=RIGHT)
# x.add_output("H", r"h^*", side=RIGHT)
# x.add_output("opt", r"y^*", side=LEFT)

# x.add_process(["output_opt", "opt", "left_output_opt"])

# x.write("kitchen_sink", cleanup=False)
# x.write_sys_specs("sink_specs")

# import numpy as np
# import matplotlib.pyplot as plt

# try:
#     from pyxdsm.XDSM import XDSM,  OPT, SOLVER, FUNC
# except ImportError:
#     print("pyXDSM package not found. Please install it using `pip install pyXDSM`.")
    

# def dict_to_xdsm(data, filename='xdsm', show_browser=True):
#     """
#     Convert a dictionary to an XDSM diagram.

#     Parameters
#     ----------
#     data : dict
#         Dictionary representing the XDSM diagram.
#         It should have the following structure:
#         {
#             'nodes': {
#                 'node1': {'shape': 'box', 'label': 'Node 1'},
#                 'node2': {'shape': 'ellipse', 'label': 'Node 2'},
#                 ...
#             },
#             'edges': [
#                 ('node1', 'node2', {'label': 'Edge Label'}),
#                 ...
#             ]
#         }
#     filename : str, optional
#         Name of the output file (without extension).
#     show_browser : bool, optional
#         If True, show the XDSM diagram in the default web browser.

#     Returns
#     -------
#     None
#     """
    

#     # Create the XDSM object
#     xdsm = XDSM()

#     # Add the nodes
#     for name, props in data['nodes'].items():
#         xdsm.add_system(name, **props)

#     # Add the edges
#     for src, tgt, props in data['edges']:
#         xdsm.connect(src, tgt, **props)

#     # Write the XDSM file
#     xdsm.write(filename, True)

#     # Show the XDSM diagram in the browser
#     # if show_browser:
#     #     xdsm.display()


# data = {
#     'nodes': {
#         'node1': {'style': OPT, 'label': 'Node 1'},
#         'node2': {'style': SOLVER, 'label': 'Node 2'},
#     },
#     'edges': [
#         ('node1', 'node2', {'label': 'Edge Label'}),
#     ]
# }

# dict_to_xdsm(data=data)

# from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# # Change `use_sfmath` to False to use computer modern
# x = XDSM(use_sfmath=True)

# x.add_system("opt", OPT, r"\text{Optimizer}")
# x.add_system("solver", SOLVER, r"\text{Newton}")
# x.add_system("D1", FUNC, "D_1")
# x.add_system("D2", FUNC, "D_2")
# x.add_system("F", FUNC, "F")
# x.add_system("G", FUNC, "G")

# x.connect("opt", "D1", "x, z")
# x.connect("opt", "D2", "z")
# x.connect("opt", "F", "x, z")
# x.connect("solver", "D1", "y_2")
# x.connect("solver", "D2", "y_1")
# x.connect("D1", "solver", r"\mathcal{R}(y_1)")
# x.connect("solver", "F", "y_1, y_2")
# x.connect("D2", "solver", r"\mathcal{R}(y_2)")
# x.connect("solver", "G", "y_1, y_2")

# x.connect("F", "opt", "f")
# x.connect("G", "opt", "g")

# x.add_output("opt", "x^*, z^*", side=LEFT)
# x.add_output("D1", "y_1^*", side=LEFT)
# x.add_output("D2", "y_2^*", side=LEFT)
# x.add_output("F", "f^*", side=LEFT)
# x.add_output("G", "g^*", side=LEFT)
# x.write("mdf")


# def dict_to_xdsm(d, output_file=None):
#     """
#     Convert a dictionary to a XDSM diagram and optionally save it to a file.
    
#     Args:
#         d (dict): The dictionary to convert to a XDSM.
#         output_file (str): The name of the file to save the XDSM to. Defaults to None.
        
#     Returns:
#         str: The XDSM diagram as a string.
#     """
    
#     # Import the necessary libraries
#     from pyxdsm.XDSM import XDSM
    
#     # Create the XDSM object
#     xdsm = XDSM()
    
#     # Loop through the dictionary and add each component
#     for comp_name, comp_dict in d.items():
#         if "comp" in comp_dict:
#             xdsm.add_system(comp_name, comp_dict["comp"])
#         elif "group" in comp_dict:
#             xdsm.add_process(comp_name, comp_dict["group"])
    
#     # Loop through the dictionary again and add each connection
#     for from_name, from_dict in d.items():
#         for to_name, to_dict in from_dict.get("connect", {}).items():
#             xdsm.add_input_output(from_name, to_name, from_dict["var"], to_dict["var"])
    
#     # Generate the XDSM diagram
#     xdsm.write(output_format="tex")
    
#     # If an output file was specified, save the XDSM diagram to the file
#     if output_file:
#         xdsm.write(output_format="pdf", output_name=output_file)
    
#     # Return the XDSM diagram as a string
#     return xdsm.tex

# from collections import defaultdict
# from graphviz import Digraph

# def dict_to_xdsm(data, filename='xdsm'):
#     nodes = defaultdict(set)
#     for key in data:
#         for item in data[key]:
#             nodes[key].add(item)
#             nodes[item]  # ensure the item is in the nodes dict even if it has no connections

#     # create graph
#     dot = Digraph(comment='XDSM Diagram', filename=filename, engine='dot')

#     # add nodes
#     for node in nodes:
#         dot.node(node, node)

#     # add edges
#     for key in data:
#         for item in data[key]:
#             dot.edge(key, item)

#     # render graph
#     dot.render(view=True)

# data = {
#     'A': ['B', 'C'],
#     'B': ['A','D'],
#     'C': ['D'],
#     'D': ['E'],
#     'E': []
# }

# dict_to_xdsm(data, filename='my_xdsm')


# import numpy as np
# import matplotlib.pyplot as plt

# def create_xdsm(diagram_dict):
#     box_width = 0.3
#     box_height = 0.2
#     box_sep = 0.1
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_axis_off()
#     x_pos = {}
#     y_pos = {}
#     i = 0
#     for node in diagram_dict['nodes']:
#         x_pos[node] = 0
#         y_pos[node] = i
#         ax.text(x_pos[node] - 0.5 * box_width, y_pos[node] + 0.5 * box_height, node, fontsize=12)
#         i += 1
#     num_nodes = i
#     for i, group in enumerate(diagram_dict['edges']):
#         for j, conn in enumerate(group):
#             from_node = conn[0]
#             to_node = conn[1]
#             arrow_style = '-|>' if conn[2] == 'out' else '<|-'
#             x_start = x_pos[from_node] + 0.5 * box_width
#             y_start = y_pos[from_node] - 0.5 * box_height
#             x_end = x_pos[to_node] - 0.5 * box_width
#             y_end = y_pos[to_node] - 0.5 * box_height
#             ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, lw=1, head_width=0.05, head_length=0.05,
#                      length_includes_head=True, shape=arrow_style, color='k')
#     plt.ylim(-num_nodes * (box_height + box_sep), box_height)
#     plt.xlim(-0.5, max(x_pos.values()) + 0.5)
#     plt.show()


# d = { 'nodes' : {
#     'A':  ['B', 'C'],
#     'B': ['D'],
#     'C': ['D', 'E'],
#     'D': ['F'],
#     'E': ['F'],
#     'F': [],
#     },
# }

# create_xdsm(d)

# import graphviz

# def dict_to_xdsm(d, filename):
#     dot = graphviz.Digraph()
    
#     # Add all the nodes
#     for node in d.keys():
#         dot.node(node)
    
#     # Add all the edges
#     for source, targets in d.items():
#         for target in targets:
#             dot.edge(source, target)
    
#     # Render the graph and save it to a file
#     dot.render(filename, format='png')

# d = {
#     'A': ['B', 'C'],
#     'B': ['D'],
#     'C': ['D', 'E'],
#     'D': ['F'],
#     'E': ['F'],
#     'F': []
# }

# import os
# file_name= os.getcwd()+'/' + 'example.png'
# dict_to_xdsm(d, file_name)

