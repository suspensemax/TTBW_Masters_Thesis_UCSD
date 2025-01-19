from lsdo_modules.utils.parameters import Parameters
from abc import ABC, abstractmethod


# NOTE Unpack kwarg dictionary 
class Module(ABC):
    def __init__(self, **kwargs) -> None:
        self.parameters = Parameters()
        self.initialize(kwargs)
        self.parameters.update(kwargs)
        
        self.inputs = dict()
        self.outputs = dict()
        self.promoted_vars = list()
        self.csdl_inputs = list()
    
    @abstractmethod
    def initialize(self, kwargs):
        raise NotImplementedError

    def set_module_input(self, name, val, units='',
                    dv_flag=False, lower=None, upper=None, scaler=None):
        
        self.inputs[name] = dict(
            val=val,
            units=units,
            dv_flag=dv_flag,
            lower=lower,
            upper=upper,
            scaler=scaler,
        )

    def connect(self): pass

    


    # TODO/ thoughts: 
    #   - re-think our convention for setting module inputs (previously caddee inputs)
    #     for components where we also need to provide the design condition 
    #   - it would be more general to always call set_modulue_input on the module itself 
    #   - what about aircraft states? Previously we pre-declared any aircraft states in the
    #     AircraftCondition class and when the use called set_caddee_input on the e.g., 
    #     cruise_condition, caddee could check before setup time whether the input exists 
    #     or not. Example: the line below is called in the initialize method of AircraftCondition
    #       self.variables_metadata.declare(name='range', default=None, types=(int, float), allow_none=True)
    #   - Proposition: introduce something similar to the module class where the user/ developer can 
    #                  declare high-level variables/parameters on the pure python side
    #   - Or         : In the above example, all these declared variables would be moved to the csdl side 
    #                  of the module package 
    #               





