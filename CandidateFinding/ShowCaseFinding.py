import inspect
from Utils import utilsHelper

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "FunctionOne": {
            "required_kwargs": [
                {"name": "rkwarg_1", "description": "Value(s) to be converted to score"},
                {"name": "rkwarg_2", "description": "lower bound"}
            ],
            "optional_kwargs": [
                {"name": "okwarg_1", "description": "Score the object will be given if it's outside the bounds, default 0"}
            ],
            "help_string": "Score value(s) as 1 or outside_bounds_score (default 0) depending on whether they're within a bound or not."
        },
        "FunctionTwo": {
            "required_kwargs": [
                {"name": "rkwarg_1", "description": "Value(s) to be converted to score"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Score value(s) based on a gaussian profile with given mean, sigma - maximum score of 1 is possible."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def FunctionOne(npyData,globalSettings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    if "okwarg_1" in provided_optional_args:
        okwarg1 = float(kwargs["okwarg_1"])
    else:
        #Default okwarg1 value
        okwarg1 = 1
    print('Function one ran!')
    return (float(kwargs["rkwarg_1"])+float(kwargs["rkwarg_2"]))*okwarg1

def FunctionTwo(npyData,globalSettings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    print('Function Two ran!')
    return (kwargs["rkwarg_1"])
