import inspect
from Utils import utilsHelper

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "FunctionOneV2": {
            "required_kwargs": [
                {"name": "rkw_1", "description": "Value(s) to be converted to score"},
                {"name": "rkw_2", "description": "lower bound"}
            ],
            "optional_kwargs": [
                {"name": "okw_1", "description": "Score the object will be given if it's outside the bounds, default 0"}
            ],
            "help_string": "Score value(s) as 1 or outside_bounds_score (default 0) depending on whether they're within a bound or not."
        },
        "FunctionTwoV2": {
            "required_kwargs": [
                {"name": "rkw_1", "description": "Value(s) to be converted to score"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Score value(s) based on a gaussian profile with given mean, sigma - maximum score of 1 is possible."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def FunctionOneV2(**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    if "okw_1" in provided_optional_args:
        okwarg1 = float(kwargs["okw_1"])
    else:
        #Default okwarg1 value
        okwarg1 = 1
    print('Function one ran!')
    return (float(kwargs["rkw_1"])+float(kwargs["rkw_2"]))*okwarg1

def FunctionTwoV2(**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    print('Function Two ran!')
    return (kwargs["rkw_1"])
