import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time

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
            "help_string": "Returns a test localization dataframe and metadata string, to check if the program is working properly."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def FunctionOne(candidate_dic,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    # Start the timer
    start_time = time.time()

    # generate the localizations from a candidate dictionary
    localizations = pd.DataFrame()
    for i in np.unique(list(candidate_dic)):
        localizations['candidate_id'] = i
        localizations['x'] = candidate_dic[i]['events']['x']
        localizations['y'] = candidate_dic[i]['events']['y']
        localizations['t'] = candidate_dic[i]['events']['t']
    
    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    performance_metadata = f"Dummy function ran for {elapsed_time} seconds."
    print('Function one ran!')

    return localizations, performance_metadata
