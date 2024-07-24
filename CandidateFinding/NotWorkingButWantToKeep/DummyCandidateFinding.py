import inspect
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
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
            ],
            "optional_kwargs": [
            ],
            "help_string": "Returns a test dictionary and metadata string, to check if the program is working properly."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def FunctionOne(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    # Start the timer
    start_time = time.time()

    # generate the candidates in from of a dictionary
    candidates = {}
    candidates[0] = {}
    candidates[0]['events'] = pd.DataFrame(npy_array)
    candidates[0]['cluster_size'] = [np.max(npy_array['y'])-np.min(npy_array['y']), np.max(npy_array['x'])-np.min(npy_array['x']), np.max(npy_array['t'])-np.min(npy_array['t'])]
    candidates[0]['N_events'] = len(npy_array)
    
    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    performance_metadata = f"Dummy function ran for {elapsed_time} seconds."
    print('Function one ran!')

    return candidates, performance_metadata
