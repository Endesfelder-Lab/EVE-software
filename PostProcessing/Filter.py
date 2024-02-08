import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time
import logging

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Regular_filter": {
            "required_kwargs": [
                {"name": "Filter_text", "description": "Filter text. E.g. \"x < 50 & y > 100\". "}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Normal filtering of data."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def Regular_filter(localizations,findingResult,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    orig_len_localizations = len(localizations)
    
    # Start the timer
    start_time = time.time()
    
    filter_text = kwargs['Filter_text']

    image = np.random.rand(100, 100)

    # Stop the timer
    end_time = time.time()


    localizations = localizations[localizations['x']>45000]

    new_len_localizations = len(localizations)
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    logging.info(f'Went from {orig_len_localizations} to {new_len_localizations} localizations in {elapsed_time} seconds.')
    #Required output: localizations
    return localizations
