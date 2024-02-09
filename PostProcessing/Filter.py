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
                {"name": "Filter_text", "description": "Filter text. E.g. \"x < 50 & y > 100\". ","default":"x>0"},
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
    
    #Assumed filter text is in the form of "x < 50 & y > 100".
    #First we clean up all spaces:
    filter_text = filter_text.replace(' ','')
    #Thus, we first split on '&', and then do the (in)equalities.
    filter_textSplit = filter_text.split('&')
    inequality_split={}
    #We loop over every entry
    for filter_entry in range(len(filter_textSplit)):
        possibleInequalities = ['==','<=','=<','>=','=>','!=','>','<']
        inequalityFound = False
        #See if there is an inequality, in this order of inequalities
        #find a specific string:
        for inequality in possibleInequalities:
            if inequality in filter_textSplit[filter_entry]:
                inequalityFound = True
                #Split on this inequality:
                inequality_split = filter_textSplit[filter_entry].split(inequality)
                #Ensure that we have 2 entries
                if len(inequality_split) != 2:
                    logging.warning(f"Expected 2 entries in the inequality expression, but found {len(inequality_split)}")
                else:
                    #check that the first entry is a variable that's found in the localization columns:
                    if not inequality_split[0] in localizations.columns:
                        logging.warning(f"Could not find {inequality_split[0]} in the localization columns! Please format as 'x<100 or y>=100'")
                    else:
                        try:
                            #Now everything should be fine to do the filtering!
                            localizations = eval("localizations[localizations[\'"+inequality_split[0]+"\']"+inequality+inequality_split[1]+"]")
                            logging.info(f"Filtering on {inequality_split[0]} {inequality} {inequality_split[1]}")
                        except:
                            logging.warning(f'Unexpected error with filtering on {filter_textSplit[filter_entry]}')
                break
        if inequalityFound == False:
            logging.warning(f'No inequality found in \"{filter_textSplit[filter_entry]}\"! Please use ==,\<=,\>=,\<,\>,!=')

    # Stop the timer
    end_time = time.time()

    new_len_localizations = len(localizations)
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    logging.info(f'Went from {orig_len_localizations} to {new_len_localizations} localizations in {elapsed_time} seconds.')
    #Required output: localizations
    return localizations
