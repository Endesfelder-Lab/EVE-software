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
        "AveragePSF": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Normal filtering of data."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def AveragePSF(localizations,findingResult,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    sumAllEventsPos = pd.DataFrame(columns=['x', 'y', 't', 'p'])
    sumAllEventsNeg = pd.DataFrame(columns=['x', 'y', 't', 'p'])
    sumAllEventsMix = pd.DataFrame(columns=['x', 'y', 't', 'p'])
    import copy
    #We loop over all localizations:
    for loc in range(0,len(localizations)):
        if np.mod(loc,1000) == 0:
            logging.info(f"Currently at: {loc} of {len(localizations)}, or {100*loc/len(localizations)}%")
        #we grab its candidate id:
        candidate_id = localizations.iloc[loc]['candidate_id']
        #We find the corresponding randidate from findingResult:
        candidate = copy.deepcopy(findingResult[candidate_id])
        #We take the events in there:
        events = candidate['events'].copy()
        
        #Correct the event for the localization x, y, time:
        events['x'] = events['x'] - localizations.iloc[loc]['x']/float(settings['PixelSize_nm']['value'])
        events['y'] = events['y'] - localizations.iloc[loc]['y']/float(settings['PixelSize_nm']['value'])
        events['t'] = events['t'] - localizations.iloc[loc]['t']*1000        
        #check if all p==1:
        if all(events['p'] == 1):
            #append all entries in events to sumalleventspos:
            sumAllEventsPos = pd.concat([sumAllEventsPos, events], ignore_index=True)
        elif all(events['p'] == 0):
            sumAllEventsNeg = pd.concat([sumAllEventsNeg, events], ignore_index=True)
        else:
            sumAllEventsMix = pd.concat([sumAllEventsMix, events], ignore_index=True)
    
    
    #Create a 2d histogram of this and show it:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    #remove nans from sumalleventspos;
    sumAllEventsPos = sumAllEventsPos.dropna()
    ax.hist2d(sumAllEventsPos['x'],sumAllEventsPos['y'],bins=50)
    #second subplot with a hist2d but logarithmic intensity:
    
    fig.show()

    return '',''