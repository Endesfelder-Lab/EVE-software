import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "AverageXYTpos": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Gives the average xyt position of each found localization.",
            "display_name": "Mean X,Y,T, position"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def AverageXYTpos(candidate_dic,settings,**kwargs):
    
    localizations = {}
    for i in np.unique(list(candidate_dic)):
        localizations[i]={}
        localizations[i]['candidate_id'] = i
        localizations[i]['x'] = np.mean(candidate_dic[i]['events']['x'])*float(settings['PixelSize_nm']['value']) #X position in nm
        localizations[i]['y'] = np.mean(candidate_dic[i]['events']['y'])*float(settings['PixelSize_nm']['value']) #Y position in nm
        mean_polarity = np.mean(candidate_dic[i]['events']['p'])
        localizations[i]['p'] = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
        localizations[i]['t'] = np.mean(candidate_dic[i]['events']['t'])/1000. 
        localizations[i]['fit_info'] = ''
        
    #Make a pd dataframe out of it - needs to be transposed
    localizations = pd.DataFrame(localizations).T
    
    return localizations, ''