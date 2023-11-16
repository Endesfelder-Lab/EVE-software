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
                {"name": "rkwarg_1", "description": "Value(s) to be converted to score"},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Gives the average xyt position of each found localization."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def AverageXYTpos(candidate_dic,settings,**kwargs):
    
    localizations = {}
    for i in np.unique(list(candidate_dic)):
        localizations[i]={}
        localizations[i]['x'] = np.mean(candidate_dic[i]['events']['x'])
        localizations[i]['y'] = np.mean(candidate_dic[i]['events']['y'])
        localizations[i]['t'] = np.mean(candidate_dic[i]['events']['t'])
    
    #Make a pd dataframe out of it
    localizations = pd.DataFrame(localizations).T
    # print('\nStarting localization list...')
    # tic = time.time()
    # localization_list = np.zeros((len(np.unique(b['DBSCAN'])),9))
    # for cl in np.unique(b['DBSCAN']):
    #     templocs = b[b['DBSCAN']==cl]
    #     mean_x = np.mean(templocs['x']) #just mean x,y
    #     mean_y = np.mean(templocs['y'])
    #     FWHM_x = np.percentile(templocs['x'],75)-np.percentile(templocs['x'],25) #FWHM at x,y
    #     FWHM_y = np.percentile(templocs['y'],75)-np.percentile(templocs['y'],25)
    #     time_start_prc = np.percentile(templocs['t'],5) #5% percentile for start-time
    #     time_end_prc = np.percentile(templocs['t'],95) #5% percentile for end-time of on-event
    #     nr_events_loc = len(templocs)
    #     #Put all in localization_list
    #     localization_list[cl-1,:] = [cl,time_start_prc,mean_x,mean_y,nr_events_loc,1,FWHM_x,FWHM_y,time_end_prc]
    #     #print('Localization '+str(cl)+' at: x: '+str(mean_x)+', y: '+str(mean_y)+', t: '+str(time_start_prc))
    # localizations = cl
    # print(str(cl)+' localizations found!\n')
    # print('Time to find localizations: ', time.time()-tic)
    
    return localizations, ''