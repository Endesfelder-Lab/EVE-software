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
        "PolarityMatching": {
            "required_kwargs": [
                {"name": "Max_xyDistance", "description": "Maximum distance in x,y in nm units (nm)","default":"50","type":float,"display_text":"Max. XY distance (nm)"},
                {"name": "Max_tDistance", "description": "Maximum time distance in ms (between positive and negative)","default":"500","type":float,"display_text":"Max time distance (ms)"},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Link and filter on positive and negative events."
        },
        "PolarityMatching_NeNA": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Perform NeNA on the matched polarities."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def PolarityMatching(localizations,findingResult,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    
    #Error message and early exit if there isn't both pos and neg events
    if not np.array_equal(np.unique(localizations['p']), [0, 1]) or np.array_equal(np.unique(localizations['p']), [1, 0]):
        logging.error('PolarityMatching requires both positive and negative events!')
        return localizations, 'PolarityMatching requires both positive and negative events!'
    else:
        #add empty columns to localizations:
        localizations['pol_link_id'] = -1
        localizations['pol_link_time'] = 0
        localizations['pol_link_xy'] = 0
        
        #Get the pos and neg events
        posEvents = localizations[localizations['p']==1]
        negEvents = localizations[localizations['p']==0]
        
        mininAll = np.searchsorted(negEvents['t'], posEvents['t'])
        maxinAll = np.searchsorted(negEvents['t'], posEvents['t'] + float(kwargs['Max_tDistance']))

        #We loop over the positive events:
        for posEventId,posEvent in posEvents.iterrows():
            if np.mod(posEventId,500) == 0:
                logging.info('PolarityMatching progress: ' + str(posEventId) + ' of ' + str(len(posEvents)))
            # if posEventId < 5000:
            minin = mininAll[posEventId]
            maxin = maxinAll[posEventId]
        
            negEventsInTime = negEvents[minin:maxin]
            
            x_diff = negEventsInTime['x'].values - posEvent['x']
            y_diff = negEventsInTime['y'].values - posEvent['y']
            distance = np.sqrt(x_diff**2 + y_diff**2)

            foundNegEventId = distance < float(kwargs['Max_xyDistance'])

            #If we found at least one:
            if sum(foundNegEventId) > 0:
                #Find the first id of True (the closest in time)
                foundNegEventId = np.argmax(foundNegEventId)
                #Find the corresponding event
                negEventsWithinDistance = negEventsInTime.iloc[foundNegEventId]
                #And find the event distance belonging to it
                eventDistance = distance[foundNegEventId]
                
                
                # negEventFound = negEventsWithinDistance.iloc[0]

                # pos_index = localizations.index[localizations['candidate_id'] == posEvent.candidate_id].tolist()[0]
                # neg_index = (localizations.index[localizations['candidate_id'] == negEventFound.candidate_id]).tolist()[0]
                
                # localizations.at[pos_index, 'polarity_linked_id'] = negEventFound.candidate_id
                # localizations.at[pos_index, 'polarity_linked_time'] = negEventFound.t - posEvent.t

                # localizations.at[neg_index, 'polarity_linked_id'] = posEvent.candidate_id
                # localizations.at[neg_index, 'polarity_linked_time'] = posEvent.t - negEventFound.t
                
                # if len(negEventsWithinDistance) > 0:
                
                #Renaming
                negEventFound = negEventsWithinDistance
                
                #Update the positive candidate
                localizations.loc[localizations['candidate_id'] == posEvent.candidate_id,'pol_link_id'] = (negEventFound.candidate_id)
                localizations.loc[localizations['candidate_id'] == posEvent.candidate_id,'pol_link_time'] = (negEventFound.t - posEvent.t)
                localizations.loc[localizations['candidate_id'] == posEvent.candidate_id,'pol_link_xy'] = eventDistance
                
                #And update the negative candidate
                localizations.loc[localizations['candidate_id'] == negEventFound.candidate_id,'pol_link_id'] = (posEvent.candidate_id)
                localizations.loc[localizations['candidate_id'] == negEventFound.candidate_id,'pol_link_time'] = (posEvent.t-negEventFound.t)
                localizations.loc[localizations['candidate_id'] == negEventFound.candidate_id,'pol_link_xy'] = eventDistance
                
                
                
                
                
                
                # print(f"{posEvent.id} links with {negEventsWithinDistance.id}")
            
            #We find all negative events with time within Max_tDistance:
            #Find the first entry in negEvents that is later than posEvent:
            # minin = np.argmax(negEvents['t']>posEvent['t'])
            # (negEvents['t']-posEvent['t'])<float(kwargs['Max_tDistance'])
            
            # #We should do this in a more efficient way - they're time sorted, so should be easy? 
            # negEventsInTime = (negEvents['t']-posEvent['t'])
            # negEventsInTime = negEventsInTime[negEventsInTime>0]
            # negEventsInTime = negEventsInTime[negEventsInTime<float(kwargs['Max_tDistance'])]
            # negEventsWithinTime = negEvents[(negEvents['t']-posEvent['t'])<float(kwargs['Max_tDistance'])]
            
            #We find all negative events with distance within Max_xyDistance:
            # negEventsWithinDistance = negEventsWithinTime[(negEventsWithinTime['x']-posEvent['x'])**2+(negEventsWithinTime['y']-posEvent['y'])**2<kwargs['Max_xyDistance']**2]
    
    #Required output: localizations
    metadata = 'Information or so'
    return localizations,metadata

def PolarityMatching_NeNA(localizations,findingResult,settings,**kwargs):
    #Check if we have the pre-run polarity matching:
    if not ('pol_link_id' in localizations.columns and 'pol_link_time' in localizations.columns and 'pol_link_xy' in localizations.columns):
        logging.error('PolarityMatching-NeNA requires to first run Polarity Matching!')
    else:
        #Create a new figure and plot the polarity_linked_xy as a histogram:
        #Pre-filter to only positive events (to only have all links once)
        sublocs = localizations[localizations['p'] == 1]
        #Also remove all -1 pol link xy:
        sublocs = sublocs[sublocs['pol_link_xy']>0]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(sublocs['pol_link_xy'],bins=100)
        plt.show()
    