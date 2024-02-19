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
        localizations['polarity_linked_id'] = -1
        localizations['polarity_linked_time'] = 0
        
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

            negEventsWithinDistance = negEventsInTime[distance < float(kwargs['Max_xyDistance'])]

            if not negEventsWithinDistance.empty:
                negEventFound = negEventsWithinDistance.iloc[0]

                pos_index = localizations.index[localizations['candidate_id'] == posEvent.candidate_id].tolist()[0]
                neg_index = (localizations.index[localizations['candidate_id'] == negEventFound.candidate_id]).tolist()[0]
                
                localizations.at[pos_index, 'polarity_linked_id'] = negEventFound.candidate_id
                localizations.at[pos_index, 'polarity_linked_time'] = negEventFound.t - posEvent.t

                localizations.at[neg_index, 'polarity_linked_id'] = posEvent.candidate_id
                localizations.at[neg_index, 'polarity_linked_time'] = posEvent.t - negEventFound.t
                
                
                
                
                
                
                # if len(negEventsWithinDistance) > 0:
                    
                #     negEventFound = negEventsWithinDistance.iloc[0] #Always the first
                    
                #     #Update the positive candidate
                #     localizations.loc[localizations['candidate_id'] == posEvent.candidate_id,'polarity_linked_id'] = (negEventFound.candidate_id)
                #     localizations.loc[localizations['candidate_id'] == posEvent.candidate_id,'polarity_linked_time'] = (negEventFound.t - posEvent.t)
                    
                #     #And update the negative candidate
                #     localizations.loc[localizations['candidate_id'] == negEventFound.candidate_id,'polarity_linked_id'] = (posEvent.candidate_id)
                #     localizations.loc[localizations['candidate_id'] == negEventFound.candidate_id,'polarity_linked_time'] = (posEvent.t-negEventFound.t)
                
                
                
                
                
                
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
