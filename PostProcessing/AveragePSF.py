import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time
import logging
import copy
import pickle
from datetime import date

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

    #Explanation of columns in the dataframe:
    #x: x position of the event
    #y: y position of the event
    #t: time of the event
    #p: probability of the event
    #event_id: ID for sequential order of events in cluster
    #candidate_id: ID of the candidate the event belongs to
    #pixel_incident_count: The number of events that have been detected in the same pixel as this event, including this event (a count of 5 means that this is the fifth event in this pixel in this candidate)
    #pixel_incidence_tot: The total number of events that have been detected in the same pixel as this event (a count of 10 means there are 10 events in this pixel in this candidate)
    #t_delay: The time delay of the event from the previous event on the same pixel (in us)

    sumAllEventsPos = pd.DataFrame(columns=['x', 'y', 't', 'p', 'event_id', 'candidate_id', 'pixel_incident_count', 'pixel_incidence_tot', 't_delay'])
    sumAllEventsNeg = pd.DataFrame(columns=['x', 'y', 't', 'p', 'event_id', 'candidate_id', 'pixel_incident_count', 'pixel_incidence_tot', 't_delay'])
    sumAllEventsMix = pd.DataFrame(columns=['x', 'y', 't', 'p', 'event_id', 'candidate_id', 'pixel_incident_count', 'pixel_incidence_tot', 't_delay'])
    
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


        #Now we need to count the events per pixel, at this point as well x and y values are integers:
        candidate_width = events['x'].max() - events['x'].min() + 1
        candidate_height = events['y'].max() - events['y'].min() + 1
        #We create a 2d array of zeros:
        count_grid = np.zeros((candidate_width, candidate_height))
        time_prev_event = np.zeros((candidate_width, candidate_height))

        #These arrays will be added as columns to the events dataframe:
        pixel_incident_count = np.zeros(len(events))
        time_delay = np.zeros(len(events))
        #We loop over all events in the cluster and add 1 to the corresponding pixel:
        for i in range(0,len(events)):
            event = events.iloc[i]
            x_location = event['x'] - events['x'].min()
            y_location = event['y'] - events['y'].min()
            count_grid[x_location, y_location] += 1
            pixel_incident_count[i] = count_grid[x_location, y_location]
            if count_grid[x_location, y_location] > 1:
                time_delay[i] = event['t'] - time_prev_event[x_location, y_location]
            else:
                time_delay[i] = 0
            time_prev_event[x_location, y_location] = event['t']


        #We add the pixel incident count to the events:
        events['pixel_incident_count'] = pixel_incident_count.astype(int)

        #We add the time delay to the events:
        events['t_delay'] = time_delay.astype(int)

        #We also add the total pixel incidence to the events:
        total_pixel_incidence_count = np.zeros(len(events))
        
        for i in range(0,len(events)):
            event = events.iloc[i]
            x_location = event['x'] - events['x'].min()  
            y_location = event['y'] - events['y'].min()
            total_pixel_incidence_count[i] = count_grid[x_location, y_location]
        events['pixel_incidence_tot'] = total_pixel_incidence_count.astype(int)


        #Correct the event for the localization x, y, time:l
        events['x'] = events['x'] - localizations.iloc[loc]['x']/float(settings['PixelSize_nm']['value'])
        events['y'] = events['y'] - localizations.iloc[loc]['y']/float(settings['PixelSize_nm']['value'])
        events['t'] = events['t'] - localizations.iloc[loc]['t']*1000
        
        #At this point the events are arranged in time order, so we will add their event_id and candidate_id based on timing information:
        events['event_id'] = np.arange(1, len(events)+1)
        events['candidate_id'] = np.full(len(events), candidate_id)

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
    
    
    eventdict = {}
    eventdict['pos'] = sumAllEventsPos
    eventdict['neg'] = sumAllEventsNeg
    eventdict['mix'] = sumAllEventsMix

    #For now I will store the output in a pickle file to analyze:
    filepath = "data/averagepsf_50sec_" + str(date.today()) + ".pickle"
    with open(filepath, 'wb') as file:
       pickle.dump(eventdict, file)

    fig.show()  

    return '',''