import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time, logging
from scipy import spatial

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Finding": {
            "required_kwargs": [
                {"name": "number_of_neighbours", "description": "Required number of neighbouring events in spatiotemporal voxel","default":17},
                {"name": "spatial_radius_outer", "description": "Outer radius (in px) to count the neighbours in.","default":7},
                {"name": "spatial_radius_inner", "description": "Inner radius (in px) to count the neighbours in.","default":1},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35},
            ],
            "optional_kwargs": [
               {"name": "min_consec", "description": "Minimum number of consecutive events","default":1},
               {"name": "max_consec", "description": "Maximum number of consecutive events, discards hot pixels","default":30},
            ],
            "help_string": "Returns a test dictionary and metadata string, to check if the program is working properly."
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def consec_filter(events, min_consec, max_consec):
    # This function filters out events with a minimum and maximum number of consecutive events
    logging.info('Starts consecutive events filter ...')
    # Start the timer
    start_time = time.time()

    # sorting the events by pixels and create new events array including a consecutive events weight
    # QUESTION: Is this faster than via pd dataframe?
    events = np.sort(events, order=['x', 'y', 't'])
    consec_events = np.zeros(events.size, dtype={'names':['x','y','p','t','w'], 'formats':['<u2','<u2','<i2','<i8','<i2']})
    consec_events[['x','y','p','t']] = events[['x','y','p','t']]
    consec_events['w'] = np.zeros(len(events['x']))

    # finding the consecutive events
    same_pixel = np.zeros(len(consec_events['x']), dtype=bool)
    same_pixel[1:] = (consec_events['x'][1:] == consec_events['x'][:-1]) & (consec_events['y'][1:] == consec_events['y'][:-1])
    same_polarity = np.zeros(len(consec_events['x']), dtype=bool)
    same_polarity[1:] = consec_events['p'][1:] == consec_events['p'][:-1]

    # assign weights to consecutive events
    for i in range(len(consec_events['w'])):
        if same_pixel[i] and same_polarity[i]:
            consec_events['w'][i-1] = consec_events['w'][i-1] + 1*(not bool(consec_events['w'][i-1]))
            consec_events['w'][i] = consec_events['w'][i-1] + 1

    # filtering out events with a minimum and maximum number of consecutive events
    consec_events = consec_events[(consec_events['w']>=min_consec) & (consec_events['w']<=max_consec)]

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    logging.info(f"Consecutive events filter ran for {elapsed_time} seconds.")
    logging.info('Filtering done.')

    return consec_events

def make_kdtree(events, temporal_duration, spatial_radius_outer):
    # This function creates a KDTree out of an events list

    # converting the temporal dimension to be in the same units
    ratio_micros_to_pixel = temporal_duration / spatial_radius_outer
    # Is the type conversion working?
    t = np.copy(events['t']).astype(float)
    t /= ratio_micros_to_pixel

    # Is this line similar to: points = np.c_[pos['x'].ravel(), pos['y'].ravel(), pos['t'].ravel()]?
    points = np.column_stack((events['x'], events['y'], t))
    
    # Making the tree
    tree = spatial.cKDTree(points)
    
    return points, tree

def count_neighbours(points, tree, spatial_radius_inner, spatial_radius_outer, temporal_duration):
    # This function counts the number of neighbouring events in a spatiotemporal ROI
    logging.info('Starts count neighbours ...')

    # Start the timer
    start_time = time.time()

    # creating kd tree from events
    points, tree = make_kdtree(points, temporal_duration, spatial_radius_outer)

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    logging.info(f"Count neighbours ran for {elapsed_time} seconds.")

    # Count the number of neighbours

def neighbours_filter(events, N_neighbours, spatial_radius_inner, spatial_radius_outer, temporal_duration, nr_ev_per_batch, polarity):
    # This function filters out events with a less neighbouring events than N_neighbours in a spatiotemporal ROI
    logging.info('Starts neighbours filter ...')

    # Start the timer
    start_time = time.time()

    polarity_events = events[events['p']==polarity]

    # creating kd tree from events
    points, tree = make_kdtree(polarity_events, temporal_duration, spatial_radius_outer)

    # calculating the number of batches
    nr_batches = int(np.ceil(len(polarity_events)/nr_ev_per_batch))

    # counting the number of neighbours for each pixel
    for batch in range(nr_batches):
        batch_start = batch * nr_ev_per_batch + 1 * (bool(batch))
        batch_end = min(batch_start + nr_ev_per_batch, len(polarity_events))
        batch_points = points[batch_start : batch_end + 1]
        batch_points, tree = make_kdtree(batch_points, temporal_duration, spatial_radius_outer)


    # Additional parameters--> Question: Should these be added to GUI? --> YES!
    spatial_radius_inner = 1
    spatial_radius_outer = 7
    temporal_duration = 50e3 # in µs
    nr_batches = 10
    polarity = 1

    # creating kd tree from events
    logging.info('Making tree ...')




    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    logging.info(f"Consecutive events filter ran for {elapsed_time} seconds.")
    logging.info('Filtering done.')

    return events

def clustering(events, DBSCAN_settings):
    # This function performs DBSCAN clustering on the events
    candidates = events
    return candidates

def perfect_ROI(clusters, settings):
    # This function returns a candidates dictionary with the perfect ROI
    candidates = {}
    candidates[0] = {}
    candidates[0]['events'] = pd.DataFrame(clusters)
    candidates[0]['cluster_size'] = [np.max(clusters['y'])-np.min(clusters['y']), np.max(clusters['x'])-np.min(clusters['x']), np.max(clusters['t'])-np.min(clusters['t'])]
    candidates[0]['N_events'] = len(clusters)
    return candidates


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def Finding(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    if "okwarg_1" in provided_optional_args:
        okwarg1 = float(kwargs["okwarg_1"])
    else:
        #Default okwarg1 value
        okwarg1 = 1
    # Start the timer
    start_time = time.time()
    
    consec_filter(npy_array, 1, 30)
    

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
