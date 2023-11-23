import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time, logging
from scipy import spatial
from sklearn.cluster import DBSCAN

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Finding": {
            "required_kwargs": [
                {"name": "min_cluster_size", "description": "Required number of neighbouring events in spatiotemporal voxel","default":17},
                {"name": "distance_radius_lookup", "description": "Outer radius (in px) to count the neighbours in.","default":7},
                {"name": "density_multiplier", "description": "Distance multiplier","default":1.5},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":6},
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
    # logging.info('Starts consecutive events filter ...')
    # # Start the timer
    # start_time = time.time()

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
    # end_time = time.time()

    # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    # logging.info(f"Consecutive events filter ran for {elapsed_time} seconds.")
    # logging.info('Filtering done.')

    return consec_events

def make_kdtree(events, temporal_duration_ms=35,nleaves = 16):
    # This function creates a KDTree out of an events list

    N=100
    col1 = np.empty((N,), dtype=int)
    col2 = np.empty((N,), dtype=float)
    col3 = np.empty((N,), dtype=str)

    # Stack the arrays horizontally using np.column_stack
    arr = np.column_stack((col1, col2, col3))

    # converting the temporal dimension to be in the same units
    ratio_microsec_to_pixel = 1/(temporal_duration_ms*1000)
    colx = np.empty((len(events),), dtype=int)
    colx = events['x'].astype(int)
    coly = np.empty((len(events),), dtype=int)
    coly = events['y'].astype(int)
    colt = np.empty((len(events),), dtype=float)
    colt = events['t'].astype(float)*ratio_microsec_to_pixel
    nparrfordens = np.column_stack((colx, coly, colt))
    # nparrfordens = np.column_stack(((events['x'].astype(int), events['y'].astype(int), events['t'].astype(float)*ratio_microsec_to_pixel),dtype=[int,int,float]))
    
    #Create KDtree
    tree = spatial.cKDTree(np.array(nparrfordens), leafsize=nleaves)
    
    return tree, nparrfordens

def filter_density(eventTree,nparrfordens,distance_lookup = 4, densityMultiplier = 1.5):
    
    neighbors3,_ = eventTree.query(nparrfordens,k=100,distance_upper_bound=distance_lookup,eps=distance_lookup/2,workers=-1)
    frequency = np.argmax(np.isinf(neighbors3), axis=1)
    
    freq_threshold = np.mean(frequency)*float(densityMultiplier)
    print(freq_threshold)
    freq_points_within_range = frequency>=freq_threshold
    densest_points_within_range_full = nparrfordens[freq_points_within_range,:]
    
    return densest_points_within_range_full

def clustering(events, eps=2, min_points_per_cluster=10):
    # This function performs DBSCAN clustering on the events
    # Use DBSCAN clustering
    dbscan = DBSCAN(eps=eps, n_jobs=-1, min_samples=min_points_per_cluster)
    cluster_labels = dbscan.fit_predict(events)
    
    #Print how many clusters are found
    logging.info("Number of clusters found:"+ str(max(cluster_labels)))

    # generate the candidates in from of a dictionary
    headers = ['x', 'y', 't' ]
    densest_points_within_range_full_pd = pd.DataFrame(events, columns=headers)
    #set correct types:
    densest_points_within_range_full_pd = densest_points_within_range_full_pd.astype({'x': 'int64', 'y': 'int64', 't': 'float64'})
    
    return densest_points_within_range_full_pd, cluster_labels

def perfect_ROI(clusters, settings):
    # This function returns a candidates dictionary with the perfect ROIs
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
    if "min_consec" in provided_optional_args:
        min_consec_ev = float(kwargs["min_consec"])
    else:
        #Default okwarg1 value
        min_consec_ev = 1
    if "max_consec" in provided_optional_args:
        max_consec_ev = float(kwargs["max_consec"])
    else:
        #Default okwarg1 value
        max_consec_ev = 1
    # Start the timer
    start_time = time.time()
    
    logging.info('Starting')
    consec_events = consec_filter(npy_array, min_consec_ev, max_consec_ev)
    logging.info('Consec filtering done')
    consec_event_tree, nparrfordens = make_kdtree(consec_events,temporal_duration_ms=float(kwargs['ratio_ms_to_px']),nleaves=64)
    logging.info('KDtree made done')
    high_density_events = filter_density(consec_event_tree, nparrfordens, distance_lookup = float(kwargs['distance_radius_lookup']), densityMultiplier = float(kwargs['density_multiplier']))
    logging.info('high density events obtained done')
    # Minimum number of points within a cluster
    clusters, cluster_labels = clustering(high_density_events, eps = float(kwargs['DBSCAN_eps']), min_points_per_cluster = int(kwargs['min_cluster_size']))
    logging.info('DBSCAN done')
    #Re-correct for z to time:
    clusters.loc[:,'t'] = clusters['t']*(float(kwargs['ratio_ms_to_px'])*1000)
    logging.info('re-corrected for z')
    print('done')
    
    candidates = {}
    for cl in np.unique(cluster_labels):
        if cl > -1:
            clusterEvents = clusters[cluster_labels == cl]
            candidates[cl] = {}
            candidates[cl]['events'] = clusterEvents
            candidates[cl]['cluster_size'] = [np.max(clusterEvents['y'])-np.min(clusterEvents['y']), np.max(clusterEvents['x'])-np.min(clusterEvents['x']), np.max(clusterEvents['t'])-np.min(clusterEvents['t'])]
            candidates[cl]['N_events'] = len(clusterEvents)

    # Stop the timer
    end_time = time.time()
 
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    performance_metadata = f"Dummy function ran for {elapsed_time} seconds."
    print('Function one ran!')

    return candidates, performance_metadata
