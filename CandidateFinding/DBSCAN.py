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
                {"name": "min_cluster_size", "description": "Required number of neighbouring events in spatiotemporal voxel","default":17,"type":int},
                {"name": "distance_radius_lookup", "description": "Outer radius (in px) to count the neighbours in.","default":7,"type":int},
                {"name": "density_multiplier", "description": "Distance multiplier","default":1.5,"type":float},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":6,"type":int},
            ],
            "optional_kwargs": [
               {"name": "min_consec", "description": "Minimum number of consecutive events","default":1,"type":int},
               {"name": "max_consec", "description": "Maximum number of consecutive events, discards hot pixels","default":30,"type":int},
            ],
            "help_string": "Returns a test dictionary and metadata string, to check if the program is working properly."
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def consec_filter(events, min_consec, max_consec):
    # This function filters out events with a minimum and maximum number of consecutive events

    # df creation and sorting (sufficient and faster for sort by only x,y)
    df_events = pd.DataFrame(events)
    df_events = df_events.sort_values(by=['x', 'y'])

    # finding the consecutive events
    same_pixel = np.zeros(len(df_events['x']), dtype=bool)
    same_pixel[1:] = (df_events['x'][1:].reset_index(drop=True) == df_events['x'][:-1].reset_index(drop=True)) & (df_events['y'][1:].reset_index(drop=True) == df_events['y'][:-1].reset_index(drop=True))
    same_polarity = np.zeros(len(df_events['x']), dtype=bool)
    same_polarity[1:] = df_events['p'][1:].reset_index(drop=True) == df_events['p'][:-1].reset_index(drop=True)

    # assign weights to consecutive events (loop only over same pixel and polarity)
    same_pixel_polarity = same_pixel & same_polarity
    weights = np.zeros(len(events['x']))
    indices = np.where(same_pixel_polarity == True)[0]
    for i in indices:
        weights[i-1] = weights[i-1] + 1*(not bool(weights[i-1]))
        weights[i] = weights[i-1] + 1
    df_events['w'] = weights

    # filtering out events with a minimum and maximum number of consecutive events
    df_events = df_events[(df_events['w']>=min_consec) & (df_events['w']<=max_consec)]

    # convert df back to structured numpy array
    consec_events = df_events.to_records(index=False)

    return consec_events

def make_kdtree(events, temporal_duration_ms=35,nleaves = 16):
    # This function creates a KDTree out of an events list

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
    
    # Create KDtree
    tree = spatial.cKDTree(np.array(nparrfordens), leafsize=nleaves)
    
    return tree, nparrfordens

def filter_density(eventTree,nparrfordens,polarities,distance_lookup = 4, densityMultiplier = 1.5):
    
    maxK = 100
    neighbors3,_ = eventTree.query(nparrfordens,k=maxK,distance_upper_bound=distance_lookup,eps=distance_lookup/2,workers=-1)
    frequency = np.argmax(np.isinf(neighbors3), axis=1)
    frequency[frequency==0] = maxK
    
    freq_threshold = np.mean(frequency)*float(densityMultiplier)
    freq_points_within_range = frequency>=freq_threshold
    polarities = polarities[freq_points_within_range]
    densest_points_within_range_full = nparrfordens[freq_points_within_range,:]
    
    return densest_points_within_range_full, polarities

def clustering(events, polarities, eps=2, min_points_per_cluster=10):
    if len(events) == 0:
        logging.warning('No clusters found! DBSCAN not run')
        return [],[]
    # This function performs DBSCAN clustering on the events
    # Use DBSCAN clustering
    dbscan = DBSCAN(eps=eps, n_jobs=-1, min_samples=min_points_per_cluster)
    cluster_labels = dbscan.fit_predict(events)
    
    # Print how many clusters are found
    logging.info("Number of clusters found:"+ str(max(cluster_labels)))

    # generate the candidates in from of a dictionary
    headers = ['x', 'y', 't']
    densest_points_within_range_full_pd = pd.DataFrame(events, columns=headers)
    densest_points_within_range_full_pd['p'] = polarities
    new_order = ['x', 'y', 'p', 't']
    densest_points_within_range_full_pd = densest_points_within_range_full_pd.reindex(columns=new_order)

    # set correct types:
    densest_points_within_range_full_pd = densest_points_within_range_full_pd.astype({'x': 'int64', 'y': 'int64', 'p': 'int64', 't': 'float64'})
    
    return densest_points_within_range_full_pd, cluster_labels

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def Finding(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    if "min_consec" in provided_optional_args:
        min_consec_ev = float(kwargs["min_consec"])
    else:
        # Default value for min number of consecutive events
        min_consec_ev = 1
    if "max_consec" in provided_optional_args:
        max_consec_ev = float(kwargs["max_consec"])
    else:
        # Default value for max number of consecutive events
        max_consec_ev = 30

    # Start the timer
    start_time = time.time()
    
    logging.info('Starting')
    consec_events = consec_filter(npy_array, min_consec_ev, max_consec_ev)
    logging.info('Consec filtering done')
    polarities = consec_events['p']
    consec_event_tree, nparrfordens = make_kdtree(consec_events,temporal_duration_ms=float(kwargs['ratio_ms_to_px']),nleaves=64)
    logging.info('KDtree made done')
    high_density_events, polarities = filter_density(consec_event_tree, nparrfordens, polarities, distance_lookup = float(kwargs['distance_radius_lookup']), densityMultiplier = float(kwargs['density_multiplier']))
    logging.info('high density events obtained done')
    # Minimum number of points within a cluster
    clusters, cluster_labels = clustering(high_density_events, polarities, eps = float(kwargs['DBSCAN_eps']), min_points_per_cluster = int(kwargs['min_cluster_size']))
    logging.info('DBSCAN done')
    #Re-correct for z to time:
    clusters.loc[:,'t'] = clusters['t']*(float(kwargs['ratio_ms_to_px'])*1000)
    logging.info('re-corrected for z')
    
    #old version kept here, since I haven't 100% stress-tested new method, but seems to be fine
    starttime = time.time()
    candidates = {}
    for cl in np.unique(cluster_labels):
        if cl > -1:
            clusterEvents = clusters[cluster_labels == cl]
            candidates[cl] = {}
            candidates[cl]['events'] = clusterEvents
            candidates[cl]['cluster_size'] = [np.max(clusterEvents['y'])-np.min(clusterEvents['y']), np.max(clusterEvents['x'])-np.min(clusterEvents['x']), np.max(clusterEvents['t'])-np.min(clusterEvents['t'])]
            candidates[cl]['N_events'] = len(clusterEvents)
    endtime = time.time()
    # Print the elapsed time:
    elapsed_time = endtime - starttime
    print(f"Candidateforming1: {elapsed_time} seconds.")


    #New method that is faster but not working correctly (but almost)
    # starttime = time.time()
    # #New method - via external ordering and lookup (Accumarray from MATLAB)
    # #get some sorting info
    # order_array = np.argsort(cluster_labels)
    # #Sort cluster labels on this
    # cluster_labelsSort = cluster_labels[order_array]
    # #sort the clusters pd on this - very convoluted
    # clustersSorted = clusters.assign(P=order_array)
    # clustersSorted.sort_values('P')
    # clustersSorted.drop('P', axis=1, inplace=True)
    # #Get the bincounts - know that clusters start at -1, so add 1 to srart at 0
    # accarr = np.bincount(cluster_labelsSort+1)

    # candidates = {}
    # unique_labels = np.unique(cluster_labelsSort)
    # for cl in unique_labels:
    #     if cl > -1:
    #         # clusterEvents = clustersSorted[cluster_labelsSort == cl]
    #         clusterEvents = clustersSorted[sum(accarr[0:cl+1]):sum(accarr[0:cl+1])+accarr[cl+1]]
    #         cluster_size = np.array([
    #             np.max(clusterEvents['y']) - np.min(clusterEvents['y']),
    #             np.max(clusterEvents['x']) - np.min(clusterEvents['x']),
    #             np.max(clusterEvents['t']) - np.min(clusterEvents['t'])
    #         ])
    #         candidates[cl] = {
    #             'events': clusterEvents,
    #             'cluster_size': cluster_size,
    #             'N_events': len(clusterEvents)
    #         }
    # endtime = time.time()
    # # Print the elapsed time:
    # elapsed_time = endtime - starttime
    # logging.info(f"Candidateforming: {elapsed_time} seconds.")
    # Stop the timer
    end_time = time.time()
 
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    performance_metadata = f"DBSCAN Finding ran for {elapsed_time} seconds."
    logging.info('DBSCAN finding done')

    return candidates, performance_metadata
