import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time, logging
from scipy import spatial
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "DBSCAN_onlyHighDensity": {
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
            "help_string": "DBSCAN, only return events that are considered high-density."
        },
        "DBSCAN_allEvents": {
            "required_kwargs": [
                {"name": "min_cluster_size", "description": "Required number of neighbouring events in spatiotemporal voxel","default":17,"type":int},
                {"name": "distance_radius_lookup", "description": "Outer radius (in px) to count the neighbours in.","default":7,"type":int},
                {"name": "density_multiplier", "description": "Distance multiplier","default":1.5,"type":float},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":6,"type":int},
                {"name": "padding_xy", "description": "Result padding in x,y pixels.","default":2,"type":int},
            ],
            "optional_kwargs": [
               {"name": "min_consec", "description": "Minimum number of consecutive events","default":1,"type":int},
               {"name": "max_consec", "description": "Maximum number of consecutive events, discards hot pixels","default":30,"type":int},
            ],
            "help_string": "DBSCAN on high-density, but returns all events in the bounding box specified by DBSCAN."
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def determineWeights(events):
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
    
    return weights,df_events


def consec_filter(events, min_consec, max_consec,weights = None,df_events=None):
    # This function filters out events with a minimum and maximum number of consecutive events
    if weights is None:
        weights,df_events = determineWeights(events)
    # filtering out events with a minimum and maximum number of consecutive events
    df_events = df_events[(df_events['w']>=min_consec) & (df_events['w']<=max_consec)]

    # convert df back to structured numpy array
    consec_events = df_events.to_records(index=False)

    return consec_events

def hotPixel_filter(events, max_consec, weights=None,df_events=None):
    # This function filters out events with a maximum number of consecutive events
    if weights is None:
        weights,df_events = determineWeights(events)
    
    # filtering out events with a minimum and maximum number of consecutive events
    df_events = df_events[(df_events['w']<=max_consec)]

    # convert df back to structured numpy array
    filtered_events = df_events.to_records(index=False)

    return filtered_events
    

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

def get_cluster_bounding_boxes(events, cluster_labels,padding_xy=0,padding_t=0):
    from scipy.spatial import ConvexHull
    bounding_boxes = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id != -1:  # Ignore noise points
            cluster_points = events[cluster_labels == cluster_id]
            #Get the min max values based on a convex hull
            try:
                hull = ConvexHull(np.column_stack((cluster_points['x'], cluster_points['y'], cluster_points['t'])))
                #And obtain the bounding box
                bounding_boxes[cluster_id] = (hull.min_bound[0]-padding_xy, hull.max_bound[0]+padding_xy, hull.min_bound[1]-padding_xy, hull.max_bound[1]+padding_xy, hull.min_bound[2]-padding_t, hull.max_bound[2]+padding_t)
            except:
                #if it's a 1-px-sized cluster, we can't get a convex hull, so we simply do this:
                bounding_boxes[cluster_id] = (min(cluster_points['x'])-padding_xy, max(cluster_points['x'])+padding_xy, min(cluster_points['y'])-padding_xy, max(cluster_points['y'])+padding_xy, min(cluster_points['t'])-padding_t, max(cluster_points['t'])+padding_t)
            
    #Loop over the bounding boxes, and if it's bigger than some params, remove it:
    xymaxsize = 20
    tmaxsize = np.inf
    listToRem = []
    for bboxid in bounding_boxes:
        if (bounding_boxes[bboxid][1]-bounding_boxes[bboxid][0]) > xymaxsize or (bounding_boxes[bboxid][3]-bounding_boxes[bboxid][2]) > xymaxsize or (bounding_boxes[bboxid][5]-bounding_boxes[bboxid][4]) > np.inf:
            listToRem.append(bboxid)
    for index in reversed(listToRem):
        logging.info('Removed bboxid: '+str(index)+' because it was too big')
        bounding_boxes.pop(index)
    
    return bounding_boxes

def get_events_in_bbox(npyarr,bboxes,ms_to_px,multiThread=True):
    #Get empty candidate dictionary
    candidates = {}
    if multiThread == False:        
        #Loop over all bboxes:
        for bboxid in range(len(bboxes)):
            bbox = bboxes[bboxid]
            filtered_array = npyarr[(npyarr['x'] >= bbox[0]) & (npyarr['x'] <= bbox[1]) & (npyarr['y'] >= bbox[2]) & (npyarr['y'] <= bbox[3]) & (npyarr['t'] >= bbox[4]*1000*ms_to_px) & (npyarr['t'] <= bbox[5]*1000*ms_to_px)]
            
            #Change filtered_array to a pd dataframe:
            filtered_df = pd.DataFrame(filtered_array)
            
            candidates[bboxid] = {}
            candidates[bboxid]['events'] = filtered_df
            candidates[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
            candidates[bboxid]['N_events'] = len(filtered_array)
    elif multiThread == True:
        num_cores = multiprocessing.cpu_count()
        logging.info("Bounding box finding split on "+str(num_cores)+" cores.")
        executor = ThreadPoolExecutor(max_workers=num_cores)  # Set the number of threads as desired

        # Create a list to store the results of the findbbox function
        results = []

        # Iterate over the bounding boxes
        for bbox in bboxes.values():
            # Submit each findbbox call to the ThreadPoolExecutor
            future = executor.submit(findbbox, npyarr, min_x=bbox[0], max_x=bbox[1], min_y=bbox[2], max_y=bbox[3], min_t=bbox[4]*1000*ms_to_px, max_t=bbox[5]*1000*ms_to_px)
            results.append(future)

        # Wait for all the submitted tasks to complete
        executor.shutdown()

        # Retrieve the results from the futures
        for future, bboxid in zip(results, bboxes.keys()):
            filtered_array = npyarr[future.result()]
            filtered_df2 = pd.DataFrame(filtered_array)
            candidates[bboxid] = {}
            candidates[bboxid]['events'] = filtered_df2
            candidates[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
            candidates[bboxid]['N_events'] = len(filtered_array)
    
    return candidates


def findbbox(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_t=-np.inf, max_t=np.inf):
    """ Adapted from:
https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy

    """
    bound_x = np.logical_and(points['x'] >= min_x, points['x'] <= max_x)
    bound_y = np.logical_and(points['y'] >= min_y, points['y'] <= max_y)
    bound_t = np.logical_and(points['t'] >= min_t, points['t'] <= max_t)
    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_t)
    return bb_filter

def process_bbox(args):
    bbox, npyarr, ms_to_px = args
    filtered_array = npyarr[(npyarr['x'] >= bbox[0]) & (npyarr['x'] <= bbox[1]) & (npyarr['y'] >= bbox[2]) & (npyarr['y'] <= bbox[3]) & (npyarr['t'] >= bbox[4]*1000*ms_to_px) & (npyarr['t'] <= bbox[5]*1000*ms_to_px)]
    filtered_df = pd.DataFrame(filtered_array)
    cluster_size = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
    N_events = len(filtered_array)
    return filtered_df, cluster_size, N_events

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def DBSCAN_allEvents(npy_array,settings,**kwargs):
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
    weights,df_events = determineWeights(npy_array)
    consec_events = consec_filter(npy_array, min_consec_ev, max_consec_ev,weights=weights,df_events=df_events)
    logging.info('Consec filtering done')
    polarities = consec_events['p']
    consec_event_tree, nparrfordens = make_kdtree(consec_events,temporal_duration_ms=float(kwargs['ratio_ms_to_px']),nleaves=64)
    logging.info('KDtree made done')
    high_density_events, polarities = filter_density(consec_event_tree, nparrfordens, polarities, distance_lookup = float(kwargs['distance_radius_lookup']), densityMultiplier = float(kwargs['density_multiplier']))
    logging.info('high density events obtained done')
    # Minimum number of points within a cluster
    clustersHD, cluster_labels = clustering(high_density_events, polarities, eps = float(kwargs['DBSCAN_eps']), min_points_per_cluster = int(kwargs['min_cluster_size']))
    logging.info('DBSCAN done')
    bboxes = get_cluster_bounding_boxes(clustersHD, cluster_labels,padding_xy = int(kwargs['padding_xy']),padding_t = int(kwargs['padding_xy']))
    logging.info('Getting bounding boxes done')
    hotpixel_filtered_events = hotPixel_filter(npy_array,max_consec_ev,weights=weights,df_events=df_events)
    candidates = get_events_in_bbox(hotpixel_filtered_events,bboxes,float(kwargs['ratio_ms_to_px']))
    logging.info('Candidates obtained')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    performance_metadata = f"DBSCAN Finding ran for {elapsed_time} seconds."
    logging.info('DBSCAN finding done')

    return candidates, performance_metadata

def DBSCAN_onlyHighDensity(npy_array,settings,**kwargs):
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
