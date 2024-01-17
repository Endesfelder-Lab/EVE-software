import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time, logging
from scipy import spatial
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull
import multiprocessing
from functools import partial
import open3d as o3d
from joblib import Parallel, delayed
import multiprocessing
import classix

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "CLASSIX_test": {
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
    # start_time = time.time()
    # bounding_boxes = {}
    # for cluster_id in np.unique(cluster_labels):
    #     if cluster_id != -1:  # Ignore noise points
    #         cluster_points = events[cluster_labels == cluster_id]
    #         #Get the min max values based on a convex hull
    #         try:
    #             hull = ConvexHull(np.column_stack((cluster_points['x'], cluster_points['y'], cluster_points['t'])))
    #             #And obtain the bounding box
    #             bounding_boxes[cluster_id] = (hull.min_bound[0]-padding_xy, hull.max_bound[0]+padding_xy, hull.min_bound[1]-padding_xy, hull.max_bound[1]+padding_xy, hull.min_bound[2]-padding_t, hull.max_bound[2]+padding_t)
    #         except:
    #             #if it's a 1-px-sized cluster, we can't get a convex hull, so we simply do this:
    #             bounding_boxes[cluster_id] = (min(cluster_points['x'])-padding_xy, max(cluster_points['x'])+padding_xy, min(cluster_points['y'])-padding_xy, max(cluster_points['y'])+padding_xy, min(cluster_points['t'])-padding_t, max(cluster_points['t'])+padding_t)
    
    # end_time = time.time()
    # logging.info('Time to get bounding boxes: '+str(end_time-start_time))
    
    
    start_time = time.time()
    bounding_boxes = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id != -1:  # Ignore noise points
            bounding_boxes[cluster_id] = compute_bounding_boxC(cluster_id,events=events,cluster_labels=cluster_labels,padding_xy=padding_xy,padding_t=padding_t)
    
    end_time = time.time()
    logging.info('Time to get bounding boxesC: '+str(end_time-start_time))
    
    
    # start_time = time.time()
    # bounding_boxes = {}
    # cluster_ids = np.unique(cluster_labels)
    # valid_cluster_ids = cluster_ids[cluster_ids != -1]  # Filter out noise 

    # # Define a partial function with fixed arguments
    # partial_compute_bounding_box = partial(compute_bounding_box, events=events, cluster_labels=cluster_labels, padding_xy=padding_xy, padding_t=padding_t)

    # # Create a pool of worker processes
    # pool = multiprocessing.Pool()

    # # Map the function to the cluster IDs in parallel
    # results = pool.map(partial_compute_bounding_box, valid_cluster_ids)

    # # Collect the results into the bounding_boxes dictionary
    # bounding_boxes = {cluster_id: bounding_box for cluster_id, bounding_box in results}
    
    # end_time = time.time()
    # logging.info('Time to get bounding boxes new: '+str(end_time-start_time))
    
    
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

def compute_bounding_boxC(cluster_id,events=None,cluster_labels=None,padding_xy=None,padding_t=None):
    cluster_points = events[cluster_labels == cluster_id]
    x_coordinates = cluster_points['x'].tolist()
    y_coordinates = cluster_points['y'].tolist()
    t_coordinates = cluster_points['t'].tolist()
    return [min(x_coordinates)-padding_xy, max(x_coordinates)+padding_xy, min(y_coordinates)-padding_xy, max(y_coordinates)+padding_xy, min(t_coordinates)-padding_t, max(t_coordinates)+padding_t]

# Define a function to compute the bounding box for a given cluster ID
def compute_bounding_box(cluster_id,events=None,cluster_labels=None,padding_xy=None,padding_t=None):
    cluster_points = events[cluster_labels == cluster_id]
    try:
        hull = ConvexHull(np.column_stack((cluster_points['x'], cluster_points['y'], cluster_points['t'])))
        bounding_box = (hull.min_bound[0]-padding_xy, hull.max_bound[0]+padding_xy, hull.min_bound[1]-padding_xy, hull.max_bound[1]+padding_xy, hull.min_bound[2]-padding_t, hull.max_bound[2]+padding_t)
    except:
        bounding_box = (np.min(cluster_points['x'])-padding_xy, np.max(cluster_points['x'])+padding_xy, np.min(cluster_points['y'])-padding_xy, np.max(cluster_points['y'])+padding_xy, np.min(cluster_points['t'])-padding_t, np.max(cluster_points['t'])+padding_t)
    return (cluster_id, bounding_box)


def kdtree_assisted_lookup(bbox_bunch,ms_to_px,kdtree1d,lookup_list,npyarr):
    min_x_arr = [bbox[0] for bbox in bbox_bunch.values()]
    max_x_arr = [bbox[1] for bbox in bbox_bunch.values()]
    min_y_arr = [bbox[2] for bbox in bbox_bunch.values()]
    max_y_arr = [bbox[3] for bbox in bbox_bunch.values()]
    min_t_arr = [bbox[4]*1000*ms_to_px for bbox in bbox_bunch.values()]
    max_t_arr = [bbox[5]*1000*ms_to_px for bbox in bbox_bunch.values()]
    
    result = {}
    result['x']={}
    lb_x = np.transpose(min_x_arr)
    ub_x = np.transpose(max_x_arr)
    mp_x = np.mean([lb_x,ub_x],axis=0)
    rad_x = np.subtract(ub_x,lb_x)/2
    iv = np.transpose([mp_x,mp_x])
    result['x'] = kdtree1d['x'].query_ball_point(iv, np.sqrt(2)*rad_x,workers=-1)
    
    result['y']={}
    lb_y = np.transpose(min_y_arr)
    ub_y = np.transpose(max_y_arr)
    mp_y = np.mean([lb_y,ub_y],axis=0)
    rad_y = np.subtract(ub_y,lb_y)/2
    iv = np.transpose([mp_y,mp_y])
    result['y'] = kdtree1d['y'].query_ball_point(iv, np.sqrt(2)*rad_y,workers=-1)
    
    #Get all results as candidates dataframes as wanted:
    filtered_df2={}
    for t in range(len(min_x_arr)):      
        #Filter only on x,y
        a1 = np.isin(lookup_list['x'][result['x'][t]],result['y'][t],assume_unique=True)
        filtered_array = npyarr[lookup_list['x'][result['x'][t]][a1]]
        
        #Filter on t
        bound_t = np.logical_and(filtered_array['t'] >= min_t_arr[t], filtered_array['t'] <= max_t_arr[t])
        
        #Fully filtered:
        filtered_array2 = filtered_array[bound_t]
                    
        filtered_df2[t] = pd.DataFrame(filtered_array2)
    
    return filtered_df2

def get_events_in_bbox(npyarr,bboxes,ms_to_px,multiThread=True):
    #Get empty candidate dictionary
    candidates = {}
    if multiThread == False:      
        start_time = time.time()
        #Loop over all bboxes:
        for bboxid, _ in bboxes.items():
            bbox = bboxes[bboxid]
            conditions = [
                npyarr['x'] >= bbox[0],
                npyarr['x'] <= bbox[1],
                npyarr['y'] >= bbox[2],
                npyarr['y'] <= bbox[3],
                npyarr['t'] >= bbox[4]*1000*ms_to_px,
                npyarr['t'] <= bbox[5]*1000*ms_to_px
            ]
            filtered_array = npyarr[np.logical_and.reduce(conditions)]
            #Change filtered_array to a pd dataframe:
            filtered_df = pd.DataFrame(filtered_array)
            
            candidates[bboxid] = {}
            candidates[bboxid]['events'] = filtered_df
            candidates[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
            candidates[bboxid]['N_events'] = len(filtered_array)
        end_time = time.time()
        logging.info('Time to get bounding boxes: '+str(end_time-start_time))
        
        
        candidates2 = {}
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(npyarr['x'],npyarr['y'],npyarr['t']))
        
        for bboxid, _ in bboxes.items():
            bbox = bboxes[bboxid]
            # Create the axis-aligned bounding box with the specified dimensions
            aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(bbox[0], bbox[2], bbox[4]*1000*ms_to_px), max_bound=(bbox[1], bbox[3], bbox[5]*1000*ms_to_px))

            # Get the indices of points within the bounding box
            indices = aabb.get_point_indices_within_bounding_box(point_cloud.points)
            filtered_array = npyarr[indices]
            #Change filtered_array to a pd dataframe:
            filtered_df = pd.DataFrame(filtered_array)
            
            candidates2[bboxid] = {}
            candidates2[bboxid]['events'] = filtered_df
            candidates2[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
            candidates2[bboxid]['N_events'] = len(filtered_array)
        end_time = time.time()
        logging.info('Time to get bounding boxes o3d: '+str(end_time-start_time))

        
    elif multiThread == True:
        time_start = time.time()
        
        candidates = {}
        #Create a 1d kdtree of the x-values of npyarr:
        lookup_list = {}
        kdtree1d = {}

        
        bunch_size = 50
        num_splits = max(1,int(np.ceil(len(bboxes)/bunch_size)))
                
        #Run kdtrees on x and y:
        datax = np.zeros((len(npyarr),2))
        datax[:, 0] = npyarr['x']
        datax[:, 1] = npyarr['x']
        lookup_list['x'] = np.argsort(datax[:, 0])
        kdtree1d['x'] = spatial.cKDTree(datax)
        datay = np.zeros((len(npyarr),2))
        datay[:, 0] = npyarr['y']
        datay[:, 1] = npyarr['y']
        lookup_list['y'] = np.argsort(datay[:, 0])
        kdtree1d['y'] = spatial.cKDTree(datay)
        
        #Split bboxes into bunches:
        bbox_bunches = split_dict(bboxes, num_splits)
        num_cores = multiprocessing.cpu_count()
        
        RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(kdtree_assisted_lookup)(bbox_bunches[r],ms_to_px,kdtree1d,lookup_list,npyarr) for r in range(len(bbox_bunches)))
        result = [res for res in RES]
        
        counter = 0
        for r in range(len(result)):
            for b in range(len(result[r])):
                filtered_array = result[r][b]
                filtered_df2 = pd.DataFrame(filtered_array)
                
                indexv = counter
                candidates[indexv] = {}
                candidates[indexv]['events'] = filtered_df2
                candidates[indexv]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
                candidates[indexv]['N_events'] = len(filtered_array)
                counter+=1
        
        time_end = time.time()
        print(time_end-time_start)
                
        
             
        
        # start_time = time.time()
        # num_cores = multiprocessing.cpu_count()
        # logging.info("Bounding box finding split on "+str(num_cores)+" cores.")
        
        # #Sort the bounding boxes by start-time:
        # sorted_bboxes = sorted(bboxes.values(), key=lambda bbox: bbox[4])
        
        # #Split into num_cores sections:
        # bboxes_split = np.array_split(sorted_bboxes, num_cores)
        
        # #Split the npyarr based on the min, max time of bboxes_split:
        # npyarr_split={}
        # for i in range(0,num_cores):
        #     selectionArea = (npyarr['t'] >= np.min(bboxes_split[i][:,4])*1000*ms_to_px) & (npyarr['t'] <= np.max(bboxes_split[i][:,5])*1000*ms_to_px)
        #     npyarr_split[i] = npyarr[selectionArea]

        # RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(findbboxeso3d)(npyarr_split[i],bboxes_split[i],ms_to_px) for i in range(num_cores))
        
        # #res in RES contains an array of arrays, where e.g. res[0] is the indeces of the events that belong to bbox 0, etc
        # result = [res for res in RES]
        
        # #Get all results as candidates dataframes as wanted:
        # candidates = {}
        # counter = 0
        # for r in range(len(result)):
        #     for b in range(len(result[r])):
        #         filtered_array = npyarr_split[r][result[r][b]]
        #         filtered_df2 = pd.DataFrame(filtered_array)
        #         candidates[counter] = {}
        #         candidates[counter]['events'] = filtered_df2
        #         candidates[counter]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
        #         candidates[counter]['N_events'] = len(filtered_array)
        #         counter +=1
                
        # end_time = time.time()
        # logging.info('Time to get bounding boxes o3d: '+str(end_time-start_time))
        
    
        print('Done')
    return candidates


def split_dict(dictionary, num_splits):
    keys = list(dictionary.keys())
    split_keys = np.array_split(keys, num_splits)
    split_dicts = [{key: dictionary[key] for key in split_keys[i]} for i in range(num_splits)]
    return split_dicts

# Define a custom distance function
def distance_1dkdtree(point,lb,ub):
    if point > lb and point < ub:
        return 1
    else:
        return 0 
    
def findbboxeso3d(points,bboxes,ms_to_px):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(zip(points['x'],points['y'],points['t']))
    
    res = {}
    #Split the bboxes to separate findbbox function calls:
    for bbox_id in range(len(bboxes)):
        bbox = bboxes[bbox_id]
        res[bbox_id] = findbboxo3d(point_cloud, min_x=bbox[0], max_x=bbox[1], min_y=bbox[2], max_y=bbox[3], min_t=bbox[4]*1000*ms_to_px, max_t=bbox[5]*1000*ms_to_px)
    return res

def findbboxo3d(pointCloud, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_t=-np.inf, max_t=np.inf):
    # Create the axis-aligned bounding box with the specified dimensions
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_x, min_y, min_t), max_bound=(max_x, max_y, max_t))
    # Get the indices of points within the bounding box
    indices = aabb.get_point_indices_within_bounding_box(pointCloud.points)
    return indices

def findbboxes(points, bboxes,ms_to_px):
    res = {}
    #Split the bboxes to separate findbbox function calls:
    for bbox_id in range(len(bboxes)):
        bbox = bboxes[bbox_id]
        res[bbox_id] = findbbox(points, min_x=bbox[0], max_x=bbox[1], min_y=bbox[2], max_y=bbox[3], min_t=bbox[4]*1000*ms_to_px, max_t=bbox[5]*1000*ms_to_px)
    return res

def findbbox(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_t=-np.inf, max_t=np.inf):
    """ Adapted from:
https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy

    """
    bound_x = np.logical_and(points['x'] >= min_x, points['x'] <= max_x)
    bound_y = np.logical_and(points['y'] >= min_y, points['y'] <= max_y)
    bound_t = np.logical_and(points['t'] >= min_t, points['t'] <= max_t)
    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_t)
    #Transform this to a list of indeces where bb_filter == true:
    bb_filter = np.where(bb_filter == True)
    return bb_filter

def findbboxnew(npyarr, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_t=-np.inf, max_t=np.inf):
                
    bound_x = np.logical_and(npyarr['x'] >= min_x, npyarr['x'] <= max_x)
    bound_y = np.logical_and(npyarr['y'] >= min_y, npyarr['y'] <= max_y)
    bound_t = np.logical_and(npyarr['t'] >= min_t, npyarr['t'] <= max_t)
    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_t)
    filtered_df = pd.DataFrame(npyarr[bb_filter])
    return filtered_df



def process_bbox(args):
    bbox, npyarr, ms_to_px = args
    filtered_array = npyarr[(npyarr['x'] >= bbox[0]) & (npyarr['x'] <= bbox[1]) & (npyarr['y'] >= bbox[2]) & (npyarr['y'] <= bbox[3]) & (npyarr['t'] >= bbox[4]*1000*ms_to_px) & (npyarr['t'] <= bbox[5]*1000*ms_to_px)]
    filtered_df = pd.DataFrame(filtered_array)
    cluster_size = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
    N_events = len(filtered_array)
    return filtered_df, cluster_size, N_events


def o3d_getclusterbounding_boxes(events, cluster_labels,padding_xy = 0,padding_t = 0):
    x=3
    
    
    start_time = time.time()
    bounding_boxes = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id != -1:  # Ignore noise points
            bounding_boxes[cluster_id] = compute_bounding_boxC(cluster_id,events=events,cluster_labels=cluster_labels,padding_xy=padding_xy,padding_t=padding_t)
    
    end_time = time.time()
    logging.info('Time to get bounding boxesC: '+str(end_time-start_time))
    
    
    start_time = time.time()
    bounding_boxes2 = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id != -1:  # Ignore noise points
            cluster_points = events[cluster_labels == cluster_id]
            # Create a PointCloud object from the numpy array
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(zip(cluster_points['x'],cluster_points['y'],cluster_points['t']))
            
            # Compute the axis-aligned bounding box
            aabb = point_cloud.get_axis_aligned_bounding_box()
            bounding_boxes2[cluster_id] = (aabb.get_min_bound()[0]-padding_xy, aabb.get_max_bound()[0]+padding_xy, aabb.get_min_bound()[1]-padding_xy, aabb.get_max_bound()[1]+padding_xy, aabb.get_min_bound()[2]-padding_t, aabb.get_max_bound()[2]+padding_t)
    end_time = time.time()
    logging.info('Time to get bounding boxes o3d: '+str(end_time-start_time))


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

def CLASSIX_test(npy_array,settings,**kwargs):
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
    
    # data, labels = classix.loadData('Covid3MC')
    
    consec_events = consec_filter(npy_array, min_consec_ev, max_consec_ev)
    logging.info('Consec filtering done')
    
    cluster_dist = 10
    from sklearn.datasets import make_blobs
    # print('---------------')
    
    # X, y = make_blobs(n_samples=100, centers=1, n_features=2, random_state=1)
    # #Normalize x so center is 0,0:
    # X[:,0] = X[:,0] - np.mean(X[:,0])
    # X[:,1] = X[:,1] - np.mean(X[:,1])
    # X2, y = make_blobs(n_samples=100, centers=1, n_features=2, random_state=1)
    # #Normalize x so center is 0,0:
    # X2[:,0] = X2[:,0] - np.mean(X[:,0])+cluster_dist
    # X2[:,1] = X2[:,1] - np.mean(X[:,1])+(cluster_dist)
    # X = np.concatenate((X,X2),axis=0)
    
    
    
    
    consec_events = consec_filter(npy_array, min_consec_ev, max_consec_ev)
    logging.info('Consec filtering done')
    polarities = consec_events['p']
    consec_event_tree, nparrfordens = make_kdtree(consec_events,temporal_duration_ms=float(kwargs['ratio_ms_to_px']),nleaves=64)
    logging.info('KDtree made done')
    high_density_events, polarities = filter_density(consec_event_tree, nparrfordens, polarities, distance_lookup = float(kwargs['distance_radius_lookup']), densityMultiplier = float(kwargs['density_multiplier']))
    logging.info('high density events obtained done')
    
    #How about we run classix on npy_array ?
    px_to_ms = float(kwargs['ratio_ms_to_px'])
    # data_for_clx = np.column_stack((high_density_events[:,0], high_density_events[:,1], (high_density_events[:,2]/1000)/px_to_ms))
    
    data_for_clx = np.column_stack((consec_events['x'], consec_events['y'], (consec_events['t']/1000)/px_to_ms))
    
    # data_for_clx = np.column_stack((npy_array['x'], npy_array['y'], (npy_array['t']/1000)/px_to_ms))
    
    # data_for_clx = np.column_stack((consec_events['x'], consec_events['y']))
    
    #only get data between 150-200 and 150-200 in 0, 1st column:
    data_for_clx = data_for_clx[(data_for_clx[:,0] > 100) & (data_for_clx[:,0] < 200) & (data_for_clx[:,1] > 100) & (data_for_clx[:,1] < 200)]
    
    radius_lookup = 3#float(kwargs["distance_radius_lookup"]) #0.02 gets hot pixels
    minpts = 15#int(kwargs["min_cluster_size"])
    
    #get the scale factor:
    #NOTE: we are assuming pca-based classix
    from numpy.linalg import norm
    mu = data_for_clx.mean(axis=0)
    X = data_for_clx - mu # mean center
    rds = norm(data_for_clx, axis=1) # distance of each data point from 0
    scaleFactor = np.median(rds) # 50% of data points are within that radius
    scaleFactor = 1
    clx = classix.CLASSIX(sorting='pca', group_merging = 'density', radius=radius_lookup/scaleFactor, minPts=minpts,post_alloc=False,mergeScale=1.5)#, mergeTinyGroups = True, verbose=1,post_alloc=False,sorting='pca')
   
    clx = classix.CLASSIX(sorting=None, group_merging = 'density', radius=radius_lookup/scaleFactor, minPts=minpts,post_alloc=False,mergeScale=1.5)#, mergeTinyGroups = True, verbose=1,post_alloc=False,sorting='pca')
   
    clx.fit(data_for_clx)
    print(clx.labels_) # clustering labels 
    clx.explain()
    
    # cluster_id = 11
    # cluster_label = np.unique(clx.sp_info['Cluster'])[cluster_id]
    # indices = np.where(clx.sp_info['Cluster'] == cluster_label)

    # clx.explain(min(indices[0]),plot=True)
    
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #plot only clx.labels > -1:
    ax.scatter(data_for_clx[clx.labels_>-1,0], data_for_clx[clx.labels_>-1,1],  data_for_clx[clx.labels_>-1,2], c=clx.labels_[clx.labels_>-1],cmap='Set1')
    #ax.scatter(data_for_clx[:,0], data_for_clx[:,1],  data_for_clx[:,2], c=clx.labels_,cmap='Set1')
    #change cmap to be strongly changing colors:
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #plot only clx.labels > -1:
    ax.scatter(data_for_clx[clx.labels_==-1,0], data_for_clx[clx.labels_==-1,1],  data_for_clx[clx.labels_==-1,2], c='k',alpha=0.1)
    ax.scatter(data_for_clx[clx.labels_>-1,0], data_for_clx[clx.labels_>-1,1],  data_for_clx[clx.labels_>-1,2], c=clx.labels_[clx.labels_>-1],cmap='Set1')
    #ax.scatter(data_for_clx[:,0], data_for_clx[:,1],  data_for_clx[:,2], c=clx.labels_,cmap='Set1')
    #change cmap to be strongly changing colors:
    plt.show()
    
    #old version kept here, since I haven't 100% stress-tested new method, but seems to be fine
    starttime = time.time()
    candidates = {}


    end_time = time.time()
 
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    performance_metadata = f"DBSCAN Finding ran for {elapsed_time} seconds."
    logging.info('DBSCAN finding done')

    return candidates, performance_metadata