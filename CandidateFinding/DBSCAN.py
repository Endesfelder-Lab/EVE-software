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
import bisect
import numexpr as ne

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "DBSCAN_onlyHighDensity": {
            "required_kwargs": [
                {"name": "min_cluster_size", "description": "Required number of neighbouring events in spatiotemporal voxel","default":17,"type":int,"display_text":"Minimum cluster size"},
                {"name": "distance_radius_lookup", "description": "Outer radius (in px) to count the neighbours in.","default":7,"type":int,"display_text":"Distance radius lookup"},
                {"name": "density_multiplier", "description": "Distance multiplier","default":1.5,"type":float,"display_text":"Density multiplier"},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float,"display_text":"Ratio ms to px"},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":6,"type":int,"display_text":"DBSCAN epsilon"},
            ],
            "optional_kwargs": [
               {"name": "min_consec", "description": "Minimum number of consecutive events","default":1,"type":int,"display_text":"Min. consec"},
               {"name": "max_consec", "description": "Maximum number of consecutive events, discards hot pixels","default":30,"type":int,"display_text":"Max. consec"},
            ],
            "help_string": "DBSCAN, only return events that are considered high-density.",
            "display_name": "DBSCAN returning high-density events"
        },
        "DBSCAN_allEvents": {
            "required_kwargs": [
                {"name": "min_cluster_size", "description": "Required number of neighbouring events in spatiotemporal voxel","default":17,"type":int,"display_text":"Minimum cluster size"},
                {"name": "distance_radius_lookup", "description": "Outer radius (in px) to count the neighbours in.","default":7,"type":int,"display_text":"Distance radius lookup"},
                {"name": "density_multiplier", "description": "Distance multiplier","default":1.5,"type":float,"display_text":"Density multiplier"},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float,"display_text":"Ratio ms to px"},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":6,"type":int,"display_text":"DBSCAN epsilon"},
                {"name": "padding_xy", "description": "Result padding in x,y pixels.","default":0,"type":int,"display_text":"XY padding"},
            ],
            "optional_kwargs": [
               {"name": "min_consec", "description": "Minimum number of consecutive events","default":1,"type":int,"display_text":"Min. consec"},
               {"name": "max_consec", "description": "Maximum number of consecutive events, discards hot pixels","default":30,"type":int,"display_text":"Max. consec"},
            ],
            "help_string": "DBSCAN on high-density, but returns all events in the bounding box specified by DBSCAN.",
            "display_name": "DBSCAN returning all events"
        },
        "DBSCAN_allEvents_remove_outliers": {
            "required_kwargs": [
                {"name": "neighbour_points", "description": "Removes points that has less than this number of neighbours in neighbour_radius","default":30,"type":int,"display_text":"Minimum neighbours"},
                {"name": "neighbour_radius", "description": "Removes points that has less than neighbour_points in this radius","default":3.0,"type":float,"display_text":"Neighbour radius"},
                {"name": "min_cluster_size", "description": "Required number of neighbouring events in spatiotemporal voxel","default":17,"type":int,"display_text":"Minimum cluster size"},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float,"display_text":"Ratio ms to px"},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":3,"type":int,"display_text":"DBSCAN epsilon"},
                {"name": "padding_xy", "description": "Result padding in x,y pixels.","default":2,"type":int,"display_text":"XY padding"},
            ],
            "optional_kwargs": [
               {"name": "min_consec", "description": "Minimum number of consecutive events","default":1,"type":int,"display_text":"Min. consec"},
               {"name": "max_consec", "description": "Maximum number of consecutive events, discards hot pixels","default":30,"type":int,"display_text":"Max. consec"},
            ],
            "help_string": "Removes outliers via o3d's remove_radius_outlier.",
            "display_name": "DBSCAN returning all events, using radius outlier removal"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------


def remove_radius_outlier_o3d(events,nb_points=30,radius=3,print_progress=True,ms_to_px=35):
    polarities = events['p']
    
    data_for_o3d = events
    start_time = time.time()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
    
    cleaned_pcp =point_cloud.remove_radius_outlier(30, 3, print_progress=True)
    
    pcp = np.asarray(cleaned_pcp[0].points)
    #change columns 0 and 1 to integer values:
    pcp[:,0] = pcp[:,0].astype(np.int64)
    pcp[:,1] = pcp[:,1].astype(np.int64)
    
    polarities_remaining=polarities[cleaned_pcp[1]]
    
    #transform back to titled columns:
    # events_remaining = pd.DataFrame({'x': pcp[:, 0], 'y': pcp[:, 1], 't': pcp[:, 2], 'p': polarities_remaining})
    
    logging.info(str(len(pcp))+"/"+str(len(events))+":"+str(len(pcp)/len(events)*100)+" % of data kept after remove-radius-outlier")
    return pcp, polarities_remaining
    
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
    # filtering out events with a minimum number of consecutive events
    df_events = df_events[(df_events['w']>=min_consec)]

    # filtering out pixels that have more than max_consec number of consecutive events
    high_consec_events = df_events[(df_events['w']>max_consec)]
    hotPixels = high_consec_events[['x','y']].values.tolist()
    unique_hotPixels = set(map(tuple, hotPixels))
    mask = df_events[['x', 'y']].apply(tuple, axis=1).isin(unique_hotPixels)
    df_events = df_events[~mask]

    # convert df back to structured numpy array
    consec_events = df_events.to_records(index=False)

    return consec_events

def hotPixel_filter(events, max_consec, weights=None,df_events=None):
    # This function filters out events with a maximum number of consecutive events
    if weights is None:
        weights,df_events = determineWeights(events)
    
    # filtering out pixels that have more than max_consec number of consecutive events
    high_consec_events = df_events[(df_events['w']>max_consec)]
    hotPixels = high_consec_events[['x','y']].values.tolist()
    unique_hotPixels = set(map(tuple, hotPixels))
    mask = df_events[['x', 'y']].apply(tuple, axis=1).isin(unique_hotPixels)
    df_events = df_events[~mask]

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
            candidates[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y'])+1, np.max(filtered_array['x'])-np.min(filtered_array['x'])+1, np.max(filtered_array['t'])-np.min(filtered_array['t'])]
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

def get_events_in_bbox_bisect(npyarr,bboxes,ms_to_px):
    
    #Sort on t:
    npy_sort_t = np.zeros((len(npyarr),2))
    #First column is the order:
    npy_sort_t[:,0] = np.argsort(npyarr['t'])
    #second column is the time:
    npy_sort_t[:,1] = np.sort(npyarr, order='t')['t']
    
    #Sort on x:
    npy_sort_x = np.zeros((len(npyarr),2))
    #First column is the order:
    npy_sort_x[:,0] = np.argsort(npyarr['x'])
    #second column is the time:
    npy_sort_x[:,1] = np.sort(npyarr, order='x')['x']
    
    npy_sort_y = np.zeros((len(npyarr),2))
    #First column is the order:
    npy_sort_y[:,0] = np.argsort(npyarr['y'])
    #second column is the time:
    npy_sort_y[:,1] = np.sort(npyarr, order='y')['y']
    
    candidates = {} 
    start_time = time.time()
    #Loop over all bboxes:
    for bboxid, _ in bboxes.items():
        bbox = bboxes[bboxid]
        startindex_x = bisect.bisect_left(npy_sort_x[:,1],bbox[0])
        endindex_x = bisect.bisect_right(npy_sort_x[:,1],bbox[1])
        startindex_y = bisect.bisect_left(npy_sort_y[:,1],bbox[2])
        endindex_y = bisect.bisect_right(npy_sort_y[:,1],bbox[3])
        startindex_t = bisect.bisect_left(npy_sort_t[:,1],bbox[4]*1000*ms_to_px)
        endindex_t = bisect.bisect_right(npy_sort_t[:,1],bbox[5]*1000*ms_to_px)
        
        valid_indeces_x = npy_sort_x[startindex_x:endindex_x,0]
        valid_indeces_y = npy_sort_y[startindex_y:endindex_y,0]
        valid_indeces_t = npy_sort_t[startindex_t:endindex_t,0]
        
        #find the indeces where all of those is true:
        common_values = np.intersect1d(np.intersect1d(valid_indeces_x, valid_indeces_y), valid_indeces_t)

        filtered_array = npyarr[common_values.astype(np.int64)]
        #Change filtered_array to a pd dataframe:
        filtered_df = pd.DataFrame(filtered_array)
        
        candidates[bboxid] = {}
        candidates[bboxid]['events'] = filtered_df
        candidates[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
        candidates[bboxid]['N_events'] = len(filtered_array)
    end_time = time.time()
    
    logging.info('Time to get bounding boxesBisect: '+str(end_time-start_time))
    return candidates

def get_events_in_bbox_NE(npyarr,bboxes,ms_to_px):
    #Faster bbox-finding than bisect or single/multi-thread above:
    #First, get all events within the biggest 3D sphere around the bbox via NE (numexpr)
    candidates={}
    xdata = npyarr['x']
    ydata = npyarr['y']
    tdata = npyarr['t']/(1000*ms_to_px)
    for bboxid, _ in bboxes.items():
        bbox = bboxes[bboxid]
        
        #Idea: via ne, get all locs in a circle around the bbox, then filter the events in that circle
        bbox_midpoint = (np.mean([bbox[0],bbox[1]]),np.mean([bbox[2],bbox[3]]),np.mean([bbox[4],bbox[5]]))
        cx = bbox_midpoint[0]
        cy = bbox_midpoint[1]
        ct = bbox_midpoint[2]
        bbox_radius = np.max([bbox[1]-bbox[0],bbox[3]-bbox[2],bbox[5]-bbox[4]])
        
        res = ne.evaluate('((xdata-cx)**2 + (ydata-cy)**2 + (tdata-ct))<bbox_radius**2')
        
        firstFilter = npyarr[res]
        
        #Then perform a 'normal' bbox filter which is now much faster since we already filtered out 99% of data
        conditions = [
            firstFilter['x'] >= bbox[0],
            firstFilter['x'] <= bbox[1],
            firstFilter['y'] >= bbox[2],
            firstFilter['y'] <= bbox[3],
            firstFilter['t'] >= bbox[4]*(1000*ms_to_px),
            firstFilter['t'] <= bbox[5]*(1000*ms_to_px)
        ]
        filtered_array = firstFilter[np.logical_and.reduce(conditions)]
        #Change filtered_array to a pd dataframe:
        filtered_df = pd.DataFrame(filtered_array)
        candidates[bboxid] = {}
        candidates[bboxid]['events'] = filtered_df
        candidates[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y'])+1, np.max(filtered_array['x'])-np.min(filtered_array['x'])+1, np.max(filtered_array['t'])-np.min(filtered_array['t'])]
        candidates[bboxid]['N_events'] = len(filtered_array)
    
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
    logging.info('Hotpixel filtering completed')
    # candidates = get_events_in_bbox(hotpixel_filtered_events,bboxes,float(kwargs['ratio_ms_to_px']))
    # logging.info('Candidates obtained')
    # candidates = get_events_in_bbox_bisect(hotpixel_filtered_events,bboxes,float(kwargs['ratio_ms_to_px']))
    candidates = get_events_in_bbox_NE(hotpixel_filtered_events,bboxes,float(kwargs['ratio_ms_to_px']))
    logging.info('Candidates obtained (via bbox NE)')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    performance_metadata = ""
    logging.info('DBSCAN finding done')
    
    #Remove small/large bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)


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
    if len(clusters)>0:
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

    performance_metadata = ""
    logging.info('DBSCAN finding done')
    
    #Remove small/large bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)


    return candidates, performance_metadata

def DBSCAN_allEvents_remove_outliers(npy_array,settings,**kwargs):
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
    
    
    weights,df_events = determineWeights(npy_array)
    hotpixel_filtered_events = hotPixel_filter(npy_array,max_consec_ev,weights=weights,df_events=df_events)
    logging.info('Hotpixel filtering completed')
    filtered_events, polarities = remove_radius_outlier_o3d(hotpixel_filtered_events,nb_points=30,radius=3,print_progress=True,ms_to_px=float(kwargs['ratio_ms_to_px']))
    
    
    clustersHD, cluster_labels = clustering(filtered_events, polarities, eps = float(kwargs['DBSCAN_eps']), min_points_per_cluster = int(kwargs['min_cluster_size']))
    logging.info('DBSCAN done')
    bboxes = get_cluster_bounding_boxes(clustersHD, cluster_labels,padding_xy = int(kwargs['padding_xy']),padding_t = int(kwargs['padding_xy']))
    logging.info('Getting bounding boxes done')
    
    candidates = get_events_in_bbox_bisect(hotpixel_filtered_events,bboxes,float(kwargs['ratio_ms_to_px']))
    logging.info('Candidates obtained (via bbox bisect)')
    
    #Remove small/large bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)

    
    performance_metadata = ""
    return candidates, performance_metadata