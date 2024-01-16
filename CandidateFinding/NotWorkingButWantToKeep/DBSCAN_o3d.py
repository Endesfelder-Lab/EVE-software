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
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "DBSCAN_o3d_onlyHighDensity": {
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
        "DBSCAN_o3d_allEvents": {
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

def o3d_bbox_lookup_parr(bbox_bunch,ms_to_px,point_cloud_data,npyarr):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    filtered_df2={}
    for bboxid, _ in bbox_bunch.items():
        bbox = bbox_bunch[bboxid]
        # Create the axis-aligned bounding box with the specified dimensions
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(bbox[0], bbox[2], bbox[4]*1000*ms_to_px), max_bound=(bbox[1], bbox[3], bbox[5]*1000*ms_to_px))

        # Get the indices of points within the bounding box
        indices = aabb.get_point_indices_within_bounding_box(point_cloud.points)
        filtered_array = npyarr[indices]
        #Change filtered_array to a pd dataframe:
        filtered_df2[bboxid] = pd.DataFrame(filtered_array)
        
    return filtered_df2

def get_events_in_bbox(npyarr,bboxes,ms_to_px,multiThread=False):
    #Get empty candidate dictionary
    candidates = {}
    if multiThread == False:      
        # start_time = time.time()
        # #Loop over all bboxes:
        # for bboxid, _ in bboxes.items():
        #     bbox = bboxes[bboxid]
        #     conditions = [
        #         npyarr['x'] >= bbox[0],
        #         npyarr['x'] <= bbox[1],
        #         npyarr['y'] >= bbox[2],
        #         npyarr['y'] <= bbox[3],
        #         npyarr['t'] >= bbox[4]*1000*ms_to_px,
        #         npyarr['t'] <= bbox[5]*1000*ms_to_px
        #     ]
        #     filtered_array = npyarr[np.logical_and.reduce(conditions)]
        #     #Change filtered_array to a pd dataframe:
        #     filtered_df = pd.DataFrame(filtered_array)
            
        #     candidates[bboxid] = {}
        #     candidates[bboxid]['events'] = filtered_df
        #     candidates[bboxid]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
        #     candidates[bboxid]['N_events'] = len(filtered_array)
        # end_time = time.time()
        # logging.info('Time to get bounding boxes: '+str(end_time-start_time))
        
        import numpy as np
        import pandas as pd
        
        
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
        logging.info('Time to get bounding boxes o3d single-core: '+str(end_time-start_time))
        candidates = candidates2
        

        start_time = time.time()
        point_cloud_data = np.column_stack((npyarr['x'], npyarr['y'], npyarr['t']))



        #only get data between 150-200 and 150-200 in 0, 1st column:
        data_for_o3d = npyarr[(npyarr['x'] > 00) & (npyarr['x'] < 200) & (npyarr['y'] > 00) & (npyarr['y'] < 200)]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
        # dbscan = point_cloud.cluster_dbscan(3, 10, print_progress=True)
        # print(max(dbscan))
        
        #------------------Start intermezzo https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html --------
        import pandas as pd

        df = pd.read_csv(
            filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            header=None,
            sep=',')

        df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
        df.dropna(how="all", inplace=True) # drops the empty line at file-end

        df.tail()
        X = df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].values
        from sklearn.preprocessing import StandardScaler
        X_std = StandardScaler().fit_transform(X)
        
        import numpy as np
        cov_mat = np.cov(X_std.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        print('Eigenvectors \n%s' %eig_vecs)
        print('\nEigenvalues \n%s' %eig_vals)
        
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        tot = sum(eig_vals)
        var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        
        matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

        print('Matrix W:\n', matrix_w)
        
        Y = X_std.dot(matrix_w)
        
        
        
        from sklearn.decomposition import PCA as sklearnPCA
        sklearn_pca = sklearnPCA(n_components=2)
        Y_sklearn = sklearn_pca.fit_transform(X_std)

        # -------------------------------End intermezzo-------------------
        
        

        #only get data between 150-200 and 150-200 in 0, 1st column:
        # data_for_o3d = npyarr[(npyarr['x'] > 00) & (npyarr['x'] < 200) & (npyarr['y'] > 00) & (npyarr['y'] < 200)]
        # import numpy as np
        # normal_array = np.zeros((len(data_for_o3d),3))
        # normal_array[:,0] = data_for_o3d['x']
        # normal_array[:,1] = data_for_o3d['y']
        # normal_array[:,2] = data_for_o3d['t']
        
        # X_std = StandardScaler().fit_transform(normal_array)
        
        # from sklearn.decomposition import PCA as sklearnPCA
        # sklearn_pca = sklearnPCA(n_components=2)
        # Y_sklearn = sklearn_pca.fit_transform(X_std)
        
        
        # import matplotlib.pyplot as plt
        # from matplotlib import colormaps
        # #Display the point cloud points in a 3d scatter:
        # plt.figure()
        # plt.scatter(Y_sklearn[:,0], Y_sklearn[:,1])
        # plt.show()
        
        point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))
        pcc = np.asarray(point_cloud.covariances)
        
        eig_valso3d, eig_vecso3d = np.linalg.eig(pcc)
        sort_order_eig_val = np.argsort(eig_valso3d)[::-1]
        # sortOrder = np.zeros((len(pcc),3))
        # for k in range(3):
        #     for i in range(3):
        #         sortOrder[sort_order_eig_val[:, i] == k][:][k] = 1
        n_sort_dim = 2;
        matrix_w = np.zeros((len(pcc),3,n_sort_dim))
        
        pcamatrix = np.zeros((len(pcc),2))
        
        matrix_w_full = np.zeros((len(pcc),3,2))
        spectral_gap = np.zeros((len(pcc),1))
        
        for i in range(len(pcc)):
            #find the eigenvector with the highest eigenvalue:
            sortorder = np.argsort(eig_valso3d[i])[::-1]
            
            ev = eig_vecso3d[i]
            
            #twice because n_sort_dim = 2
            matrix_w = np.hstack((ev[:,sortorder[0]].reshape(3,1),
                      ev[:,sortorder[1]].reshape(3,1)))
            
            matrix_w_full[i,:,:] = matrix_w
            
            spectral_gap[i] = (eig_valso3d[i][sortorder[0]] - eig_valso3d[i][sortorder[1]])/(sum(eig_valso3d[i]))

            pcamatrix[i,:] = np.asarray(point_cloud.points)[i].dot(matrix_w)

        plt.figure()
        plt.scatter(pcamatrix[:,0], pcamatrix[:,1])
        plt.show()

        matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
                            eig_pairs[1][1].reshape(3,1)))

        print('Matrix W:\n', matrix_w)
    
        point_cloud.estimate_normals()
        pcn = np.asarray(point_cloud.normals)
        
        meanpcn = pcn[:,2]
        
        
        maxeigenval = np.max(eig_valso3d,axis=1)
        mideigenval = np.median(eig_valso3d, axis=1)
        mineigenval = np.min(eig_valso3d,axis=1)
        sumeigenval = np.sum(eig_valso3d,axis=1)
        
        meanpcn = pcc[:,0,0]
        #Display the point cloud points in a 3d scatter:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #plot only clx.labels > -1:
        pcp = np.asarray(point_cloud.points)
        # scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=np.max(eig_valso3d,axis=1),alpha=0.1,cmap='hsv')
        scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=mideigenval,alpha=0.1,cmap='hsv')
        #show the colorbar:
        cbar = plt.colorbar(scatter)
        # ax.scatter(data_for_clx[clx.labels_>-1,0], data_for_clx[clx.labels_>-1,1],  data_for_clx[clx.labels_>-1,2], c=clx.labels_[clx.labels_>-1],cmap='Set1')
        plt.show()
        
        
        #Display the point cloud points in a 3d scatter:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #plot only clx.labels > -1:
        pcp = np.asarray(point_cloud.points)
        maxeigenval = np.max(eig_valso3d,axis=1)
        # scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=np.max(eig_valso3d,axis=1),alpha=0.1,cmap='hsv')
        mevcutoff = 5
        scatter  = ax.scatter(pcp[maxeigenval>=mevcutoff,0], pcp[maxeigenval>=mevcutoff,1],pcp[maxeigenval>=mevcutoff,2], alpha=0.05,c='k',s=1)
        scatter  = ax.scatter(pcp[maxeigenval<mevcutoff,0], pcp[maxeigenval<mevcutoff,1],pcp[maxeigenval<mevcutoff,2],alpha=0.1,c='r')
        #show the colorbar:
        cbar = plt.colorbar(scatter)
        # ax.scatter(data_for_clx[clx.labels_>-1,0], data_for_clx[clx.labels_>-1,1],  data_for_clx[clx.labels_>-1,2], c=clx.labels_[clx.labels_>-1],cmap='Set1')
        plt.show()
        
        
        
        fig = plt.figure()
        # plt.hist(np.max(eig_valso3d,axis=1),bins=100)
        plt.hist(maxeigenval,bins=100)
        plt.xlabel('Maximum Eigenvalue')
        plt.ylabel('Frequency')
        plt.title('Histogram of Maximum Eigenvalues')
        plt.show()
        
        
        bunch_size = 500
        num_cores = multiprocessing.cpu_count()
        num_splits = num_cores#max(1,int(np.ceil(len(bboxes)/bunch_size)))
        
        bbox_bunches = split_dict(bboxes, num_splits)
        
        # o3d_bbox_lookup_parr(bbox_bunches[0],ms_to_px,point_cloud_data,npyarr)
        
        RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(o3d_bbox_lookup_parr)(bbox_bunches[r],ms_to_px,point_cloud_data,npyarr) for r in range(len(bbox_bunches)))
        result = [res for res in RES]
        
        candidates={}
        counter = 0
        for r in range(len(result)):
            for b in result[r].items():
                filtered_array = b[1]
                filtered_df2 = pd.DataFrame(filtered_array)
                
                indexv = counter
                candidates[indexv] = {}
                candidates[indexv]['events'] = filtered_df2
                candidates[indexv]['cluster_size'] = [np.max(filtered_array['y'])-np.min(filtered_array['y']), np.max(filtered_array['x'])-np.min(filtered_array['x']), np.max(filtered_array['t'])-np.min(filtered_array['t'])]
                candidates[indexv]['N_events'] = len(filtered_array)
                counter+=1
                
        end_time = time.time()
        logging.info('Time to get bounding boxes o3d multi-core: '+str(end_time-start_time))
        
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


# def parrallell_o3dkdtreelookup(tarr,pcd_tree3,distance_lookup):
def parrallell_o3dkdtreelookup(args):
    import numpy as np
    import open3d as o3d
    
    tarr, pcd_tree3, distance_lookup = args
    dist_arr = np.zeros(len(tarr))
    #Loop over all points and get the number of neighbours in a certain radius:
    for i in range(len(tarr)):#len(consec_events_arr)):
        [k, _, _] = pcd_tree3.search_radius_vector_3d(tarr[i], distance_lookup)
        # [k, idx, _] = pcd_tree3.search_hybrid_vector_3d(consec_events_arr[i], distance_lookup,100)
        dist_arr[i] = k-1  # Subtract 1 to exclude the point itself
        # print(f"Number of points in radius {distance_lookup} from point {i}: {k}")
    return np.array(dist_arr)

def filter_density_o3d(consec_events,temporal_duration_ms=35,distance_lookup = 4, densityMultiplier = 1.5):
    import open3d as o3d
    
    consec_events_x = consec_events['x']  # Extract 'x' field
    consec_events_y = consec_events['y']  # Extract 'y' field
    consec_events_t = consec_events['t']/(temporal_duration_ms*1000)  # Extract 't' field

    consec_events_arr = np.column_stack((consec_events_x, consec_events_y, consec_events_t))
    #create a point cloud from consec_events:
    # pcd_consec_events = o3d.t.geometry.PointCloud(np.array(consec_events_arr, dtype=np.float64))
    
    
    time_start = time.time()
    searcham = 1000000
    temparr=  consec_events_arr[0:searcham]
    
    #Create a kd-tree from this:
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(temparr)
    pcd_tree3 = o3d.geometry.KDTreeFlann(pcd3)

    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    
    slices = []
    start = 0

    part_size = temparr.shape[0] // num_cores
    remainder = temparr.shape[0] % num_cores
    for i in range(num_cores):
        end = start + part_size + (i < remainder)
        slices.append(temparr[start:end])
        start = end
    
    # # t = parrallell_o3dkdtreelookup(slices[0], pcd_tree3, distance_lookup)
    # # Create a multiprocessing Pool with the desired number of worker processes
    # with multiprocessing.Pool() as pool:
    #     # Prepare the arguments for parallel execution
    #     args = [(slices[i], pcd_tree3, distance_lookup) for i in range(len(slices))]

    #     # Map the function to the pool of worker processes
    #     results = pool.map(parrallell_o3dkdtreelookup, args)


    import concurrent.futures
    # Create a ThreadPoolExecutor with the desired number of worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        # Submit the tasks to the executor
        futures = [
            executor.submit(parrallell_o3dkdtreelookup, slices[i], pcd_tree3, distance_lookup)
            for i in range(len(slices))
        ]

        # Retrieve the results as they become available
        RES = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Flatten the RES array into a single 1D array
    nr_neighbours = np.concatenate(RES)
    
    time_end = time.time()
    print("Time taken: ", time_end - time_start)
    print("Expected total time taken: ", (time_end - time_start)*(len(consec_events_arr)/searcham))
    
#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def DBSCAN_o3d_allEvents(npy_array,settings,**kwargs):
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
    
    # filter_density_o3d(consec_events,temporal_duration_ms=float(kwargs['ratio_ms_to_px']),distance_lookup = float(kwargs['distance_radius_lookup']), densityMultiplier = float(kwargs['density_multiplier']))
    
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

def DBSCAN_o3d_onlyHighDensity(npy_array,settings,**kwargs):
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




    end_time = time.time()
 
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    performance_metadata = f"DBSCAN Finding ran for {elapsed_time} seconds."
    logging.info('DBSCAN finding done')

    return candidates, performance_metadata
