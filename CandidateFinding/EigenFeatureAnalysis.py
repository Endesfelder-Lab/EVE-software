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
from scipy.optimize import curve_fit, root
import numexpr as ne

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "eigenFeature_analysis": {
            "required_kwargs": [
                {"name": "linearity_cutoff", "description": "Linearity (0-1) cutoff","default":0.7,"type":float},
                {"name": "max_eigenval_cutoff", "description": "Cutoff of maximum eigenvalue. Set to zero to auto-determine this!","default":0.0,"type":float},
                {"name": "search_n_neighbours", "description": "Number of (closest) neighbours for the covariance determination","default":50,"type":int},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":20.0,"type":float},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":4,"type":int},
                {"name": "DBSCAN_n_neighbours", "description": "Minimum nr of points for DBSCAN cluster.","default":20,"type":int},
                
            ],
            "optional_kwargs": [
                {"name": "debug", "description": "Get some debug info.","default":False},
            ],
            "help_string": "Eigen-feature analysis",
            "display_name": "Eigen-feature analysis"
        },
        "eigen_feature_analysis_autoRadiusSelect": {
            "required_kwargs": [
                {"name": "linearity_cutoff", "description": "Cutoff of normalized eigenvalues","default":0.6,"type":float},
                {"name": "max_eigenval_cutoff", "description": "Cutoff of maximum eigenvalue. Set to zero to auto-determine this!","default":0.0,"type":float},
                {"name": "min_radius", "description": "Minimal radius","default":4,"type":float},
                {"name": "max_radius", "description": "Maximum radius","default":10,"type":float},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":3,"type":int},
                {"name": "DBSCAN_n_neighbours", "description": "Minimum nr of points for DBSCAN cluster.","default":17,"type":int},
            ],
            "optional_kwargs": [
                {"name": "debug", "description": "Get some debug info.","default":False},
            ],
            "help_string": "Eigen-feature analysis with entropy-baed radius finding.",
            "display_name": "Eigen-feature analysis, automatic radius finding"
        },
        "eigenFeature_analysis_and_bbox_finding": {
            "required_kwargs": [
                {"name": "norm_eigenval_cutoff", "description": "Cutoff of normalized eigenvalues","default":0.7,"type":float},
                {"name": "max_eigenval_cutoff", "description": "Cutoff of maximum eigenvalue. Set to zero to auto-determine this!","default":0.0,"type":float},
                {"name": "search_n_neighbours", "description": "Number of (closest) neighbours for the covariance determination","default":50,"type":int},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":20.0,"type":float},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":3,"type":int},
                {"name": "DBSCAN_n_neighbours", "description": "Minimum nr of points for DBSCAN cluster.","default":20,"type":int},
                {"name": "bbox_padding", "description": "Padding of the bounding box (in px equivalents).","default":1,"type":int},
            ],
            "optional_kwargs": [
                {"name": "debug", "description": "Get some debug info.","default":False},
            ],
            "help_string": "Eigen-feature analysis with bbox finding.",
            "display_name": "Eigen-feature analysis with bbox finding"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def three_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3):
    """Function to fit a histogram with 3 Gaussian curves."""
    return (a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) +
            a2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) +
            a3 * np.exp(-0.5 * ((x - mu3) / sigma3) ** 2))
def single_gaussian(x, a1, mu1, sigma1):
    """Function to get a single gauss."""
    return (a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2))
       
def find_gaussian_cross(a1, b1, c1, a2, b2, c2, minp = 0, maxp = 100, accuracy = 0.1): 
    """Function to find the intersection of two gaussians."""
    x = np.arange(minp, maxp, accuracy)
    switchpoint=[]
    #find which gauss is higher at start:
    if single_gaussian(0, a1, b1, c1) > single_gaussian(0, a2, b2, c2):
        currp = 0
    else:
        currp = 1
    #loop over values of x:
    for i in x:
        if single_gaussian(i, a1, b1, c1) > single_gaussian(i, a2, b2, c2):
            if currp == 1:
                switchpoint.append(i)
                currp = 0
        else:
            if currp == 0:
                switchpoint.append(i)
                currp = 1
    
    return switchpoint

def compute_eig(pcc):
    return np.linalg.eig(pcc)

def EigenValueCalculation(npyarr,kwargs):
    ms_to_px=float(kwargs['ratio_ms_to_px'])
    data_for_o3d = npyarr
    multiCore = False
    if multiCore == True:
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
        
        logging.debug('point cloud created')
        point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(kwargs['search_n_neighbours'])))
        logging.debug('pcc estimated')
        pcc = np.asarray(point_cloud.covariances)
        point_cloud.estimate_normals(fast_normal_computation=True,search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(kwargs['search_n_neighbours'])))
        
        N = multiprocessing.cpu_count()  # Number of chunks
        pcc_chunks = np.array_split(pcc, N)

        results = Parallel(n_jobs=-1)(delayed(compute_eig)(pcc_chunk) for pcc_chunk in pcc_chunks)
        
        eig_valso3d_list = []
        eig_valso3d_list.extend([res[0] for res in results])
        eig_valso3d = np.concatenate(eig_valso3d_list, axis=0)
        logging.debug('eigv calculated')
        end_time = time.time()
        logging.info('Eigenvalue calculation time: '+ str(end_time - start_time))
    
    else:
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
        logging.debug('point cloud created')
        point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(kwargs['search_n_neighbours'])))
        logging.debug('pcc estimated')
        pcc = np.asarray(point_cloud.covariances)        
        eig_valso3d = np.linalg.svd(pcc,compute_uv=False,hermitian=True)
        #svdres = np.linalg.svd(pcc,compute_uv=True,hermitian=True) #eigenvals == svdres.S == singular values
        logging.debug('eigv calculated')
        end_time = time.time()
        logging.info('Eigenvalue calculation time: '+ str(end_time - start_time))
    
    return eig_valso3d, point_cloud

def EigenValueCalculation_radius(npyarr,radius,kwargs):
    ms_to_px=float(kwargs['ratio_ms_to_px'])
    data_for_o3d = npyarr
    multiCore = False
    if multiCore == True:
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
        
        logging.debug('point cloud created')
        point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=float(radius)))

        N = 150  # Number of chunks
        pcc_chunks = np.array_split(pcc, N)

        results = Parallel(n_jobs=-1)(delayed(compute_eig)(pcc_chunk) for pcc_chunk in pcc_chunks)
        
        eig_valso3d_list = []
        eig_valso3d_list.extend([res[0] for res in results])
        eig_valso3d = np.concatenate(eig_valso3d_list, axis=0)
        
        
        logging.debug('eigv calculated')
        end_time = time.time()
        logging.info('Eigenvalue calculation time: '+ str(end_time - start_time))
    
    else:
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
        logging.debug('point cloud created')
        point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=float(radius)))
        logging.debug('pcc estimated')
        pcc = np.asarray(point_cloud.covariances)        
        eig_valso3d = np.linalg.svd(pcc,compute_uv=False,hermitian=True)
        logging.debug('eigv calculated')
        end_time = time.time()
        logging.info('Eigenvalue calculation time: '+ str(end_time - start_time))
    
    return eig_valso3d, point_cloud

def determineEigenValueCutoffComputationally(maxeigenval,kwargs):
    #Fit this histogram with three gaussians:
    #create a normalized histogram:
    hist, bin_edges = np.histogram(maxeigenval, bins=100, density=True)
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ratio_bg_signal = 0.2
    initial_noise = (np.max(hist)*(1-ratio_bg_signal),np.median(maxeigenval),np.std(maxeigenval)/2)
    initial_signal = (np.max(hist[bin_centers<5])*(1-ratio_bg_signal),np.median(maxeigenval[maxeigenval<5]),np.std(maxeigenval[maxeigenval<5]))
    initial_bg = (np.max(hist)*ratio_bg_signal,np.median(maxeigenval),np.std(maxeigenval))
    #add these together in a single array of 9-by-1:
    # Concatenate initial_noise, initial_signal, and initial_bg into a single array
    initial_params = np.concatenate((initial_noise, initial_signal, initial_bg)).flatten()
    # Fit the histogram with two Gaussian curves
    try:
        
        
        # Old, cumbersome method:
        # params, _ = curve_fit(three_gaussians, bin_centers, hist, p0=initial_params)
        # # Use scipy.optimize.root to find the x-position where the two Gaussians cross
        # #Find the cross of the lowest with second-lowest gaussian:
        # param_order = np.argsort(params[[1,4,7]])
        
        # #Get the crossings:
        # result1 = find_gaussian_cross(a1=params[param_order[0]*3],b1=params[param_order[0]*3+1],c1=params[param_order[0]*3+2],a2=params[param_order[1]*3],b2=params[param_order[1]*3+1],c2=params[param_order[1]*3+2],minp = 0,maxp=max(bin_centers),accuracy = 0.01)
        
        # result2 = find_gaussian_cross(a1=params[param_order[0]*3],b1=params[param_order[0]*3+1],c1=params[param_order[0]*3+2],a2=params[param_order[2]*3],b2=params[param_order[2]*3+1],c2=params[param_order[2]*3+2],minp = 0,maxp=max(bin_centers),accuracy = 0.01)
        
        # #Find the lowest crossing:
        # lowest_crossing = min(min(result1), min(result2))
        # maxeigenvalcutoff = lowest_crossing
        
        #Better method:
        
        from scipy import signal
        nbins=100
        hist, bin_edges = np.histogram(maxeigenval, np.linspace(0,max(maxeigenval),nbins))  # Adjust the number of bins as needed
        peakind = signal.find_peaks_cwt(max(hist)-hist, np.arange(1,nbins/4))
        maxeigenvalcutoff = min(bin_edges[peakind]+bin_edges[1])
            
        
        if maxeigenvalcutoff < 1:
            logging.info(f'Max eigenval thought to be {maxeigenvalcutoff}')
            logging.error('Eigenvalue is very very low! Changed to a sensible value but should be investigated further')
            maxeigenvalcutoff = 7
        
        logging.info(f'Max eigenval determined to be {maxeigenvalcutoff}')
    except Exception as e:
        logging.error('Automatical eigenvalue determination failed with error: ', e)
        logging.error('Eigenvalue set to a logical value but should be investigated further')
        maxeigenvalcutoff = 7;

        
    if utilsHelper.strtobool(kwargs['debug']):
        try:
            from scipy import signal
            nbins=100
            hist, bin_edges = np.histogram(maxeigenval, np.linspace(0,max(maxeigenval),nbins))  # Adjust the number of bins as needed
            peakind = signal.find_peaks_cwt(max(hist)-hist, np.arange(1,nbins/4))
            print(bin_edges[peakind]+bin_edges[1])
            

            fig, ax1 = plt.subplots()

            # Plot the histogram on the first y-axis
            ax1.hist(maxeigenval, np.linspace(0,max(maxeigenval),nbins), alpha=0.5, label='Histogram', density=True, color='tab:blue')
            ax1.set_xlabel('X data')
            ax1.set_ylabel('Histogram', color='tab:blue')

            # Create a second y-axis
            ax2 = ax1.twinx()

            # Display the plot
            plt.show()
            
            
        except:
            pass
        
    return maxeigenvalcutoff

def clusterPoints_to_candidates(clusterpoints,cluster_labels,ms_to_px):
    # generate the candidates in from of a dictionary
    headers = ['x', 'y', 't', 'p']
    #time back to micros:
    clusterpoints[:,2] *=1000*ms_to_px
    #Integerise the data:
    new_arr = pd.DataFrame(clusterpoints, columns=headers)
    new_arr['x'] = new_arr['x'].astype(int)
    new_arr['y'] = new_arr['y'].astype(int)
    new_arr['t'] = new_arr['t'].astype(np.int64)
    new_arr['p'] = new_arr['p'].astype(int)
    
    candidates={}
    starttime = time.time()
    new_arr['Cluster_Labels'] = cluster_labels
    grouped_clusters = new_arr.groupby('Cluster_Labels')
    #This should be sped up: 
    midtime = time.time()
    for cl, cluster_events in grouped_clusters:
        if cl > -1:
            clusterEvents = cluster_events.drop(columns='Cluster_Labels')
            candidates[cl] = {}
            candidates[cl]['events'] = clusterEvents
            candidates[cl]['N_events'] = int(len(clusterEvents))
            # candidates[cl]['cluster_size'] = [int((clusterEvents['y']).max()-(clusterEvents['y']).min()), int((clusterEvents['x']).max()-(clusterEvents['x']).min()), int((clusterEvents['t']).max()-(clusterEvents['t']).min())]
            
            #We know that they're time-sorted, so for time, we can do end-start, which is max-min. Speeds up a little
            candidates[cl]['cluster_size'] = [int((clusterEvents['y']).max()-(clusterEvents['y']).min()), int((clusterEvents['x']).max()-(clusterEvents['x']).min()), int(clusterEvents['t'].iloc[-1]-clusterEvents['t'].iloc[0])]
            
    endtime = time.time()
    
    return candidates

def get_cluster_bounding_boxes(events, cluster_labels,padding_xy=0,padding_t=0):
    start_time = time.time()
    bounding_boxes = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id != -1:  # Ignore noise points
            bounding_boxes[cluster_id] = compute_bounding_boxC(cluster_id,events=events,cluster_labels=cluster_labels,padding_xy=padding_xy,padding_t=padding_t)
    
    end_time = time.time()
    logging.info('Time to get bounding boxesC: '+str(end_time-start_time))
    
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
    x_coordinates = cluster_points[:,0].tolist()
    y_coordinates = cluster_points[:,1].tolist()
    t_coordinates = cluster_points[:,2].tolist()
    return [min(x_coordinates)-padding_xy, max(x_coordinates)+padding_xy, min(y_coordinates)-padding_xy, max(y_coordinates)+padding_xy, min(t_coordinates)-padding_t, max(t_coordinates)+padding_t]


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

def dbscan_with_trialError(clusterpoints,eps=3,n_neighbours=15):
    if len(clusterpoints) > 0:
        logging.info("DBSCANning started")
        
        #This seems to fail like 0.1% of the time, but can't replicate it, even with the exact same data, thus:
        try:
            #Throw DBSCAN on the 'pre-clustered' data:
            dbscan = DBSCAN(eps=eps, n_jobs=-1, min_samples=n_neighbours)
            cluster_labels = dbscan.fit_predict(clusterpoints)
        except:
            try:
                logging.warning("DBSCANning multicore errored on this data, unsure why, trying single-core")
                dbscan = DBSCAN(eps=eps, n_jobs=1, min_samples=n_neighbours)
                cluster_labels = dbscan.fit_predict(clusterpoints)
            except:
                cluster_labels = [-1]*len(clusterpoints)
                logging.warning("DBSCANning errored on this data, unsure why")
        
        # Print how many clusters are found
        logging.info("Number of clusters found:"+ str(max(cluster_labels)+1))
        return cluster_labels

def eigenFeature_analysis(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    starttime = time.time()
    npyarr=npy_array
    polarities = npyarr['p']
    ms_to_px=float(kwargs['ratio_ms_to_px'])
    
    eig_valso3d, point_cloud = EigenValueCalculation(npyarr,kwargs)
    
    #Calculate linearity
    linearity = (eig_valso3d[:,0]-eig_valso3d[:,1])/eig_valso3d[:,0]
    
    maxeigenval = np.max(eig_valso3d,axis=1)
    mideigenval = np.median(eig_valso3d, axis=1)
    mineigenval = np.min(eig_valso3d,axis=1)
    sumeigenval = np.sum(eig_valso3d,axis=1)
    
    normeigenval = eig_valso3d/sumeigenval.reshape(-1, 1)
    stdeigenval =np.std(normeigenval,axis=1)
    
    # highstdeigenval = normeigenval[stdeigenval>0.45]
    
    linearity_cutoff = float(kwargs['linearity_cutoff']) #0.6
    maxeigenvalcutoff = float(kwargs['max_eigenval_cutoff']) #6
    
    #If set to zero, we do it computationally:
    if maxeigenvalcutoff == 0:
        maxeigenvalcutoff = determineEigenValueCutoffComputationally(maxeigenval,kwargs)
    
    points = np.asarray(point_cloud.points)
    #Add polarity back to points
    points = np.concatenate((points, polarities.reshape(-1,1)), axis=1)
    
    clusterpoints = points[(linearity < linearity_cutoff) & (maxeigenval < maxeigenvalcutoff), :]
    noisepoints = points[(linearity >= linearity_cutoff) | (maxeigenval >= maxeigenvalcutoff), :]

    cluster_labels = dbscan_with_trialError(clusterpoints,eps=int(kwargs['DBSCAN_eps']),n_neighbours=int(kwargs['DBSCAN_n_neighbours']))
    
    if len(clusterpoints) > 0:
        candidates = clusterPoints_to_candidates(clusterpoints,cluster_labels,ms_to_px)
    else:
        logging.error("No clusterpoints found via spectral clustering - maximum eigenval probably wrong!")
        candidates = {}
    
    #Remove large bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)

    performance_metadata = f"SpectralClustering Finding ran for {time.time() - starttime} seconds."
    
    return candidates, performance_metadata

def eigenFeature_analysis_and_bbox_finding(npy_array,settings,**kwargs):
        #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    starttime = time.time()
    npyarr=npy_array
    polarities = npyarr['p']
    ms_to_px=float(kwargs['ratio_ms_to_px'])
    
    eig_valso3d, point_cloud = EigenValueCalculation(npyarr,kwargs)
    
    #Calculate linearity
    linearity = (eig_valso3d[:,0]-eig_valso3d[:,1])/eig_valso3d[:,0]
    
    maxeigenval = np.max(eig_valso3d,axis=1)
    mideigenval = np.median(eig_valso3d, axis=1)
    mineigenval = np.min(eig_valso3d,axis=1)
    sumeigenval = np.sum(eig_valso3d,axis=1)
    
    normeigenval = eig_valso3d/sumeigenval.reshape(-1, 1)
    stdeigenval =np.std(normeigenval,axis=1)
    
    # highstdeigenval = normeigenval[stdeigenval>0.45]
    
    linearity_cutoff = float(kwargs['linearity_cutoff']) #0.6
    maxeigenvalcutoff = float(kwargs['max_eigenval_cutoff']) #6
    
    #If set to zero, we do it computationally:
    if maxeigenvalcutoff == 0:
        maxeigenvalcutoff = determineEigenValueCutoffComputationally(maxeigenval,kwargs)
    
    points = np.asarray(point_cloud.points)
    #Add polarity back to points
    points = np.concatenate((points, polarities.reshape(-1,1)), axis=1)
    
    clusterpoints = points[(linearity < linearity_cutoff) & (maxeigenval < maxeigenvalcutoff), :]
    noisepoints = points[(linearity >= linearity_cutoff) | (maxeigenval >= maxeigenvalcutoff), :]

    cluster_labels = dbscan_with_trialError(clusterpoints,eps=int(kwargs['DBSCAN_eps']),n_neighbours=int(kwargs['DBSCAN_n_neighbours']))
    
    #Now we get BBOXes around each cluster:
    bboxes = get_cluster_bounding_boxes(clusterpoints, cluster_labels,padding_xy=float(kwargs['bbox_padding']),padding_t=float(kwargs['bbox_padding']))
    
    #adapt noHotPixelPoints array so that the columns are 'named' again:
    noHotPixelPoints = points[(np.max(normeigenval, axis=1) < linearity_cutoff), :]
    noHotPixelPoints_rec = pd.DataFrame(noHotPixelPoints,columns=['x','y','t','p'])
    noHotPixelPoints_rec['x'] = noHotPixelPoints_rec['x'].astype(int)
    noHotPixelPoints_rec['y'] = noHotPixelPoints_rec['y'].astype(int)
    noHotPixelPoints_rec['t'] *= ms_to_px*1000
    noHotPixelPoints_rec['t'] = noHotPixelPoints_rec['t'].astype(np.int64)
    noHotPixelPoints_rec['p'] = noHotPixelPoints_rec['p'].astype(int)
    
    candidates = get_events_in_bbox_NE(noHotPixelPoints_rec,bboxes,float(kwargs['ratio_ms_to_px']))
    
    #Remove large/small bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)
    
    performance_metadata = f"SpectralClustering Finding ran for {time.time() - starttime} seconds."
    
    return candidates, performance_metadata
    
def eigen_feature_analysis_autoRadiusSelect(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    
    # starttime = time.time()
    # npyarr=npy_array
    # polarities = npyarr['p']
    # ms_to_px=float(kwargs['ratio_ms_to_px'])
    # kwargs['search_n_neighbours'] = 100
    
    # eig_valso3d, point_cloud = EigenValueCalculation_radius(npyarr,20,kwargs)
    
    # #First sort the eigenvalues large to small:
    # eig_valso3d = np.sort(eig_valso3d, axis=1)[:, ::-1]
    # #Normalise them to sum to one?
    # eig_valso3d = eig_valso3d/np.sum(eig_valso3d,axis=1).reshape(-1, 1)
    
    # linearity = (eig_valso3d[:,0]-eig_valso3d[:,1])/eig_valso3d[:,0]
    # planarity = (eig_valso3d[:,1]-eig_valso3d[:,2])/eig_valso3d[:,0]
    # sphericity = (eig_valso3d[:,2])/eig_valso3d[:,0]
    
    # #Make a histogram or 3:
    
    # fig = plt.figure()
    # ax = fig.add_subplot(331)
    # #Add polarity back to points
    # ax.hist(eig_valso3d[:,0],bins=50)
    # ax.set_title('MaxEig')
    # ax = fig.add_subplot(332)
    # ax.hist(eig_valso3d[:,1],bins=50)
    # ax.set_title('MidEig')
    # ax = fig.add_subplot(333)
    # ax.hist(eig_valso3d[:,2],bins=50)
    # ax.set_title('MinEig')
    # ax = fig.add_subplot(334)
    # ax.hist(linearity,bins=50)
    # ax.set_title('Linearity')
    # ax = fig.add_subplot(335)
    # ax.hist(planarity,bins=50)
    # ax.set_title('Planarity')
    # ax = fig.add_subplot(336)
    # ax.hist(sphericity,bins=50)
    # ax.set_title('Sphericity')
    # ax = fig.add_subplot(337)
    # ax.hist(np.sum(eig_valso3d,axis=1),bins=50)
    # ax.set_title('Sum eigenvals')
    # plt.show()
        
    
    # maxeigenval = np.max(eig_valso3d,axis=1)
    # mideigenval = np.median(eig_valso3d, axis=1)
    # mineigenval = np.min(eig_valso3d,axis=1)
    # sumeigenval = np.sum(eig_valso3d,axis=1)
    
    
    
    # normeigenval = eig_valso3d/sumeigenval.reshape(-1, 1)
    # stdeigenval =np.std(normeigenval,axis=1)
    
    # # highstdeigenval = normeigenval[stdeigenval>0.45]
    
    # normeigenvalcutoff = float(kwargs['norm_eigenval_cutoff']) #0.6
    # maxeigenvalcutoff = float(kwargs['max_eigenval_cutoff']) #6
    
    # #If set to zero, we do it computationally:
    # if maxeigenvalcutoff == 0:
    #     maxeigenvalcutoff = determineEigenValueCutoffComputationally(maxeigenval,kwargs)
    
    # points = np.asarray(point_cloud.points)
    # #Add polarity back to points
    # points = np.concatenate((points, polarities.reshape(-1,1)), axis=1)
    
    # clusterpoints = points[(np.max(normeigenval, axis=1) < normeigenvalcutoff) & (maxeigenval < maxeigenvalcutoff), :]
    # noisepoints = points[(np.max(normeigenval, axis=1) >= normeigenvalcutoff) | (maxeigenval >= maxeigenvalcutoff), :]

    # if len(clusterpoints) > 0:
    #     logging.info("DBSCANning started")
    #     #Throw DBSCAN on the 'pre-clustered' data:
    #     dbscan = DBSCAN(eps=int(kwargs['DBSCAN_eps']), n_jobs=-1, min_samples=int(kwargs['DBSCAN_n_neighbours']))
    #     cluster_labels = dbscan.fit_predict(clusterpoints)
        
    #     # Print how many clusters are found
    #     logging.info("Number of clusters found:"+ str(max(cluster_labels)))
        
    #     candidates = clusterPoints_to_candidates(clusterpoints,cluster_labels,ms_to_px)
    # else:
    #     logging.error("No clusterpoints found via spectral clustering - maximum eigenval probably wrong!")
    #     candidates = {}
        
    # performance_metadata = f"SpectralClustering Finding ran for {time.time() - starttime} seconds."
    
    # return candidates, performance_metadata


    starttime = time.time()
    npyarr=npy_array
    polarities = npyarr['p']
    ms_to_px=float(kwargs['ratio_ms_to_px'])
    
    
    radoptions = np.linspace(float(kwargs['min_radius']),float(kwargs['max_radius']),10,dtype=float)
    
    lin = np.zeros((len(npyarr),radoptions.shape[0]))
    eigval1 = np.zeros((len(npyarr),radoptions.shape[0]))
    eigval2 = np.zeros((len(npyarr),radoptions.shape[0]))
    eigval3 = np.zeros((len(npyarr),radoptions.shape[0]))
    plan = np.zeros((len(npyarr),radoptions.shape[0]))
    spher = np.zeros((len(npyarr),radoptions.shape[0]))
    shannentr = np.zeros((len(npyarr),radoptions.shape[0]))
    
    for rad in range(len(radoptions)):
        radval = radoptions[rad]
        eig_valso3d, point_cloud = EigenValueCalculation_radius(npyarr,radval,kwargs)
        
        #Determination of linearlity/planarity/sphericity
        #First sort the eigenvalues large to small:
        eig_valso3d = np.sort(eig_valso3d, axis=1)[:, ::-1]
        linearity = (eig_valso3d[:,0]-eig_valso3d[:,1])/eig_valso3d[:,0]
        planarity = (eig_valso3d[:,1]-eig_valso3d[:,2])/eig_valso3d[:,0]
        sphericity = (eig_valso3d[:,2])/eig_valso3d[:,0]
        
        shannonEntropy = -linearity*np.log2(linearity)-planarity*np.log2(planarity)-sphericity*np.log2(sphericity)
        
        eigval1[:,rad] = eig_valso3d[:,0]
        eigval2[:,rad] = eig_valso3d[:,1]
        eigval3[:,rad] = eig_valso3d[:,2]
        lin[:,rad] = linearity
        plan[:,rad] = planarity
        spher[:,rad] = sphericity
        shannentr[:,rad] = shannonEntropy
    
    #Find the index of the minimal shannentr, setting nans to high values:
    minshannentr = np.nanargmin(shannentr,axis=1)
    finallin = lin[np.arange(len(lin)), minshannentr]
    finalplan = plan[np.arange(len(lin)), minshannentr]
    finalspher = spher[np.arange(len(lin)), minshannentr]
    
    
    
    # if kwargs['debug']:
    #     #Show this data as a point cloud for now
    #     dataToShow = npyarr
    #     minx = 0
    #     maxx = 200
    #     miny = 0
    #     maxy = 200
    #     sel = (dataToShow['x'] > minx) & (dataToShow['x'] < maxx) & (dataToShow['y'] > miny) & (dataToShow['y'] < maxy)
    #     dataToShow = dataToShow[sel]
    #     l_show = finallin[sel]
    #     p_show = finalplan[sel]
    #     s_show = finalspher[sel]
    #     #create figure
    #     fig = plt.figure()
    #     ax = fig.add_subplot(331, projection='3d')
    #     #Add polarity back to points
    #     ax.scatter(dataToShow['x'],dataToShow['y'],dataToShow['t'],c=l_show,cmap='plasma',alpha=0.5,s=1,vmin=0,vmax=1)
    #     ax.set_title('Linearity')
    #     ax = fig.add_subplot(332, projection='3d')
    #     #Add polarity back to points
    #     ax.scatter(dataToShow['x'],dataToShow['y'],dataToShow['t'],c=p_show,cmap='plasma',alpha=0.5,s=1,vmin=0,vmax=1)
    #     ax.set_title('Planarity')
    #     ax = fig.add_subplot(333, projection='3d')
    #     #Add polarity back to points
    #     ax.scatter(dataToShow['x'],dataToShow['y'],dataToShow['t'],c=s_show,cmap='plasma',alpha=0.5,s=1,vmin=0,vmax=1)
    #     ax.set_title('Sphericity')
        
    #     ax = fig.add_subplot(334, projection='3d')
    #     #Add polarity back to points
    #     sel = (l_show>p_show)&(l_show>s_show)
    #     ax.scatter(dataToShow[~sel]['x'],dataToShow[~sel]['y'],dataToShow[~sel]['t'],c='black',alpha=0.1,s=1)
    #     ax.scatter(dataToShow[sel]['x'],dataToShow[sel]['y'],dataToShow[sel]['t'],c='red',s=1)
    #     ax.set_title('Mostly-linear-points')
    #     ax = fig.add_subplot(335, projection='3d')
    #     #Add polarity back to points
    #     sel = (p_show>l_show)&(p_show>s_show)
    #     ax.scatter(dataToShow[~sel]['x'],dataToShow[~sel]['y'],dataToShow[~sel]['t'],c='black',alpha=0.1,s=1)
    #     ax.scatter(dataToShow[sel]['x'],dataToShow[sel]['y'],dataToShow[sel]['t'],c='red',s=1)
    #     ax.set_title('Mostly-planar-points')
    #     ax = fig.add_subplot(336, projection='3d')
    #     #Add polarity back to points
    #     sel = (s_show>p_show)&(s_show>l_show)
    #     ax.scatter(dataToShow[~sel]['x'],dataToShow[~sel]['y'],dataToShow[~sel]['t'],c='black',alpha=0.1,s=1)
    #     ax.scatter(dataToShow[sel]['x'],dataToShow[sel]['y'],dataToShow[sel]['t'],c='red',s=1)
    #     ax.set_title('Mostly-spherical-points')
        
    #     ax = fig.add_subplot(337, projection='3d')
    #     #Add polarity back to points
    #     sel = (full_data_labels>-1)
    #     ax.scatter(npyarr[~sel]['x'],npyarr[~sel]['y'],npyarr[~sel]['t'],c='black',alpha=0.1,s=1)
    #     ax.scatter(npyarr[sel]['x'],npyarr[sel]['y'],npyarr[sel]['t'],c='red',s=1)
    #     ax.set_title('DBSCANNED Mostly-spherical-points')
        
    #     plt.show()
    
    points = np.asarray(point_cloud.points)
    #Add polarity back to points
    points = np.concatenate((points, polarities.reshape(-1,1)), axis=1)
    
    
    points_mostly_spher = points[(finalspher>finallin)&(finalspher>finalplan)]
    sphere_indeces = np.where((finalspher>finallin)&(finalspher>finalplan))[0]
    
    #Run DBSCAN on the mostly-sphere points:
    dbscan = DBSCAN(eps=int(kwargs['DBSCAN_eps']), n_jobs=-1, min_samples=int(kwargs['DBSCAN_n_neighbours']))
    labels = dbscan.fit_predict(points_mostly_spher)
    full_data_labels=  np.ones(len(npyarr))*-1
    full_data_labels[sphere_indeces] = labels
    
    # Print how many clusters are found
    logging.info("Number of clusters found:"+ str(max(labels)+1))
    
    candidates = clusterPoints_to_candidates(points_mostly_spher,labels,ms_to_px)
        
    performance_metadata = f"SpectralClustering Finding ran for {time.time() - starttime} seconds."
    
    return candidates, performance_metadata

#This entire function (below) is basically a 'I'm figuring things out' and can be safely ignored.
def spectral_clustering_showcase(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    # [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    # data = np.zeros((len(npy_array),3))
    # data[:,0] = npy_array['x']
    # data[:,1] = npy_array['y']
    # data[:,2] = npy_array['t']
    
    
    npyarr=npy_array
    ms_to_px=35
    
    #only get data between 150-200 and 150-200 in 0, 1st column:
    data_for_o3d = npyarr[(npyarr['x'] > 100) & (npyarr['x'] < 250) & (npyarr['y'] > 100) & (npyarr['y'] < 250)]
    
    
    start_time = time.time()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
    print('point cloud created')
    point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    print('pcc estimated')
    pcc = np.asarray(point_cloud.covariances)
    eig_valso3d, eig_vecso3d = np.linalg.eig(pcc)
    print('eigv calculated')
    end_time = time.time()
    print('Eigenvalue calculation time1: ', end_time - start_time)
    start_time = time.time()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
    print('point cloud created')
    point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    print('pcc estimated')
    pcc = np.asarray(point_cloud.covariances)
    u,s,v = np.linalg.svd(pcc)
    print('eigv calculated')
    end_time = time.time()
    print('Eigenvalue calculation time2: ', end_time - start_time)
        
    maxeigenval = np.max(eig_valso3d,axis=1)
    mideigenval = np.median(eig_valso3d, axis=1)
    mineigenval = np.min(eig_valso3d,axis=1)
    sumeigenval = np.sum(eig_valso3d,axis=1)
    
    normeigenval = eig_valso3d/sumeigenval.reshape(-1, 1)
    stdeigenval =np.std(normeigenval,axis=1)
    
    highstdeigenval = normeigenval[stdeigenval>0.45]
    
    normeigenvalcutoff = 0.6
    maxeigenvalcutoff = 6
    
    clusterpoints = np.asarray(point_cloud.points)[(np.max(normeigenval, axis=1) < normeigenvalcutoff) & (maxeigenval < maxeigenvalcutoff), :]
    noisepoints = np.asarray(point_cloud.points)[(np.max(normeigenval, axis=1) >= normeigenvalcutoff) | (maxeigenval >= maxeigenvalcutoff), :]
    pcp = np.asarray(point_cloud.points)
    
    plt.figure(9)
    ax = plt.subplot(projection='3d')
    scatter  = ax.scatter(noisepoints[:,0], noisepoints[:,1],noisepoints[:,2], c='k',alpha=0.1,s=2)
    scatter  = ax.scatter(clusterpoints[:,0], clusterpoints[:,1],clusterpoints[:,2], c='r',alpha=0.1,s=2)
    plt.show()
    
    plt.figure(8)
    plt.hist(np.max(normeigenval,axis=1),bins=100)
    plt.xlabel('max(norm(Eigenvalues))')
    plt.ylabel('Frequency')
    plt.title('Histogram of max(norm(Eigenvalues))')
    plt.show()
    
    
    #Plot some histograms of covariances:
    plt.figure(10)
    #set projection of this subplot:
    ax = plt.subplot(3,3,1, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,0,0],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 0,0')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,2, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,0,1],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 0,1')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,3, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,0,2],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 0,2')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,4, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,1,0],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 1,0')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,5, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,1,1],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 1,1')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,6, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,1,2],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 1,2')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,7, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,2,0],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 2,0')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,8, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,2,1],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 2,1')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,9, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=pcc[:,2,2],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('cov 2,2')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    plt.show()
    
    
    #Plot some histograms of eigenvalues:
    plt.figure(11)
    ax = plt.subplot(3,3,1, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=eig_valso3d[:,0],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('First Eigenvalue')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,2, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=eig_valso3d[:,1],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('Second Eigenvalue')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,3, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=eig_valso3d[:,2],alpha=0.1,cmap='hsv',s=1)
    ax.set_title('Last Eigenvalue')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    #set projection of this subplot:
    ax = plt.subplot(3,3,4, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=maxeigenval,alpha=0.1,cmap='hsv',s=1)
    ax.set_title('Max Eigenvalue')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,5, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=mideigenval,alpha=0.1,cmap='hsv',s=1)
    ax.set_title('Mid Eigenvalue')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,6, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=mineigenval,alpha=0.1,cmap='hsv',s=1)
    ax.set_title('Min Eigenvalue')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,7, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=sumeigenval,alpha=0.1,cmap='hsv',s=1)
    ax.set_title('Sum Eigenvalue')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    ax = plt.subplot(3,3,8, projection='3d')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=stdeigenval,alpha=0.1,cmap='hsv',s=1)
    ax.set_title('Std Norm(Eigenvalue)')
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    
    plt.show()
    
    plt.figure(12)
    ax = plt.subplot(1,4,1)
    plt.hist(maxeigenval,bins=100)
    plt.xlabel('Maximum Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Histogram of Maximum Eigenvalues')
    ax = plt.subplot(1,4,2)
    plt.hist(mideigenval,bins=100)
    plt.xlabel('Mid Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mid Eigenvalues')
    ax = plt.subplot(1,4,3)
    plt.hist(mineigenval,bins=100)
    plt.xlabel('Min Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Histogram of Min Eigenvalues')
    ax = plt.subplot(1,4,4)
    plt.hist(stdeigenval,bins=100)
    plt.xlabel('Std Norm(Eigenvalue) Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Histogram of Std Norm(Eigenvalue) Eigenvalues')
    plt.show()
    
    #Display the point cloud points in a 3d scatter:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #plot only clx.labels > -1:
    pcp = np.asarray(point_cloud.points)
    maxeigenval = np.max(eig_valso3d,axis=1)
    # scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=np.max(eig_valso3d,axis=1),alpha=0.1,cmap='hsv')
    mevcutoff = 0.99
    # scatter  = ax.scatter(pcp[np.max(normeigenval,axis=1)>=mevcutoff,0], pcp[np.max(normeigenval,axis=1)>=mevcutoff,1],pcp[np.max(normeigenval,axis=1)>=mevcutoff,2], alpha=0.2,c='k',s=1)
    # scatter  = ax.scatter(pcp[np.max(normeigenval,axis=1)<mevcutoff,0], pcp[np.max(normeigenval,axis=1)<mevcutoff,1],pcp[np.max(normeigenval,axis=1)<mevcutoff,2], alpha=0.2,c='k',s=1)
    ax.scatter(clusterpoints[:,0],clusterpoints[:,1],clusterpoints[:,2],c='k',s=1)
    # scatter  = ax.scatter(pcp[stdeigenval<mevcutoff,0], pcp[stdeigenval<mevcutoff,1],pcp[stdeigenval<mevcutoff,2],alpha=0.1,c='r',s=1)
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    # ax.scatter(data_for_clx[clx.labels_>-1,0], data_for_clx[clx.labels_>-1,1],  data_for_clx[clx.labels_>-1,2], c=clx.labels_[clx.labels_>-1],cmap='Set1')
    plt.show()
        

        
    fig = plt.figure()
    # plt.hist(np.max(eig_valso3d,axis=1),bins=100) 
    plt.hist(np.max(normeigenval,axis=1),bins=100)
    plt.xlabel('Maximum Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Histogram of Maximum Eigenvalues')
    plt.show()
    
    
    #Display the point cloud points in a 3d scatter:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #plot only clx.labels > -1:
    pcp = np.asarray(point_cloud.points)
    # scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=np.max(eig_valso3d,axis=1),alpha=0.1,cmap='hsv')
    scatter  = ax.scatter(pcp[:,0], pcp[:,1],pcp[:,2], c=stdeigenval,alpha=0.1,cmap='hsv')
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
    mevcutoff = 3
    scatter  = ax.scatter(pcp[maxeigenval>=mevcutoff,0], pcp[maxeigenval>=mevcutoff,1],pcp[maxeigenval>=mevcutoff,2], alpha=0.05,c='k',s=1)
    scatter  = ax.scatter(pcp[maxeigenval<mevcutoff,0], pcp[maxeigenval<mevcutoff,1],pcp[maxeigenval<mevcutoff,2],alpha=0.1,c='r')
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
    mevcutoff = 0.99
    # scatter  = ax.scatter(pcp[np.max(normeigenval,axis=1)>=mevcutoff,0], pcp[np.max(normeigenval,axis=1)>=mevcutoff,1],pcp[np.max(normeigenval,axis=1)>=mevcutoff,2], alpha=0.2,c='k',s=1)
    # scatter  = ax.scatter(pcp[np.max(normeigenval,axis=1)<mevcutoff,0], pcp[np.max(normeigenval,axis=1)<mevcutoff,1],pcp[np.max(normeigenval,axis=1)<mevcutoff,2], alpha=0.2,c='k',s=1)
    ax.scatter(clusterpoints[:,0],clusterpoints[:,1],clusterpoints[:,2],c='k',s=1)
    # scatter  = ax.scatter(pcp[stdeigenval<mevcutoff,0], pcp[stdeigenval<mevcutoff,1],pcp[stdeigenval<mevcutoff,2],alpha=0.1,c='r',s=1)
    #show the colorbar:
    cbar = plt.colorbar(scatter)
    # ax.scatter(data_for_clx[clx.labels_>-1,0], data_for_clx[clx.labels_>-1,1],  data_for_clx[clx.labels_>-1,2], c=clx.labels_[clx.labels_>-1],cmap='Set1')
    plt.show()
        
    
    
    #Throw DBSCAN on the 'pre-clustered' data:
    dbscan = DBSCAN(eps=3, n_jobs=-1, min_samples=17)
    cluster_labels = dbscan.fit_predict(clusterpoints)
    
    # Print how many clusters are found
    logging.info("Number of clusters found:"+ str(max(cluster_labels)+1))
    
    
    clusterpoints_dbscan = clusterpoints[cluster_labels>-1]
    
    #Display the point cloud points in a 3d scatter:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(clusterpoints_dbscan[:,0],clusterpoints_dbscan[:,1],clusterpoints_dbscan[:,2],c=cluster_labels[cluster_labels>-1],s=1,cmap='tab20')
    plt.show()
    
    
    # #old version kept here, since I haven't 100% stress-tested new method, but seems to be fine
    # starttime = time.time()
    # candidates = {}
    # for cl in np.unique(cluster_labels):
    #     if cl > -1:
    #         clusterEvents = clusters[cluster_labels == cl]
    #         candidates[cl] = {}
    #         candidates[cl]['events'] = clusterEvents
    #         candidates[cl]['cluster_size'] = [np.max(clusterEvents['y'])-np.min(clusterEvents['y']), np.max(clusterEvents['x'])-np.min(clusterEvents['x']), np.max(clusterEvents['t'])-np.min(clusterEvents['t'])]
    #         candidates[cl]['N_events'] = len(clusterEvents)
    # endtime = time.time()
    # # Print the elapsed time:



        
    
    # # Perform Spectral Clustering
    # n_clusters = 3
    # cluster_labels = spectral_clustering2(data, n_clusters)

    # return candidates, performance_metadata
