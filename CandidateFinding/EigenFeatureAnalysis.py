import inspect
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
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
                {"name": "search_n_neighbours","display_text":"Number of neighbours", "description": "Number of (closest) neighbouring events for the covariance determination","default":50,"type":int},
                {"name": "max_eigenval_cutoff","display_text":"Maximum Eigenvalue cutoff", "description": "Cutoff of maximum eigenvalue of events. Set to zero to auto-determine this-  and use Debug to visualise!","default":0.0,"type":float},
                {"name": "linearity_cutoff","display_text":"Linearity cutoff", "description": "Linearity (0-1) cutoff","default":0.7,"type":float},
                {"name": "ratio_ms_to_px","display_text":"Ratio ms to px", "description": "Ratio of milliseconds to pixels.","default":20.0,"type":float},
                {"name": "DBSCAN_eps","display_text":"DBSCAN epsilon", "description": "Eps of DBSCAN.","default":4,"type":int},
                {"name": "DBSCAN_n_neighbours","display_text":"DBSCAN nr. neighbours", "description": "Minimum nr of points for DBSCAN cluster.","default":20,"type":int},
                
            ],
            "optional_kwargs": [
                {"name": "debug", "display_text":"Debug Boolean", "description": "Get some debug info.","default":False},
            ],
            "help_string": "Eigen-feature analysis. Performs spectral clustering methodology on the data to separate SMLM signal from noise. Practically finds clusters based on the Eigenvalues of the covariance matrix obtained by looking at the [Number of neighbours] closest points. Only events with an Eigenvalue lower than [Maximum Eigenvalue cutoff] are used. Also selects against linearity via [Linearity cutoff]",
            "display_name": "Eigen-feature analysis"
        },
        # Not sure if this method works good (August 2024), removing for now.
        # "eigen_feature_analysis_autoRadiusSelect": {
        #     "required_kwargs": [
        #         {"name": "max_eigenval_cutoff","display_text":"Maximum Eigenvalue cutoff", "description": "Cutoff of maximum eigenvalue of events. Set to zero to auto-determine this-  and use Debug to visualise!","default":0.0,"type":float},
        #         {"name": "linearity_cutoff","display_text":"Linearity cutoff", "description": "Linearity (0-1) cutoff","default":0.7,"type":float},
        #         {"name": "min_radius","display_text":"Minimum radius", "description": "Minimal radius","default":4,"type":float},
        #         {"name": "max_radius","display_text":"Maximum radius", "description": "Maximum radius","default":10,"type":float},
        #         {"name": "ratio_ms_to_px","display_text":"Ratio ms to px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float},
        #         {"name": "DBSCAN_eps","display_text":"DBSCAN epsilon", "description": "Eps of DBSCAN.","default":3,"type":int},
        #         {"name": "DBSCAN_n_neighbours","display_text":"DBSCAN nr. neighbours", "description": "Minimum nr of points for DBSCAN cluster.","default":17,"type":int},
        #     ],
        #     "optional_kwargs": [
        #         {"name": "debug", "display_text":"Debug Boolean", "description": "Get some debug info.","default":False},
        #     ],
        #     "help_string": "Eigen-feature analysis. Performs spectral clustering methodology on the data to separate SMLM signal from noise. Practically finds clusters based on the Eigenvalues of the covariance matrix obtained by looking neighbours in a radius. It tries to auto-select the best radius (highest contrast) via [Minimum radius] and [Maximum radius]. Only events with an Eigenvalue lower than [Maximum Eigenvalue cutoff] are used. Also selects against linearity via [Linearity cutoff]. Experimental method.",
        #     "display_name": "Eigen-feature analysis, automatic radius finding"
        # },
        "eigenFeature_analysis_and_bbox_finding": {
            "required_kwargs": [
                {"name": "search_n_neighbours","display_text":"Number of neighbours", "description": "Number of (closest) neighbouring events for the covariance determination","default":50,"type":int},
                {"name": "max_eigenval_cutoff","display_text":"Maximum Eigenvalue cutoff", "description": "Cutoff of maximum eigenvalue of events. Set to zero to auto-determine this-  and use Debug to visualise!","default":0.0,"type":float},
                {"name": "linearity_cutoff","display_text":"Linearity cutoff", "description": "Linearity (0-1) cutoff","default":0.7,"type":float},
                {"name": "ratio_ms_to_px","display_text":"Ratio ms to px", "description": "Ratio of milliseconds to pixels.","default":20.0,"type":float},
                {"name": "DBSCAN_eps","display_text":"DBSCAN epsilon", "description": "Eps of DBSCAN.","default":3,"type":int},
                {"name": "DBSCAN_n_neighbours","display_text":"DBSCAN nr. neighbours", "description": "Minimum nr of points for DBSCAN cluster.","default":20,"type":int},
                {"name": "bbox_padding","display_text":"Bounding-box padding", "description": "Padding of the bounding box (in px equivalents).","default":1,"type":int},
            ],
            "optional_kwargs": [
                {"name": "debug", "display_text":"Debug Boolean", "description": "Get some debug info.","default":False},
            ],
            "help_string": "Eigen-feature analysis. Performs spectral clustering methodology on the data to separate SMLM signal from noise. Practically finds clusters based on the Eigenvalues of the covariance matrix obtained by looking at the [Number of neighbours] closest points. Only events with an Eigenvalue lower than [Maximum Eigenvalue cutoff] are used. Also selects against linearity via [Linearity cutoff].Ends with adding a boundingbox around the cluster with some padding given by [Bounding-box padding].",
            "display_name": "Eigen-feature analysis with bbox finding"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def compute_eig(pcc):
    """
    Compute the eigenvalue - on a separate function for parallellization
    """ 
    
    return np.linalg.eig(pcc)

def EigenValueCalculation(npyarr,kwargs):
    """
    Calculates the eigenvalues of the covariance matrix of the points. Returns the eigenvalues and the point cloud with the N neighbouring events.
    """
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
    """
    Calculates the eigenvalues of the covariance matrix of the points - using a radius rather than N number of events. Returns the eigenvalues and the point cloud.
    """
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
    """
    Algorithmically (attempt to) find the best division of the eigenvalues into background and signal.
    """
    
    #Fit this histogram with three gaussians:
    #create a normalized histogram:
    hist, bin_edges = np.histogram(maxeigenval, bins=100, density=True)
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find the peak (cwt)
    try:
        from scipy import signal
        nbins=100
        hist, bin_edges = np.histogram(maxeigenval, np.linspace(0,max(maxeigenval),nbins))
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
            hist, bin_edges = np.histogram(maxeigenval, np.linspace(0,max(maxeigenval),nbins)) 
            peakind = signal.find_peaks_cwt(max(hist)-hist, np.arange(1,nbins/4))
            print(bin_edges[peakind]+bin_edges[1])
            
        except:
            pass
        
    return maxeigenvalcutoff

def clusterPoints_to_candidates(clusterpoints,cluster_labels,ms_to_px):
    """Generate the candidates in from of a dictionary"""
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
            # candidates[cl]['cluster_size'] = [int((clusterEvents['y']).max()-(clusterEvents['y']).min()+1), int((clusterEvents['x']).max()-(clusterEvents['x']).min()+1), int((clusterEvents['t']).max()-(clusterEvents['t']).min())]
            
            #We know that they're time-sorted, so for time, we can do end-start, which is max-min. Speeds up a little
            candidates[cl]['cluster_size'] = [int((clusterEvents['y']).max()-(clusterEvents['y']).min()+1), int((clusterEvents['x']).max()-(clusterEvents['x']).min()+1), int(clusterEvents['t'].iloc[-1]-clusterEvents['t'].iloc[0])]
            
    endtime = time.time()
    
    return candidates

def get_cluster_bounding_boxes(events, cluster_labels,padding_xy=0,padding_t=0):
    """
    Öbtain the bounding boxes for each cluster with an additionall padding in XY and T if so wanted.
    """
    
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
    """
    Child function to compute the bounding box for a given cluster.
    """
    cluster_points = events[cluster_labels == cluster_id]
    x_coordinates = cluster_points[:,0].tolist()
    y_coordinates = cluster_points[:,1].tolist()
    t_coordinates = cluster_points[:,2].tolist()
    return [min(x_coordinates)-padding_xy, max(x_coordinates)+padding_xy, min(y_coordinates)-padding_xy, max(y_coordinates)+padding_xy, min(t_coordinates)-padding_t, max(t_coordinates)+padding_t]


def get_events_in_bbox_NE(npyarr,bboxes,ms_to_px):
    """
    From all events, get the events in certain bounding boxes. Very speed-optimized.
    """
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
    """
    DBSCAN clustering with try/exception because it very very very occasionally fails. Haven't seen it in a while.
    """
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

def showDebugInfo(maxeigenval,maxeigenvalcutoff):
    """
    Subfunction to show the distribution of the maximum eigenvalue. (DEBUG)
    """
    nbins=100
    fig, ax1 = plt.subplots()

    # Plot the histogram on the first y-axis
    counts, bin_edges = np.histogram(maxeigenval, np.linspace(0,max(maxeigenval),nbins), density=True)
    # Create color array
    colors = ['tab:red' if edge < maxeigenvalcutoff else 'black' for edge in bin_edges[:-1]]
    # Plot the histogram with color-coding
    ax1.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), alpha=0.5, color=colors, align='edge')
    
    #Add a vertical line at maxeigenvalcutoff:
    ax1.axvline(x=maxeigenvalcutoff, color='tab:red', linestyle='dashed', linewidth=1, label='Max eigenvalue cutoff')
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Maximum eigenvalue')
    legend = ax1.legend(loc='upper right')
    
    #Add a title:
    ax1.set_title('Histogram of maximum eigenvalues')

    # Display the plot
    plt.show()

def eigenFeature_analysis(npy_array,settings,**kwargs):
    """
    General EigenFeature analysis function.
    """
    
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
    
    #Visualise the histogram
    if kwargs['debug'] == 'True':
        showDebugInfo(maxeigenval,maxeigenvalcutoff)

        # Display the plot
        plt.show()
    
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
    
    time0 = time.time()
    #Remove small/large bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)
    time1 = time.time()
    
    #Remove xytOutliers-  removed in pre-release, probably a bad idea
    # candidates = utilsHelper.removeCandidates_xytoutliers(candidates,settings)
    time2 = time.time()
    
    
    performance_metadata = f"Eigen-Feature Finding ran for {time.time() - starttime} seconds."
    
    return candidates, performance_metadata

def eigenFeature_analysis_and_bbox_finding(npy_array,settings,**kwargs):
    """
    General EigenFeature-with-bounding-box analysis function.
    """
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
    
    #Visualise the histogram
    if kwargs['debug'] == 'True':
        showDebugInfo(maxeigenval,maxeigenvalcutoff)
        
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
    
    performance_metadata = f"Eigen-Feature w/ BBOX Finding ran for {time.time() - starttime} seconds."
    
    return candidates, performance_metadata
    
def eigen_feature_analysis_autoRadiusSelect(npy_array,settings,**kwargs):
    """
    General EigenFeature-automatic-radius-select analysis function. Basically tries to minimize Shannon entropy given linearity/planarity/sphericity -- assumption is that a perfect cluster is 100% spherical, so low entropy.
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

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
        
        #Core idea: calculate the Shannon entropy of the linearlity/planarity/sphericity distribution. Since ideally, the radius is chosen so that sphericity is high and other 2 are low, the ideal radius has a low entropy.
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
    
    if kwargs['debug'] == 'True':
        #Show this data as a point cloud for now
        dataToShow = npyarr
        minx = 0
        maxx = 200
        miny = 0
        maxy = 200
        sel = (dataToShow['x'] > minx) & (dataToShow['x'] < maxx) & (dataToShow['y'] > miny) & (dataToShow['y'] < maxy)
        dataToShow = dataToShow[sel]
        l_show = finallin[sel]
        p_show = finalplan[sel]
        s_show = finalspher[sel]
        #create figure
        fig = plt.figure()
        ax = fig.add_subplot(331, projection='3d')
        #Add polarity back to points
        ax.scatter(dataToShow['x'],dataToShow['y'],dataToShow['t'],c=l_show,cmap='plasma',alpha=0.5,s=1,vmin=0,vmax=1)
        ax.set_title('Linearity')
        ax = fig.add_subplot(332, projection='3d')
        #Add polarity back to points
        ax.scatter(dataToShow['x'],dataToShow['y'],dataToShow['t'],c=p_show,cmap='plasma',alpha=0.5,s=1,vmin=0,vmax=1)
        ax.set_title('Planarity')
        ax = fig.add_subplot(333, projection='3d')
        #Add polarity back to points
        ax.scatter(dataToShow['x'],dataToShow['y'],dataToShow['t'],c=s_show,cmap='plasma',alpha=0.5,s=1,vmin=0,vmax=1)
        ax.set_title('Sphericity')
        
        ax = fig.add_subplot(334, projection='3d')
        #Add polarity back to points
        sel = (l_show>p_show)&(l_show>s_show)
        ax.scatter(dataToShow[~sel]['x'],dataToShow[~sel]['y'],dataToShow[~sel]['t'],c='black',alpha=0.1,s=1)
        ax.scatter(dataToShow[sel]['x'],dataToShow[sel]['y'],dataToShow[sel]['t'],c='red',s=1)
        ax.set_title('Mostly-linear-points')
        ax = fig.add_subplot(335, projection='3d')
        #Add polarity back to points
        sel = (p_show>l_show)&(p_show>s_show)
        ax.scatter(dataToShow[~sel]['x'],dataToShow[~sel]['y'],dataToShow[~sel]['t'],c='black',alpha=0.1,s=1)
        ax.scatter(dataToShow[sel]['x'],dataToShow[sel]['y'],dataToShow[sel]['t'],c='red',s=1)
        ax.set_title('Mostly-planar-points')
        ax = fig.add_subplot(336, projection='3d')
        #Add polarity back to points
        sel = (s_show>p_show)&(s_show>l_show)
        ax.scatter(dataToShow[~sel]['x'],dataToShow[~sel]['y'],dataToShow[~sel]['t'],c='black',alpha=0.1,s=1)
        ax.scatter(dataToShow[sel]['x'],dataToShow[sel]['y'],dataToShow[sel]['t'],c='red',s=1)
        ax.set_title('Mostly-spherical-points')
        
        ax = fig.add_subplot(337, projection='3d')
        #Add polarity back to points
        sel = (full_data_labels>-1)
        ax.scatter(npyarr[~sel]['x'],npyarr[~sel]['y'],npyarr[~sel]['t'],c='black',alpha=0.1,s=1)
        ax.scatter(npyarr[sel]['x'],npyarr[sel]['y'],npyarr[sel]['t'],c='red',s=1)
        ax.set_title('DBSCANNED Mostly-spherical-points')
        
        plt.show()
    
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
        
    #Remove small/large bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)

    performance_metadata = f"Eigen-Feature auto-radius Finding ran for {time.time() - starttime} seconds."
    
    return candidates, performance_metadata