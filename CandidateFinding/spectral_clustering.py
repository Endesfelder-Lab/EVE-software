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

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "spectral_clustering": {
            "required_kwargs": [
                {"name": "norm_eigenval_cutoff", "description": "Cutoff of normalized eigenvalues","default":0.6,"type":float},
                {"name": "max_eigenval_cutoff", "description": "Cutoff of maximum eigenvalue. Set to zero to auto-determine this!","default":0.0,"type":float},
                {"name": "search_n_neighbours", "description": "Number of (closest) neighbours for the covariance determination","default":50,"type":int},
                {"name": "ratio_ms_to_px", "description": "Ratio of milliseconds to pixels.","default":35.0,"type":float},
                {"name": "DBSCAN_eps", "description": "Eps of DBSCAN.","default":3,"type":int},
                {"name": "DBSCAN_n_neighbours", "description": "Minimum nr of points for DBSCAN cluster.","default":17,"type":int},
                
            ],
            "optional_kwargs": [
                {"name": "debug", "description": "Get some debug info.","default":False},
            ],
            "help_string": "Pseudo-spectral clustering."
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


def spectral_clustering(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    
    npyarr=npy_array
    polarities = npyarr['p']
    ms_to_px=float(kwargs['ratio_ms_to_px'])
    
    #only get data between 150-200 and 150-200 in 0, 1st column:
    data_for_o3d = npyarr
    multiCore = False
    if multiCore == True:
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
        
        print('point cloud created')
        point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(kwargs['search_n_neighbours'])))
        print('pcc estimated')
        pcc = np.asarray(point_cloud.covariances)
        point_cloud.estimate_normals(fast_normal_computation=True,search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(kwargs['search_n_neighbours'])))
        
        N = 150  # Number of chunks
        pcc_chunks = np.array_split(pcc, N)

        results = Parallel(n_jobs=-1)(delayed(compute_eig)(pcc_chunk) for pcc_chunk in pcc_chunks)
        
        eig_valso3d_list = []
        eig_valso3d_list.extend([res[0] for res in results])
        eig_valso3d = np.concatenate(eig_valso3d_list, axis=0)
        
        # eig_vecso3d_list = []
        # eig_vecso3d_list.extend([res[1] for res in results])    
        
        print('eigv calculated')
        end_time = time.time()
        print('Eigenvalue calculation time1: ', end_time - start_time)
    
    else:
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(zip(data_for_o3d['x'],data_for_o3d['y'],data_for_o3d['t']/(1000*ms_to_px)))
        print('point cloud created')
        point_cloud.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(kwargs['search_n_neighbours'])))
        print('pcc estimated')
        pcc = np.asarray(point_cloud.covariances)        
        eig_valso3d = np.linalg.svd(pcc,compute_uv=False,hermitian=True)
        print('eigv calculated')
        end_time = time.time()
        print('Eigenvalue calculation time2: ', end_time - start_time)

            
    maxeigenval = np.max(eig_valso3d,axis=1)
    mideigenval = np.median(eig_valso3d, axis=1)
    mineigenval = np.min(eig_valso3d,axis=1)
    sumeigenval = np.sum(eig_valso3d,axis=1)
    
    normeigenval = eig_valso3d/sumeigenval.reshape(-1, 1)
    stdeigenval =np.std(normeigenval,axis=1)
    
    # highstdeigenval = normeigenval[stdeigenval>0.45]
    
    normeigenvalcutoff = float(kwargs['norm_eigenval_cutoff']) #0.6
    maxeigenvalcutoff = float(kwargs['max_eigenval_cutoff']) #6
    
    #If set to zero, we do it computationally:
    if maxeigenvalcutoff == 0:
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
        params, _ = curve_fit(three_gaussians, bin_centers, hist, p0=initial_params)

        # Use scipy.optimize.root to find the x-position where the two Gaussians cross
        #Find the cross of the lowest with second-lowest gaussian:
        param_order = np.argsort(params[[1,4,7]])
        
        #Get the crossings:
        result1 = find_gaussian_cross(a1=params[param_order[0]*3],b1=params[param_order[0]*3+1],c1=params[param_order[0]*3+2],a2=params[param_order[1]*3],b2=params[param_order[1]*3+1],c2=params[param_order[1]*3+2],minp = 0,maxp=max(bin_centers),accuracy = 0.01)
        
        result2 = find_gaussian_cross(a1=params[param_order[0]*3],b1=params[param_order[0]*3+1],c1=params[param_order[0]*3+2],a2=params[param_order[2]*3],b2=params[param_order[2]*3+1],c2=params[param_order[2]*3+2],minp = 0,maxp=max(bin_centers),accuracy = 0.01)
        
        
        #Find the lowest crossing:
        lowest_crossing = min(min(result1), min(result2))
        maxeigenvalcutoff = lowest_crossing
        logging.info(f'Max eigenval determined to be {maxeigenvalcutoff}')
        
    if utilsHelper.strtobool(kwargs['debug']):
        try:
            #If debugging,w e plot this information:
            plt.hist(maxeigenval, bins=100, alpha=0.5, label='Histogram', density=True)
            plt.plot(bin_centers, three_gaussians(bin_centers, *params), 'r-', label='Three Gaussians')
            #Plot the individual gaussians:
            plt.plot(bin_centers, single_gaussian(bin_centers, *params[0:3]), 'r--')
            plt.plot(bin_centers, single_gaussian(bin_centers, *params[3:6]), 'r--')
            plt.plot(bin_centers, single_gaussian(bin_centers, *params[6:9]), 'r--')
            
            plt.plot(bin_centers, three_gaussians(bin_centers, *initial_params), 'k--', label='Three Gaussians Initial')
            plt.legend()
            plt.show()
        except:
            pass
        
    clusterpoints = np.asarray(point_cloud.points)[(np.max(normeigenval, axis=1) < normeigenvalcutoff) & (maxeigenval < maxeigenvalcutoff), :]
    noisepoints = np.asarray(point_cloud.points)[(np.max(normeigenval, axis=1) >= normeigenvalcutoff) | (maxeigenval >= maxeigenvalcutoff), :]

    logging.info("DBSCANning started")
    #Throw DBSCAN on the 'pre-clustered' data:
    dbscan = DBSCAN(eps=int(kwargs['DBSCAN_eps']), n_jobs=-1, min_samples=int(kwargs['DBSCAN_n_neighbours']))
    cluster_labels = dbscan.fit_predict(clusterpoints)
    
    # Print how many clusters are found
    logging.info("Number of clusters found:"+ str(max(cluster_labels)))
    
    # generate the candidates in from of a dictionary
    headers = ['x', 'y', 't']
    #time back to micros:
    clusterpoints[:,2] *=1000*ms_to_px
    #Integerise the data:
    new_arr = pd.DataFrame(clusterpoints, columns=headers)
    new_arr['x'] = new_arr['x'].astype(int)
    new_arr['y'] = new_arr['y'].astype(int)
    new_arr['t'] = new_arr['t'].astype(int)
    #add a 'p' header and fill with zeros, should be fixed later:
    new_arr['p'] = 0
    
    starttime = time.time()
    new_arr['Cluster_Labels'] = cluster_labels
    grouped_clusters = new_arr.groupby('Cluster_Labels')
    #This should be sped up: 
    candidates = {}
    for cl, cluster_events in grouped_clusters:
        if cl > -1:
            clusterEvents = cluster_events.drop(columns='Cluster_Labels')
            candidates[cl] = {}
            candidates[cl]['events'] = clusterEvents
            candidates[cl]['cluster_size'] = [int(np.max(clusterEvents['y'])-np.min(clusterEvents['y'])), int(np.max(clusterEvents['x'])-np.min(clusterEvents['x'])), int(np.max(clusterEvents['t'])-np.min(clusterEvents['t']))]
            candidates[cl]['N_events'] = int(len(clusterEvents))
    endtime = time.time()
    logging.info("candidates generated in: "+ str(endtime - starttime)+"s")
    
    performance_metadata = f"SpectralClustering Finding ran for {endtime - starttime} seconds."
    
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
    logging.info("Number of clusters found:"+ str(max(cluster_labels)))
    
    
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
