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
from shapely.geometry import Polygon
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "VoronoiFinding": {
            "required_kwargs": [
                {"name": "multiplier", "description": "Threshold for wavelet detection","default":1.25},
                {"name": "minDBSCANpoints", "description": "Threshold for wavelet detection","default":10},
                {"name": "ztimefactor", "description": "Threshold for wavelet detection","default":10000},
                {"name": "DBebs", "description": "Threshold for wavelet detection","default":5},
                {"name": "denslookup", "description": "Threshold for wavelet detection","default":3},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Returns a test dictionary and metadata string, to check if the program is working properly.",
            "display_name": "Voronoi WIP"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def VoronoiFinding(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    
    ztimefactor = 1/float(kwargs['ztimefactor'])
    
    
    # densest_points_within_range_full = []
    # densityinfo_full = []
    # vorpoints = {}
    # densityinfo={}
    # nrdatapoints = 5000
    # nrits = len(npy_array)//nrdatapoints+1
    # for i in range(0,nrits):
    #     print('Iteration: ' + str(i))
    #     #Conceptually, i'm going to divide a let's say 10000 datapoint set in 10x 1000 points. For each, I do -200:1200 and get the voronoi infomation. Then I calculate the density, but also filter on the fact that they need to be in 0:1000 (the -200, +200 is only for correct voronoiíng). Then we merge the densities and tadaa    
    #     densityinfo[i],vorpoints[i] = getVoronoiSplitFunctionWhatever(fullData = npy_array, startindex = i*nrdatapoints, endindex = (i+1)*nrdatapoints, paddingindex = 200, ztimefactor=ztimefactor)
    #     densityinfo_full.extend(densityinfo[i])
    
    # density_threshold = np.mean(densityinfo_full)*float(kwargs['multiplier'])
    
    # #Now get 'hardcoded' density and filter on this
    # for i in range(0,nrits):
    #     print('Iteration2: ' + str(i))
    #     densest_points_within_range = getDensePointsFromVoro(vorpoints[i],densityinfo[i],density_threshold)
    #     densest_points_within_range_full.extend(densest_points_within_range)
    
    #Hm trying this computeDensity thing
    logging.info('Compute density thing started!')
    
    import scipy.spatial as spatial
    nparrfordens = np.column_stack((npy_array['x'], npy_array['y'], npy_array['t']*ztimefactor))
    # Time the first line
    # start_time_fasttree = time.time()
    # fasttree = KDTree(nparrfordens,leafsize=16)
    # freq = computeDensitypykd(fasttree,float(kwargs['denslookup']),nparrfordens)
    # end_time_fasttree = time.time()
    
    # elapsed_time_fasttree = end_time_fasttree - start_time_fasttree

    # Time the second line
    start_time_tree = time.time()
    tree = spatial.KDTree(np.array(nparrfordens))
    freq = computeDensity(tree,float(kwargs['denslookup']))
    end_time_tree = time.time()
    elapsed_time_tree = end_time_tree - start_time_tree
    # print("Elapsed time for fasttree:", elapsed_time_fasttree)
    print("Elapsed time for tree:", elapsed_time_tree)
    
    # # Time the second line
    # start_time_tree = time.time()
    # tree = spatial.KDTree(np.array(nparrfordens),compact_nodes=True,balanced_tree=False)
    # freq2 = computeDensity(tree,float(kwargs['denslookup']))
    # end_time_tree = time.time()
    # elapsed_time_tree = end_time_tree - start_time_tree
    # # print("Elapsed time for fasttree:", elapsed_time_fasttree)
    # print("Elapsed time for ctree:", elapsed_time_tree)
    
    
    freq_threshold = np.mean(freq)*float(kwargs['multiplier'])
    freq_points_within_range = freq>=freq_threshold
    densest_points_within_range_full = nparrfordens[freq_points_within_range,:]
    print('done')
    
    # Minimum number of points within a cluster
    min_points_per_cluster = int(kwargs['minDBSCANpoints'])

    # Use DBSCAN clustering
    dbscan = DBSCAN(eps=float(kwargs['DBebs']), n_jobs=-1, min_samples=min_points_per_cluster)
    cluster_labels = dbscan.fit_predict(densest_points_within_range_full)
    
    #Print how many clusters are found
    logging.info("Number of clusters found:"+ str(max(cluster_labels)))

    # generate the candidates in from of a dictionary
    headers = ['x', 'y', 't']
    densest_points_within_range_full_pd = pd.DataFrame(densest_points_within_range_full, columns=headers)
    
    #Also add a ['p'] and give them all value 1:
    densest_points_within_range_full_pd.loc[:,'p'] = 1
    candidates = {}
    for cl in np.unique(cluster_labels):
        if cl > -1:
            clusterEvents = densest_points_within_range_full_pd[cluster_labels == cl]

            #Correct for Z time actually here:
            clusterEvents.loc[:,'t'] = clusterEvents['t']/ztimefactor
        
            candidates[cl] = {}
            candidates[cl]['events'] = clusterEvents
            candidates[cl]['cluster_size'] = [np.max(clusterEvents['y'])-np.min(clusterEvents['y']), np.max(clusterEvents['x'])-np.min(clusterEvents['x']), np.max(clusterEvents['t'])-np.min(clusterEvents['t'])]
            candidates[cl]['N_events'] = len(clusterEvents)

    logging.info('Finished whatever this density finding is')
    
    performance_metadata = ''
    return candidates, performance_metadata

#Conceptually, i'm going to divide a let's say 10000 datapoint set in 10x 1000 points. For each, I do -200:1200 and get the voronoi infomation. Then I calculate the density, but also filter on the fact that they need to be in 0:1000 (the -200, +200 is only for correct voronoiíng). Then we merge the densities and tadaa    
def getVoronoiSplitFunctionWhatever(fullData, startindex = 0, endindex = 1000, paddingindex = 200, ztimefactor=1/1000):
    #Start i is startindex-paddingindex, but bound by and size of fullData:
    
    starti = min(max(0,startindex-paddingindex),len(fullData))
    endi = min(max(0,endindex+paddingindex),len(fullData))
    
    nparraydata = np.column_stack((fullData[starti:endi]['x'], fullData[starti:endi]['y'], fullData[starti:endi]['t']*ztimefactor))
    
    vor = spatial.Voronoi(nparraydata)
    density = np.zeros(vor.npoints+1)
    for i, region in enumerate(vor.regions):
        # if i < len(region):
        if -1 not in region and len(region) > 0:
            density[i] = len(region)
    
    #Remove all points that are in the first or last paddingindex range:
    if starti == 0:
        density = density[:-paddingindex]
    elif endi == len(fullData):
        density = density[paddingindex:]
    else:
        density = density[paddingindex:-paddingindex]
    
    return density, vor
    

def getDensePointsFromVoro(vor,vordensity,densevalue):
    
    # Set a minimum density threshold
    min_density_threshold = densevalue

    # Get indices of regions that meet the minimum density criterion
    dense_regions_indices = np.where(vordensity >= min_density_threshold)[0]

    # Get points in the densest regions
    densest_points = np.concatenate([vor.regions[i] for i in dense_regions_indices if -1 not in vor.regions[i]])

    # densest_vors = np.concatenate([vor[i] for i in dense_regions_indices if -1 not in vor.regions[i]])
#
    # Remove duplicate points and points outside the Voronoi diagram
    densest_points = np.unique(densest_points)

    densest_points = vor.points[dense_regions_indices,:]
    
    return densest_points

def computeDensity(tree,radius):
    neighbors = tree.query_ball_tree(tree,radius,eps=radius/2)
    frequency = np.array([len(i) for i in neighbors], dtype=np.float64)
    return frequency

def computeDensitypykd(tree,radius,nparr):
    maxk = 100
    neighbors = tree.query(nparr,k=maxk,distance_upper_bound=radius)
    #Time this:
    # Start timing
    # start_time = time.time()
    # frequency = np.array([np.where(arr == len(nparr))[0][0] if len(nparr) in arr else 10 for arr in neighbors[1]])
    # # End timing
    # end_time = time.time()
    # # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print("Elapsed time 1:", elapsed_time)
    
    # # Start timing
    # start_time = time.time()
    # index_of_inf = np.array([np.where(np.isinf(arr))[0][0] if np.any(np.isinf(arr)) else -1 for arr in neighbors[0]])
    
    frequency = np.argmax(np.isinf(neighbors[0]), axis=1)
    frequency[frequency==0] = maxk
    # index_of_inf[np.logical_not(np.any(np.isinf(neighbors), axis=1))] = -1
    
    # End timing
    # end_time = time.time()
    # # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print("Elapsed time 2:", elapsed_time)
    # frequency = np.array([np.where(arr == len(nparr))[0][0] for arr in neighbors[1]])
    return frequency