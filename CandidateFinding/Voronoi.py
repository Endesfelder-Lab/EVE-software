import inspect
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
    
    ztimefactor = 0.0001
    
    
    densest_points_within_range_full = []
    densityinfo_full = []
    vorpoints = {}
    densityinfo={}
    nrdatapoints = 5000
    nrits = len(npy_array)//nrdatapoints+1
    for i in range(0,nrits):
        print('Iteration: ' + str(i))
        #Conceptually, i'm going to divide a let's say 10000 datapoint set in 10x 1000 points. For each, I do -200:1200 and get the voronoi infomation. Then I calculate the density, but also filter on the fact that they need to be in 0:1000 (the -200, +200 is only for correct voronoiíng). Then we merge the densities and tadaa    
        densityinfo[i],vorpoints[i] = getVoronoiSplitFunctionWhatever(fullData = npy_array, startindex = i*nrdatapoints, endindex = (i+1)*nrdatapoints, paddingindex = 200, ztimefactor=ztimefactor)
        densityinfo_full.extend(densityinfo[i])
    
    density_threshold = np.mean(densityinfo_full)*1.25
    
    #Now get 'hardcoded' density and filter on this
    for i in range(0,nrits):
        print('Iteration2: ' + str(i))
        densest_points_within_range = getDensePointsFromVoro(vorpoints[i],densityinfo[i],density_threshold)
        densest_points_within_range_full.extend(densest_points_within_range)
    
    # # # Extract Voronoi vertices
    # # vor_vertices = outputv.vertices

    # Minimum number of points within a cluster
    min_points_per_cluster = 10

    # Use DBSCAN clustering
    dbscan = DBSCAN(eps=2, n_jobs=-1, min_samples=min_points_per_cluster)
    cluster_labels = dbscan.fit_predict(densest_points_within_range_full)
    
    nparraydataN = [densest_points_within_range_full[i] for i in range(len(cluster_labels)) if cluster_labels[i] == 1]
    
    
    # generate the candidates in from of a dictionary
    candidates = {}
    for cl in np.unique(cluster_labels):
        if cl > -1:
            clusterEvents = [densest_points_within_range_full[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cl]

            headers = ['x', 'y', 't']
            clusterEvents = pd.DataFrame(clusterEvents, columns=headers)
            
            #Correct for Z time actually here:
            clusterEvents['t'] = clusterEvents['t']/ztimefactor
            
            #Also add a ['p'] and give them all value 1:
            clusterEvents['p'] = 1
        
            candidates[cl] = {}
            candidates[cl]['events'] = clusterEvents
            candidates[cl]['cluster_size'] = [np.max(clusterEvents['y'])-np.min(clusterEvents['y']), np.max(clusterEvents['x'])-np.min(clusterEvents['x']), np.max(clusterEvents['t'])-np.min(clusterEvents['t'])]
            candidates[cl]['N_events'] = len(clusterEvents)
    
    
    # # compare this to a full voronoi fo the full 10k:
    # VorDensity,vorS = getVoronoiSplitFunctionWhatever(fullData = npy_array, startindex = 0, endindex = 10000, paddingindex = 0)
    # density_thresholdS = np.mean(VorDensity)*1.0
    # densest_points_within_range_full_second = getDensePointsFromVoro(vorS,VorDensity,density_thresholdS)
    # # clusters = cluster_labels[cluster_labels>=0]
    # dbscan = DBSCAN(eps=5, n_jobs=-1, min_samples=min_points_per_cluster)
    # cluster_labels_second = dbscan.fit_predict(densest_points_within_range_full_second)
    
    # nparraydataN_second = [densest_points_within_range_full_second[i] for i in range(len(cluster_labels_second)) if cluster_labels_second[i] == 2]
    

    performance_metadata = ''
    return candidates, performance_metadata

#Conceptually, i'm going to divide a let's say 10000 datapoint set in 10x 1000 points. For each, I do -200:1200 and get the voronoi infomation. Then I calculate the density, but also filter on the fact that they need to be in 0:1000 (the -200, +200 is only for correct voronoiíng). Then we merge the densities and tadaa    
def getVoronoiSplitFunctionWhatever(fullData, startindex = 0, endindex = 1000, paddingindex = 200, ztimefactor=1/1000):
    #Start i is startindex-paddingindex, but bound by and size of fullData:
    
    starti = min(max(0,startindex-paddingindex),len(fullData))
    endi = min(max(0,endindex+paddingindex),len(fullData))
    
    nparraydata = np.column_stack((fullData[starti:endi]['x'], fullData[starti:endi]['y'], fullData[starti:endi]['t']*ztimefactor))
    
    vor = spatial.Voronoi(nparraydata)
    density = np.zeros(vor.npoints)
    for i, region in enumerate(vor.regions):
        if i < len(region):
            if -1 not in region and len(region) > 0:
                density[i] = len(region)
    
    #Remove all points that are in the first or last paddingindex range:
    if paddingindex>0:
        density = density[paddingindex:-paddingindex]
    
    return density, vor
    

def getDensePointsFromVoro(vor,vordensity,densevalue):
    
    # Set a minimum density threshold
    min_density_threshold = densevalue

    # Get indices of regions that meet the minimum density criterion
    dense_regions_indices = np.where(vordensity >= min_density_threshold)[0]

    # Get points in the densest regions
    densest_points = np.concatenate([vor.regions[i] for i in dense_regions_indices if -1 not in vor.regions[i]])

    # Remove duplicate points and points outside the Voronoi diagram
    densest_points = np.unique(densest_points)

    densest_points = vor.points[dense_regions_indices]
    
    return densest_points