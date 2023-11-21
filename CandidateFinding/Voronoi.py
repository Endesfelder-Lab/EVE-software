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
    
    npoints = 100000
    
    nparraydata = np.column_stack((npy_array[:npoints]['x'], npy_array[:npoints]['y'], npy_array[:npoints]['t']*0.0001))
    outputv = spatial.Voronoi(nparraydata)
    
    density = np.zeros(outputv.npoints)
    for i, region in enumerate(outputv.regions):
        if -1 not in region and len(region) > 0:
            density[i] = len(region)
        
    
    # Set a minimum density threshold
    min_density_threshold = 25

    # Get indices of regions that meet the minimum density criterion
    dense_regions_indices = np.where(density >= min_density_threshold)[0]

    # Get points in the densest regions
    densest_points = np.concatenate([outputv.regions[i] for i in dense_regions_indices if -1 not in outputv.regions[i]])

    # Remove duplicate points and points outside the Voronoi diagram
    densest_points = np.unique(densest_points)

    densest_points = outputv.points[dense_regions_indices]
        
    # # Extract Voronoi vertices
    # vor_vertices = outputv.vertices

    # Minimum number of points within a cluster
    min_points_per_cluster = 2

    # Use DBSCAN clustering
    dbscan = DBSCAN(min_samples=min_points_per_cluster)
    cluster_labels = dbscan.fit_predict(densest_points)
    
    nparraydataN = np.column_stack((densest_points, cluster_labels))
    nparraydataN = nparraydataN[cluster_labels >= 0]
    # clusters = cluster_labels[cluster_labels>=0]

    candidates = []
    performance_metadata = ''
    return candidates, performance_metadata
