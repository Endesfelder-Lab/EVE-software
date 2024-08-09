import inspect
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
    from Utils import utilsHelper
import pandas as pd
import numpy as np
import time

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "PseudoScatter": {
            "required_kwargs": [
                {"name": "ZoomValue", "description": "Pixel-to-pseudopixel ratio","Default":10},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Returns a test dictionary and metadata string, to check if the program is working properly."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def PseudoScatter(resultArray,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    # Start the timer
    start_time = time.time()
    
    zoomvalue=float(kwargs['ZoomValue'])
    
    #Idea: create an empty array with the right size, i.e. ZoomValue times bigger than the maximum size of the results.
    #Then simply increase the value of the pixels in that array based on resultArray
    
    #Get the min/max bounds in pixel units:
    xoffset = np.min(resultArray['x']) / settings['PixelSize_nm']['value']
    maxx = np.max(resultArray['x']) / settings['PixelSize_nm']['value']
    yoffset = np.min(resultArray['y']) / settings['PixelSize_nm']['value']
    maxy = np.max(resultArray['y']) / settings['PixelSize_nm']['value']
    #Scale them so that minx, miny = 0:
    minx = 0
    miny = 0
    maxx = maxx - xoffset
    maxy = maxy - yoffset
    
    #Create the upscaled image:
    upScaledImage = np.zeros((int(maxx*zoomvalue),int(maxy*zoomvalue)))
    #Loop through the results:
    for i in range(len(resultArray)):
        #Get the pixel coordinates:
        #Check if it's in the list (when filtered out):
        if i in resultArray['id'].values:
            #Check if it's not a nan (when fitting failed)
            if not np.isnan(resultArray['x'][i]):
                if not np.isnan(resultArray['y'][i]):
                    x,y = int(np.floor((resultArray['x'][i] / settings['PixelSize_nm']['value'] - xoffset)*zoomvalue)), int(np.floor((resultArray['y'][i] / settings['PixelSize_nm']['value'] - yoffset)*zoomvalue))
                    if x>=0 and y>=0 and x<upScaledImage.shape[0] and y<upScaledImage.shape[1]:
                        upScaledImage[x,y] += 1
    
    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    performance_metadata = f"Dummy function ran for {elapsed_time} seconds."
    print('Function one ran!')

    return upScaledImage.T, performance_metadata
