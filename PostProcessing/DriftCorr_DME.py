import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time

# from .dme import *

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "DriftCorr_entropyMin": {
            "required_kwargs": [
                {"name": "frame_time_for_dme", "description": "Frame-time used for drift-correction (in ms)","default":100.,"type":float,"display_text":"Frame time used in DME"},
                {"name": "frames_per_bin", "description": "Number of frames in every bin for dme drift correction ","default":50,"type":int,"display_text":"Frames per bin"},
                {"name": "use_cuda", "description": "Use CUDA-GPU rather than CPU.","default":False,"display_text":"Use CUDA"},
                {"name": "visualisation", "description": "Visualisation of the drift traces.","default":True,"display_text":"Visualisation"},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Drift correction from Cnossen et al.."
        }
    }





# from .dme import *
#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def DriftCorr_entropyMin(resultArray,findingResult,settings,**kwargs):
    """ 
    Implementation of DME drift correction based on Cnossen et al. 2021 (https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245). 
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    #Import the correct package
    from .dme.dme import dme
    
    #Set user variables
    frame_time_for_dme = float(kwargs['frame_time_for_dme']) #in ms
    framesperbinv = int(kwargs['frames_per_bin'])#in 'frames'
    use_cuda= utilsHelper.strtobool(kwargs['use_cuda'])
    visualisation=utilsHelper.strtobool(kwargs['visualisation'])
    
    #Hard-coded variables
    fov_width = 200 #in pixels

    #Obtain the localizations from the resultArray
    resultArray=resultArray.dropna()
    locs_for_dme = np.column_stack((resultArray['x'].values-min(resultArray['x']),resultArray['y'].values-min(resultArray['y'])))
    #Convert it to pixel-units
    locs_for_dme/=(float(settings['PixelSize_nm']['value']))
    
    #Get the 'frame' for each localization based on user-defined frame_time_for_dme
    framenumFull = np.floor(np.array(resultArray['t'].values)/(frame_time_for_dme))
    #Pop -1 entries:
    locs_for_dme = locs_for_dme[framenumFull != -1]
    framenum = framenumFull[framenumFull != -1]
    framenum -= min(framenum)
    framenum = framenum.astype(int)
    
    
    
    #to prevent unexpected errors: remove the bottom and top 0.1 percentile:
    bottom_percentile = np.percentile(framenum, 0.1)
    top_percentile = np.percentile(framenum, 99.9)
    locs_for_dme = locs_for_dme[(framenum > bottom_percentile) & (framenum < top_percentile)]
    framenum = framenum[(framenum > bottom_percentile) & (framenum < top_percentile)]
    
    framenum -= min(framenum)
    
    # #Histogram the framenrs and show:
    #     import matplotlib.pyplot as plt
    #     #Create a new figure
    #     plt.figure(88)
    #     #Plot the drift traces
    #     plt.plot(np.arange(int(len(framenum)))*(frame_time_for_dme),framenum)
    #     plt.show()
        
    
    #CRLB is hardcoded at half a pixel in x,y. Probably not the best implementation, but it seems to work
    crlb = np.ones(locs_for_dme.shape) * np.array((0.5,0.5))[None]
    
    #Estimate the drift!
    estimated_drift = dme.dme_estimate(locs_for_dme, framenum, 
                crlb, 
                framesperbin = framesperbinv, 
                imgshape=[fov_width, fov_width], 
                coarseFramesPerBin=int(np.floor(min(framesperbinv*10,max(framenum)/20))),
                coarseSigma=[0.2,0.2],
                useCuda=use_cuda,
                useDebugLibrary=False,
                estimatePrecision=False,
                display=False)

    #Briefly visualise the drift if wanted:
    if visualisation:
        import matplotlib.pyplot as plt
        #Close all previous plots, assumed max 10
        for i in range(10):
            plt.close()
        #Create a new figure
        plt.figure(88)
        #Plot the drift traces
        plt.plot(np.arange(int(len(estimated_drift)))*(frame_time_for_dme),estimated_drift[:,0]*(float(settings['PixelSize_nm']['value'])))
        plt.plot(np.arange(int(len(estimated_drift)))*(frame_time_for_dme),estimated_drift[:,1]*(float(settings['PixelSize_nm']['value'])))
        #Add axis labels:
        plt.xlabel('Time (ms)')
        plt.ylabel('Drift (nm)')
        plt.legend(['X drift', 'Y drift'])
        #Add a title:
        plt.title('Drift Estimation')
        plt.show()

    import logging
    #Remove all entries where a negative time was given:
    if len(resultArray[resultArray['t'] <= 0]):
        logging.warning('Removing ' + str(len(resultArray[resultArray['t'] <= 0])) + ' negative times')
    resultArray = resultArray[resultArray['t'] > 0]
    

    framenumFull = np.floor(np.array(resultArray['t'].values)/(frame_time_for_dme))
    framenumFull -= min(framenumFull)
    framenumFull = framenumFull.astype(int)
    #Get the drift of every localization - note the back-conversion from px to nm
    drift_locs = ([estimated_drift[min(i,len(estimated_drift)-1)][0]*(float(settings['PixelSize_nm']['value'])) for i in framenumFull],[estimated_drift[min(i,len(estimated_drift)-1)][1]*(float(settings['PixelSize_nm']['value'])) for i in framenumFull])
    
    import copy
    #Correct the resultarray for the drift
    drift_corr_locs = copy.deepcopy(resultArray)
    drift_corr_locs.loc[:,'x'] -= drift_locs[0]
    drift_corr_locs.loc[:,'y'] -= drift_locs[1]
    
    print(drift_locs[0][0], drift_locs[1][0])
    print(drift_locs[0][-1], drift_locs[1][-1])




    performance_metadata = f"Dummy function ran for seconds."
    print('Function one ran!')

    return drift_corr_locs, performance_metadata


# Simulate an SMLM dataset in 3D with blinking molecules
def smlm_simulation(
        drift_trace,
        fov_width, # field of view size in pixels
        loc_error, # localization error XYZ 
        n_sites, # number of locations where molecules blink on and off
        n_frames,
        on_prob = 0.1, # probability of a binding site generating a localization in a frame
        ): 
    
    """
    localization error is set to 20nm XY and 50nm Z precision 
    (assumping Z coordinates are in um and XY are in pixels)
    """

    # typical 2D acquisition with small Z range and large XY range        
    binding_sites = np.random.uniform([0,0,-1], [fov_width,fov_width,1], size=(n_sites,3))
    
    localizations = []
    framenum = []
    
    for i in range(n_frames):
        on = np.random.binomial(1, on_prob, size=n_sites).astype(bool)
        locs = binding_sites[on]*1
        # add localization error
        locs += drift_trace[i] + np.random.normal(0, loc_error, size=locs.shape)
        framenum.append(np.ones(len(locs),dtype=np.int32)*i)
        localizations.append(locs)
        
    return np.concatenate(localizations), np.concatenate(framenum)

