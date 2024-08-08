import inspect
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
    from Utils import utilsHelper
import pandas as pd
import numpy as np
import logging
import time

# from .dme import *

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Load_storedData": {
            "required_kwargs": [
                {"name": "fileLoc", "description": "File location (*.npz file)","default":'',"type":"fileLoc","display_text":"File location"},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Load a stored .npz obtained from DME/RCC drift correction performed in EVE.",
            "display_name": "Load stored drift correction DME/RCC"
        },
        "DriftCorr_entropyMin": {
            "required_kwargs": [
                {"name": "frame_time_for_dme", "description": "Frame-time used for drift-correction (in ms)","default":100.,"type":float,"display_text":"Frame time used in DME"},
                {"name": "frames_per_bin", "description": "Number of frames in every bin for dme drift correction ","default":50,"type":int,"display_text":"Frames per bin"},
                {"name": "visualisation", "description": "Visualisation of the drift traces (Boolean).","default":True,"display_text":"Visualisation"},
            ],
            "optional_kwargs": [
                {"name": "storeLoc", "description": "File location (*.npz file)","default":'',"type":"fileLoc","display_text":"Storage location (*.npz)"},
            ],
            "help_string": "Corrects drift based on entropy minimization in two dimensions. Original implementation from Cnossen et al., Optics Express, 2021.",
            "display_name": "Drift correction by entropy minimization [2D]"
        },
        "DriftCorr_entropyMin_3D": {
            "required_kwargs": [
                {"name": "frame_time_for_dme", "description": "Frame-time used for drift-correction (in ms)","default":100.,"type":float,"display_text":"Frame time used in DME"},
                {"name": "frames_per_bin", "description": "Number of frames in every bin for dme drift correction ","default":50,"type":int,"display_text":"Frames per bin"},
                {"name": "visualisation", "description": "Visualisation of the drift traces.","default":True,"display_text":"Visualisation"},
            ],
            "optional_kwargs": [
                {"name": "storeLoc", "description": "File location (*.npz file)","default":'',"type":"fileLoc","display_text":"Storage location (*.npz)"},
            ],
            "help_string": "Corrects drift based on entropy minimization in three dimensions. Original implementation from Cnossen et al., Optics Express, 2021.",
            "display_name": "Drift correction by entropy minimization [3D]"
        },
        "DriftCorr_RCC": {
            "required_kwargs": [
                {"name": "frame_time_for_dme", "description": "Frame-time used for drift-correction (in ms)","default":100.,"type":float,"display_text":"Frame time used in DME"},
                {"name": "nr_time_bins", "description": "Number of time bins","default":10,"type":int,"display_text":"Number of bins"},
                {"name": "zoom_level", "description": "Zoom level","default":2,"type":int,"display_text":"Zoom of RCC plots"},
                {"name": "visualisation", "description": "Visualisation of the drift traces.","default":True,"display_text":"Visualisation"},
                {"name": "ConvHist", "description": "Use convoluted histogram, ideally do not use - required for Linux","default":False,"display_text":"Use ConvHist (Linux; Boolean)"},
            ],
            "optional_kwargs": [
                {"name": "storeLoc", "description": "File location (*.npz file)","default":'',"type":"fileLoc","display_text":"Storage location (*.npz)"},
            ],
            "help_string": "Redudant cross-correlation drift correction. Based on the implementation from Cnossen et al., Optics Express, 2021; on linux, based on the implementatino from Martens et al., 2022",
            "display_name": "Drift correction by RCC (redundant cross-correlation)"
        }
    }





# from .dme import *
#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def Load_storedData(resultArray,findingResult,settings, **kwargs):
    import logging
    """
    Load and apply stored drift correction data.
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    
    # Load the .npz file
    loaded_data = np.load(kwargs['fileLoc'])
    try:
    # Access the variables
        frame_time_for_dme = loaded_data['frame_time_for_dme']
        estimated_drift = loaded_data['estimated_drift']
        pixelsize_nm = loaded_data['pixelsize_nm']
    except: 
        logging.error("Could not load the drift correction data from the file. Please check the file location.")
        return

    #2D drift corr
    if estimated_drift.shape[1] == 2:
        framenumFull = np.floor(np.array(resultArray['t'].values)/(frame_time_for_dme))
        framenumFull -= min(framenumFull)
        framenumFull = framenumFull.astype(int)
        #Get the drift of every localization - note the back-conversion from px to nm
        drift_locs = ([estimated_drift[min(i,len(estimated_drift)-1)][0]*(pixelsize_nm) for i in framenumFull],[estimated_drift[min(i,len(estimated_drift)-1)][1]*(pixelsize_nm) for i in framenumFull])
        
        import copy
        #Correct the resultarray for the drift
        drift_corr_locs = copy.deepcopy(resultArray)
        drift_corr_locs.loc[:,'x'] -= drift_locs[0]
        drift_corr_locs.loc[:,'y'] -= drift_locs[1]

        performance_metadata = f"2D driftcorr-load applied with settings {kwargs}."
        logging.info(f"2D drift corrected from file {kwargs['fileLoc']}.")

        return drift_corr_locs, performance_metadata
    elif estimated_drift.shape[1] == 3: #3D drift corr
        framenumFull = np.floor(np.array(resultArray['t'].values)/(frame_time_for_dme))
        framenumFull -= min(framenumFull)
        framenumFull = framenumFull.astype(int)
        #Get the drift of every localization - note the back-conversion from px to nm
        drift_locs = ([estimated_drift[min(i,len(estimated_drift)-1)][0]*(pixelsize_nm) for i in framenumFull],[estimated_drift[min(i,len(estimated_drift)-1)][1]*(pixelsize_nm) for i in framenumFull], [estimated_drift[min(i,len(estimated_drift)-1)][2]*(pixelsize_nm) for i in framenumFull])
        
        import copy
        #Correct the resultarray for the drift
        drift_corr_locs = copy.deepcopy(resultArray)
        drift_corr_locs.loc[:,'x'] -= drift_locs[0]
        drift_corr_locs.loc[:,'y'] -= drift_locs[1]
        drift_corr_locs.loc[:,'z [nm]'] -= drift_locs[2]
        
        performance_metadata = f"3D driftcorr-load applied with settings {kwargs}."
        logging.info(f"3D drift corrected from file {kwargs['fileLoc']}.")

        return drift_corr_locs, performance_metadata

def DriftCorr_entropyMin(resultArray,findingResult,settings,**kwargs):
    """ 
    Implementation of DME drift correction based on Cnossen et al. 2021 (https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245). 
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    #Import the correct package
    from .dme.dme import dme
    import logging
    
    #Set user variables
    frame_time_for_dme = float(kwargs['frame_time_for_dme']) #in ms
    framesperbinv = int(kwargs['frames_per_bin'])#in 'frames'
    use_cuda= settings['UseCUDA']['value']>0
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
    
    #CRLB is hardcoded at half a pixel in x,y. Probably not the best implementation, but it seems to work
    crlb = np.ones(locs_for_dme.shape) * np.array((0.5,0.5))[None]
    
    #Estimate the drift!
    estimated_drift = dme.dme_estimate(locs_for_dme, framenum, 
                crlb, 
                framesperbin = framesperbinv, 
                imgshape=[fov_width, fov_width], 
                coarseFramesPerBin=int(np.floor(min(framesperbinv*10,max(framenum)/20))),
                coarseSigma=[1,1],
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

    #To store, we should store the following data: fraem_time_for_dme, estimated_drift, pixelsize_nm
    storeLoc = kwargs['storeLoc']
    if storeLoc != '' and storeLoc != None and storeLoc != 'None' and storeLoc != ' ':
        try:
            #Check if storeLoc ends in .npz, otherwise add it:
            if storeLoc[-4:]!= '.npz':
                storeLoc += '.npz'
            # Save the variables to a single .npz file
            np.savez(storeLoc, 
                    frame_time_for_dme=frame_time_for_dme, 
                    estimated_drift=estimated_drift, 
                    pixelsize_nm=float(settings['PixelSize_nm']['value']))
        except:
            logging.error('Could not save drift correction data to'+ storeLoc)
    else:
        logging.debug("No drift correction stored")

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

    performance_metadata = f"Driftcorrection DME-2D applied with settings {kwargs}."

    return drift_corr_locs, performance_metadata

def DriftCorr_entropyMin_3D(resultArray,findingResult,settings,**kwargs):
    """ 
    Implementation of DME drift correction based on Cnossen et al. 2021 (https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245). 
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    #Import the correct package
    from .dme.dme import dme
    import logging
    
    #Set user variables
    frame_time_for_dme = float(kwargs['frame_time_for_dme']) #in ms
    framesperbinv = int(kwargs['frames_per_bin'])#in 'frames'
    use_cuda= settings['UseCUDA']['value']>0
    visualisation=utilsHelper.strtobool(kwargs['visualisation'])
    
    #Hard-coded variables
    fov_width = 200 #in pixels

    #Obtain the localizations from the resultArray
    resultArray=resultArray.dropna()
    locs_for_dme = np.column_stack(((resultArray['x'].values-min(resultArray['x'])),
                                    (resultArray['y'].values-min(resultArray['y'])),
                                    resultArray['z [nm]'].values))
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
    
    #CRLB is hardcoded at half a pixel in x,y. Probably not the best implementation, but it seems to work
    crlb = np.ones(locs_for_dme.shape) * np.array((0.5,0.5,0.5))[None]
    
    #Estimate the drift!
    # estimated_drift = np.load('C:\\Data\\EBS\\3D_Tubulin\\3d_drift_small_top_right.npy')
    estimated_drift = dme.dme_estimate(locs_for_dme, framenum, 
                crlb, 
                framesperbin = framesperbinv, 
                imgshape=[fov_width, fov_width,fov_width], 
                coarseFramesPerBin=int(np.floor(min(framesperbinv*10,max(framenum)/20))),
                coarseSigma=[1,1,1],
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
        plt.plot(np.arange(int(len(estimated_drift)))*(frame_time_for_dme),estimated_drift[:,2]*(float(settings['PixelSize_nm']['value'])))
        #Add axis labels:
        plt.xlabel('Time (ms)')
        plt.ylabel('Drift (nm)')
        plt.legend(['X drift', 'Y drift','Z drift'])
        #Add a title:
        plt.title('Drift Estimation')
        plt.show()

    #To store, we should store the following data: fraem_time_for_dme, estimated_drift, pixelsize_nm
    storeLoc = kwargs['storeLoc']
    if storeLoc != '' and storeLoc != None and storeLoc != 'None' and storeLoc != ' ':
        try:
            #Check if storeLoc ends in .npz, otherwise add it:
            if storeLoc[-4:]!= '.npz':
                storeLoc += '.npz'
            # Save the variables to a single .npz file
            np.savez(storeLoc, 
                    frame_time_for_dme=frame_time_for_dme, 
                    estimated_drift=estimated_drift, 
                    pixelsize_nm=float(settings['PixelSize_nm']['value']))
        except:
            logging.error('Could not save drift correction data to'+ storeLoc)
    else:
        logging.debug("No drift correction stored")
        
    #Remove all entries where a negative time was given:
    if len(resultArray[resultArray['t'] <= 0]):
        logging.warning('Removing ' + str(len(resultArray[resultArray['t'] <= 0])) + ' negative times')
    resultArray = resultArray[resultArray['t'] > 0]
    

    framenumFull = np.floor(np.array(resultArray['t'].values)/(frame_time_for_dme))
    framenumFull -= min(framenumFull)
    framenumFull = framenumFull.astype(int)
    #Get the drift of every localization - note the back-conversion from px to nm
    drift_locs = ([estimated_drift[min(i,len(estimated_drift)-1)][0]*(float(settings['PixelSize_nm']['value'])) for i in framenumFull],[estimated_drift[min(i,len(estimated_drift)-1)][1]*(float(settings['PixelSize_nm']['value'])) for i in framenumFull], [estimated_drift[min(i,len(estimated_drift)-1)][2]*(float(settings['PixelSize_nm']['value'])) for i in framenumFull])
    
    import copy
    #Correct the resultarray for the drift
    drift_corr_locs = copy.deepcopy(resultArray)
    drift_corr_locs.loc[:,'x'] -= drift_locs[0]
    drift_corr_locs.loc[:,'y'] -= drift_locs[1]
    drift_corr_locs.loc[:,'z [nm]'] -= drift_locs[2]

    performance_metadata = f"Driftcorrection DME-3D applied with settings {kwargs}."

    return drift_corr_locs, performance_metadata

def DriftCorr_RCC(resultArray,findingResult,settings,**kwargs):
    """ 
    Implementation of DME drift correction based on Cnossen et al. 2021 (https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245). 
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    #Import the correct package
    from .dme.dme import dme
    from .dme.dme.rcc import rcc, rcc3D
    import logging
    
    
    #Set user variables
    frame_time_for_dme = float(kwargs['frame_time_for_dme']) #in ms
    nr_time_bins = int(kwargs['nr_time_bins'])
    zoom_level = int(kwargs['zoom_level'])
    use_cuda= settings['UseCUDA']['value']>0
    visualisation=utilsHelper.strtobool(kwargs['visualisation'])

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
    
    #RCC drift
    shift_px = rcc_own(locs_for_dme, framenum, nr_time_bins, zoom=zoom_level, 
        sigma=1, maxpairs=1000, use_cuda=use_cuda, use_conv_hist = utilsHelper.strtobool(kwargs['ConvHist']))

    #Briefly visualise the drift if wanted:
    if visualisation:
        import matplotlib.pyplot as plt
        #Close all previous plots, assumed max 10
        for _ in range(10):
            plt.close()
        #Create a new figure
        plt.figure(88)
        #Plot the drift traces
        plt.plot(np.arange(int(len(shift_px[0])))*(frame_time_for_dme),shift_px[0][:,0]*(float(settings['PixelSize_nm']['value'])))
        plt.plot(np.arange(int(len(shift_px[0])))*(frame_time_for_dme),shift_px[0][:,1]*(float(settings['PixelSize_nm']['value'])))
        #Add axis labels:
        plt.xlabel('Time (ms)')
        plt.ylabel('Drift (nm)')
        plt.legend(['X drift', 'Y drift'])
        #Add a title:
        plt.title('Drift Estimation')
        plt.show()

    #To store, we should store the following data: fraem_time_for_dme, estimated_drift, pixelsize_nm
    storeLoc = kwargs['storeLoc']
    if storeLoc != '' and storeLoc != None and storeLoc != 'None' and storeLoc != ' ':
        try:
            #Check if storeLoc ends in .npz, otherwise add it:
            if storeLoc[-4:]!= '.npz':
                storeLoc += '.npz'
            # Save the variables to a single .npz file
            np.savez(storeLoc, 
                    frame_time_for_dme=frame_time_for_dme, 
                    estimated_drift=shift_px, 
                    pixelsize_nm=float(settings['PixelSize_nm']['value']))
        except:
            logging.error('Could not save drift correction data to'+ storeLoc)
    else:
        logging.debug("No drift correction stored")

    #Remove all entries where a negative time was given:
    if len(resultArray[resultArray['t'] <= 0]):
        logging.warning('Removing ' + str(len(resultArray[resultArray['t'] <= 0])) + ' negative times')
    resultArray = resultArray[resultArray['t'] > 0]
    

    framenumFull = np.floor(np.array(resultArray['t'].values)/(frame_time_for_dme))
    framenumFull -= min(framenumFull)
    framenumFull = framenumFull.astype(int)
    #Get the drift of every localization - note the back-conversion from px to nm
    drift_locs = ([shift_px[0][min(i,len(shift_px[0])-1)][0]*(float(settings['PixelSize_nm']['value'])) for i in framenumFull],[shift_px[0][min(i,len(shift_px[0])-1)][1]*(float(settings['PixelSize_nm']['value'])) for i in framenumFull])
    
    import copy
    #Correct the resultarray for the drift
    drift_corr_locs = copy.deepcopy(resultArray)
    drift_corr_locs.loc[:,'x'] -= drift_locs[0]
    drift_corr_locs.loc[:,'y'] -= drift_locs[1]

    performance_metadata = f"Driftcorrection RCC applied with settings {kwargs}."

    return drift_corr_locs, performance_metadata


def rcc_own(xy, framenum, timebins, zoom=1, 
        sigma=1, maxpairs=1000, use_cuda=False, use_conv_hist = False):
    """
    Child function to do RCC
    """
    
    from .dme.dme import dme
    from .dme.dme.rcc import findshift_pairs, InterpolatedUnivariateSpline
    
    rendersize = int(np.max(xy))
    area = np.array([rendersize,rendersize])
    nframes = np.max(framenum)+1
    framesperbin = nframes/timebins
    
    imgshape = area*zoom
    images = np.zeros((timebins, *imgshape))

    #Normally use Cnossen's RCC
    if use_conv_hist == False:
        from .dme.dme.native_api import NativeAPI
        with NativeAPI(use_cuda) as dll:
            for k in range(timebins):
                img = np.zeros(imgshape,dtype=np.float32)
                
                indices = np.nonzero((0.5 + framenum/framesperbin).astype(int)==k)[0]

                spots = np.zeros((len(indices), 5), dtype=np.float32)
                spots[:, 0] = xy[indices,0] * zoom
                spots[:, 1] = xy[indices,1] * zoom
                spots[:, 2] = sigma
                spots[:, 3] = sigma
                spots[:, 4] = 1
                
                if len(spots) == 0:
                    raise ValueError(f'no spots in bin {k}')

                images[k] = dll.DrawGaussians(img, spots)
    elif use_conv_hist == True:
        #On Linux, use Martens' implementation -- slower
        maxx = imgshape[0]/zoom
        maxy = imgshape[1]/zoom
        for k in range(timebins):
            img = np.zeros(imgshape,dtype=np.float32)
            
            indices = np.nonzero((0.5 + framenum/framesperbin).astype(int)==k)[0]

            spots = np.zeros((len(indices), 5), dtype=np.float32)
            spots[:, 0] = xy[indices,0] * zoom
            spots[:, 1] = xy[indices,1] * zoom
            spots[:, 2] = sigma
            spots[:, 3] = sigma
            spots[:, 4] = 1
            
            if len(spots) == 0:
                raise ValueError(f'no spots in bin {k}')
            
            
            histogram_original = np.histogram2d(spots[:, 0], spots[:, 1], range=[[0,maxx], [0,maxy]],
                                            bins=[range(int(maxx*zoom+1)), range(int(maxy*zoom+1))])
            
            kernel = create_kernel(7)
            import scipy
            images[k] = scipy.signal.convolve2d(histogram_original[0], kernel, mode='same').T #Transpose to fix x,y shifting wrt DME
        
    #print(f"RCC pairs: {timebins*(timebins-1)//2}. Bins={timebins}")
    pairs = np.array(np.triu_indices(timebins,1)).T
    if len(pairs)>maxpairs:
        pairs = pairs[np.random.choice(len(pairs),maxpairs)]
    pair_shifts = findshift_pairs(images, pairs)
    
    A = np.zeros((len(pairs),timebins))
    A[np.arange(len(pairs)),pairs[:,0]] = 1
    A[np.arange(len(pairs)),pairs[:,1]] = -1
    
    inv = np.linalg.pinv(A)
    shift_x = inv @ pair_shifts[:,0]
    shift_y = inv @ pair_shifts[:,1]
    shift_y -= shift_y[0]
    shift_x -= shift_x[0]
    shift = -np.vstack((shift_x,shift_y)).T / zoom
        
    t = (0.5+np.arange(timebins))*framesperbin
    
    shift -= np.mean(shift,0)

    shift_estim = np.zeros((len(shift),3))
    shift_estim[:,[0,1]] = shift
    shift_estim[:,2] = t

    if timebins != nframes:
        spl_x = InterpolatedUnivariateSpline(t, shift[:,0], k=2)
        spl_y = InterpolatedUnivariateSpline(t, shift[:,1], k=2)
    
        shift_interp = np.zeros((nframes,2))
        shift_interp[:,0] = spl_x(np.arange(nframes))
        shift_interp[:,1] = spl_y(np.arange(nframes))
    else:
        c = shift
    
    return shift_interp, shift_estim, images

def create_kernel(size):
    """
    Kernel required for frame visualisation for RCC
    """
    #Check if size is odd:
    if size % 2 == 0:
        import logging
        size += 1
        logging.info(f'Kerning size was even, made odd and changed to size {size}')
    kernel = np.zeros((size,size))
    
    center = (size/2-.5,size/2-.5)
    
    
    for xx in range(size):
        for yy in range(size):
            dist_to_center = np.ceil(np.sqrt((xx-center[0])**2+(yy-center[1])**2))
            kernel[xx,yy] = max(0,center[0]+1-dist_to_center)
            
    return kernel