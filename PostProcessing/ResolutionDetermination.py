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
        "FRC": {
            "required_kwargs": [
                {"name": "pixel_sixe_recon", "description": "Zoom level","default":10,"type":int,"display_text":"Reconstruction Pixel Size (nm)"},
                {"name": "sigma", "description": "Precision (nm)","default":20,"type":float,"display_text":"Precision of localizations (nm)"},
                {"name": "visualisation", "description": "Visualisation of the drift traces.","default":True,"display_text":"Visualisation"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "FRC",
            "display_name": "FRC (Fourier Ring Correlation)"
        }
    }





# from .dme import *
#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

def FRC(resultArray,findingResult,settings,**kwargs):

    from .dme.dme.native_api import NativeAPI
    
    sigma = float(kwargs['sigma'])  
    use_cuda= settings['UseCUDA']['value']>0
    
    pixel_recon_dim = (float(kwargs['pixel_sixe_recon']))
    zoom = float(settings['PixelSize_nm']['value'])/pixel_recon_dim
    visualisation=utilsHelper.strtobool(kwargs['visualisation'])
    
    #Obtain the localizations from the resultArray
    resultArray=resultArray.dropna()
    xy = np.column_stack((resultArray['x'].values-min(resultArray['x']),resultArray['y'].values-min(resultArray['y'])))
    #Convert it to pixel-units
    xy/=(float(settings['PixelSize_nm']['value']))
    
    
    coin = np.random.binomial(1,0.5, len(xy)).reshape(len(xy),)
    xy1, xy2 = xy[coin == 0], xy[coin == 1]
    
    
    rendersize = int(np.max(xy))
    area = np.array([rendersize,rendersize])
    
    imgshape = np.ceil(area*zoom).astype(int)
    FRC_im1 = np.zeros((0, *imgshape))
    FRC_im2 = np.zeros((0, *imgshape))

    #I have no idea why, but FRC doesn't work with the Gaussian drawin, but does work with histograms+lurring of those - so that's what I'm forcing to do!
    # if utilsHelper.strtobool(kwargs['ConvHist']) == False:
    #     with NativeAPI(use_cuda) as dll:
    #         img = np.zeros(imgshape,dtype=np.float32)
    #         spots = np.zeros((len(xy1), 5), dtype=np.float32)
    #         spots[:, 0] = xy1[:,0] * zoom
    #         spots[:, 1] = xy1[:,1] * zoom
    #         spots[:, 2] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
    #         spots[:, 3] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
    #         spots[:, 4] = 1
    #         FRC_im1 = dll.DrawGaussians(img, spots)
            
    #         img = np.zeros(imgshape,dtype=np.float32)
    #         spots = np.zeros((len(xy2), 5), dtype=np.float32)
    #         spots[:, 0] = xy2[:,0] * zoom
    #         spots[:, 1] = xy2[:,1] * zoom
    #         spots[:, 2] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
    #         spots[:, 3] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
    #         spots[:, 4] = 1
    #         FRC_im2 = dll.DrawGaussians(img, spots)
    # else:
    maxx = area[0]
    maxy = area[1]
    img = np.zeros(imgshape,dtype=np.float32)

    spots = np.zeros((len(xy1), 5), dtype=np.float32)
    spots[:, 0] = xy1[:,0] * zoom
    spots[:, 1] = xy1[:,1] * zoom
    spots[:, 4] = 1
    histogram_original = np.histogram2d(spots[:, 0], spots[:, 1], range=[[0,maxx], [0,maxy]],
                                    bins=[range(int(maxx*zoom+1)), range(int(maxy*zoom+1))])
    kernel = create_kernel(int((sigma/(float(settings['PixelSize_nm']['value'])) * zoom)*2.355))
    import scipy
    FRC_im1 = scipy.signal.convolve2d(histogram_original[0], kernel, mode='same').T #Transpose to fix x,y 
    
    spots = np.zeros((len(xy2), 5), dtype=np.float32)
    spots[:, 0] = xy2[:,0] * zoom
    spots[:, 1] = xy2[:,1] * zoom
    spots[:, 2] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
    spots[:, 3] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
    spots[:, 4] = 1
    histogram_original = np.histogram2d(spots[:, 0], spots[:, 1], range=[[0,maxx], [0,maxy]],
                                    bins=[range(int(maxx*zoom+1)), range(int(maxy*zoom+1))])
    kernel = create_kernel(int((sigma/(float(settings['PixelSize_nm']['value'])) * zoom)*2.355))
    import scipy
    FRC_im2 = scipy.signal.convolve2d(histogram_original[0], kernel, mode='same').T #Transpose to fix x,y 
        
        
    import logging
    logging.info('Image created!')
    
    
    #Need to be normalized for FRC:
    FRC_im1 /= np.sum(FRC_im1)
    FRC_im2 /= np.sum(FRC_im2)

    # Two-dimensional discrete Fourier transforms are computed for both images
    image1_f, image2_f = np.fft.fft2(FRC_im1), np.fft.fft2(FRC_im2)
    # The product of the two Fourier transforms is computed...
    # af1f2 = np.fft.fftshift(np.real(image1_f * np.conj(image2_f)))
    af1f2 = np.fft.fftshift(np.real(image1_f * np.conj(image2_f)))
    # ...as well as the absolute values of squared Fourier transforms. These values 
    # will be later used to compute the corrleation of both Fourier transforms
    af1_2, af2_2 = np.fft.fftshift(np.abs(image1_f**2)), np.fft.fftshift(np.abs(image2_f**2))
    
    
    #Next, we create a distance map that simply contains the distances of every
    #pixel to the center of the image.
    distancemap = np.zeros(FRC_im1.shape)
    midpoint = np.array(FRC_im1.shape) // 2
    xx, yy = np.ogrid[:FRC_im1.shape[0], :FRC_im1.shape[1]]
    distancemap = np.round(np.sqrt((xx - midpoint[0])**2 + (yy - midpoint[1])**2))

    #Now we calculate the total value that the af1f2, af1_2 and af2_2 images
    #have at specific distances from the center of the image (i.e. at specific
    #rings centered around the center of the image)


    #First we assign a maximum distance (1-dimensional length of the image),
    #and create a few empty matrices.
    max_dist = int(np.floor(np.shape(distancemap)[0]/2));
    f1f2_r = np.zeros([max_dist,1])
    f12_r = np.zeros([max_dist,1])
    f22_r = np.zeros([max_dist,1])
    FRC_values = np.zeros([max_dist,1])

    #Now we loop over all pixel distances, find which pixels in the distancemap have
    #these distances (i.e. this creates circles in the map), and
    #calculate the sum of these pixels in the af1f2, af1_2, and af2_2 images.
    # for d in range(0,max_dist):
    #     ringMap = (distancemap == d)
    #     f1f2_r[d] = np.sum(af1f2*ringMap)
    #     f12_r[d] = np.sum(af1_2*ringMap)
    #     f22_r[d] = np.sum(af2_2*ringMap)
    #     #Now the FRC values are calculated as follows
    #     FRC_values[d] = (f1f2_r[d]/(np.sqrt(f12_r[d])*np.sqrt(f22_r[d])))
    
    # plt.imshow(ringMap)
    # plt.show()
    # # Initialize arrays for storing results
    f1f2_r = np.zeros(max_dist - 1)
    f12_r = np.zeros(max_dist - 1)
    f22_r = np.zeros(max_dist - 1)
    FRC_values = np.zeros(max_dist - 1)

    # Create a boolean mask for the ring pixels
    mask = distancemap[:, :, np.newaxis] == np.arange(1, max_dist)

    # Calculate the sum of pixels in the af1f2, af1_2, and af2_2 images for each distance
    f1f2_r = np.sum(af1f2[:,:,np.newaxis] * mask, axis=(0, 1))
    f12_r = np.sum(af1_2[:,:,np.newaxis] * mask, axis=(0, 1))
    f22_r = np.sum(af2_2[:,:,np.newaxis] * mask, axis=(0, 1))

    # Calculate the FRC values
    FRC_values = f1f2_r / np.sqrt(f12_r * f22_r)

    # The NaN-values need to be interpolated out
    FRC_values = pd.DataFrame(FRC_values, columns=['FRC'])
    FRC_values.interpolate(inplace=True, method='linear')

    #Normalize FRC values? Not sure if we should? But doesn't make sense otherwise
    if np.min(FRC_values)>0:
        logging.warning('FRC curve had to be normalized - probably FRC was not properly calculated!')
        FRC_values = FRC_values-np.min(FRC_values)
        FRC_values=FRC_values/np.max(FRC_values)

    import matplotlib.pyplot as plt

    #The x-axis (spatial frequency) goes from 0 to 1/(2*pixel_recon_dim)
    spat_freq = np.linspace(0, 1/(2*pixel_recon_dim), len(FRC_values))

    #We find the FRC resolution
    FRCresolution=0
    try:
        FRCresolutionID = np.min(np.where(FRC_values<(1/7))[0])
        FRCresolution = 1/spat_freq[FRCresolutionID]
    except:
        FRCresolution = 0

    #Briefly visualise the drift if wanted:
    if visualisation:
        import matplotlib.pyplot as plt
        #Close all previous plots, assumed max 10
        for _ in range(10):
            plt.close()
        
        #Create a new figure
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs[0,0].imshow(FRC_im1, cmap='gray')
        axs[0,0].set_title('Image1')
        axs[0,1].imshow(FRC_im2, cmap='gray')
        axs[0,1].set_title('Image2')
        axs[1,0].imshow(np.real(image1_f), cmap='gray')
        axs[1,0].set_title('af1f2')
        #Plot the FRC val
        axs[1,1].plot(spat_freq, FRC_values)
        axs[1,1].axhline(1/7, color='r', linestyle='-')
        axs[1,1].set_xlabel('Spatial frequency [1/nm]')
        axs[1,1].set_ylabel('FRC')
        axs[1,1].set_title(f"FRC resolution: {np.round(FRCresolution, 1)} nm");
        plt.show()

    import logging
    logging.info(f"FRC resolution found: {np.round(FRCresolution, 1)} nm")
    performance_metadata = f"Dummy function ran for seconds."
    print('Function one ran!')

    return resultArray, performance_metadata


def FRC_old(resultArray,findingResult,settings,**kwargs):
    """ 
    Implementation of DME drift correction based on Cnossen et al. 2021 (https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245). 
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    #Import the correct package
    from .dme.dme import dme
    from .dme.dme.rcc import rcc, rcc3D
    
    #Set user variables
    frame_time_for_dme = 100 #in ms
    zoom_level = int(kwargs['zoom_level'])
    use_cuda= settings['UseCUDA']['value']>0
    visualisation=utilsHelper.strtobool(kwargs['visualisation'])

    #Obtain the localizations from the resultArray
    resultArray=resultArray.dropna()
    locs_for_vis = np.column_stack((resultArray['x'].values-min(resultArray['x']),resultArray['y'].values-min(resultArray['y'])))
    #Convert it to pixel-units
    # locs_for_vis/=(float(settings['PixelSize_nm']['value']))
    
    # #Get the 'frame' for each localization based on user-defined frame_time_for_dme
    # framenumFull = np.floor(np.array(resultArray['t'].values)/(frame_time_for_dme))
    # #Pop -1 entries:
    # locs_for_vis = locs_for_vis[framenumFull != -1]
    # framenum = framenumFull[framenumFull != -1]
    # framenum -= min(framenum)
    # framenum = framenum.astype(int)
    
    # #to prevent unexpected errors: remove the bottom and top 0.1 percentile:
    # bottom_percentile = np.percentile(framenum, 0.1)
    # top_percentile = np.percentile(framenum, 99.9)
    # locs_for_vis = locs_for_vis[(framenum > bottom_percentile) & (framenum < top_percentile)]
    
    #Split in two for FRC:
    coin = np.random.binomial(1,0.5, len(locs_for_vis)).reshape(len(locs_for_vis),)
    locs_for_vis1, locs_for_vis2 = locs_for_vis[coin == 0], locs_for_vis[coin == 1]
    
    rendersize = [int(np.max(locs_for_vis[:,0])),int(np.max(locs_for_vis[:,1]))]
    rendersize_pxUnits = [int(np.max(locs_for_vis[:,0])//float(settings['PixelSize_nm']['value'])),int(np.max(locs_for_vis[:,1])//float(settings['PixelSize_nm']['value']))]
    
    pixel_recon_dim = float(settings['PixelSize_nm']['value'])/float(zoom_level)
    sigma = float(kwargs['display_prec'])
    
    #Get images
    FRC_im1 = create_vis(locs_for_vis1/float(settings['PixelSize_nm']['value']), rendersize_pxUnits, zoom=zoom_level, 
        sigma=sigma, use_cuda=use_cuda, use_conv_hist = utilsHelper.strtobool(kwargs['ConvHist']))
    FRC_im2 = create_vis(locs_for_vis2/float(settings['PixelSize_nm']['value']), rendersize_pxUnits, zoom=zoom_level, 
        sigma=sigma, use_cuda=use_cuda, use_conv_hist = utilsHelper.strtobool(kwargs['ConvHist']))

    # FRC_im1, _, _ = np.histogram2d(locs_for_vis1[:,0], locs_for_vis1[:,1], bins=[int((rendersize[0])/pixel_recon_dim), int((rendersize[1])/pixel_recon_dim)])
    # FRC_im2, _, _ = np.histogram2d(locs_for_vis2[:,0], locs_for_vis2[:,1], bins=[int((rendersize[0])/pixel_recon_dim), int((rendersize[1])/pixel_recon_dim)])

    #Need to be normalized for FRC:
    FRC_im1 /= np.sum(FRC_im1)
    FRC_im2 /= np.sum(FRC_im2)

    # Two-dimensional discrete Fourier transforms are computed for both images
    image1_f, image2_f = np.fft.fft2(FRC_im1), np.fft.fft2(FRC_im2)
    # The product of the two Fourier transforms is computed...
    af1f2 = np.fft.fftshift(np.real(image1_f * np.conj(image2_f)))
    # ...as well as the absolute values of squared Fourier transforms. These values 
    # will be later used to compute the corrleation of both Fourier transforms
    af1_2, af2_2 = np.fft.fftshift(np.abs(image1_f)**2), np.fft.fftshift(np.abs(image2_f)**2)
    
    
    #Next, we create a distance map that simply contains the distances of every
    #pixel to the center of the image.
    distancemap = np.zeros(FRC_im1.shape)
    midpoint = np.array(FRC_im1.shape) // 2
    xx, yy = np.ogrid[:FRC_im1.shape[0], :FRC_im1.shape[1]]
    distancemap = np.round(np.sqrt((xx - midpoint[0])**2 + (yy - midpoint[1])**2))

    #Now we calculate the total value that the af1f2, af1_2 and af2_2 images
    #have at specific distances from the center of the image (i.e. at specific
    #rings centered around the center of the image)


    #First we assign a maximum distance (1-dimensional length of the image),
    #and create a few empty matrices.
    max_dist = int(np.floor(np.shape(distancemap)[0]/2));
    f1f2_r = np.zeros([max_dist,1])
    f12_r = np.zeros([max_dist,1])
    f22_r = np.zeros([max_dist,1])
    FRC_values = np.zeros([max_dist,1])

    #Now we loop over all pixel distances, find which pixels in the distancemap have
    #these distances (i.e. this creates circles in the map), and
    #calculate the sum of these pixels in the af1f2, af1_2, and af2_2 images.
    for d in range(1,max_dist):
        ringMap = (distancemap == d)
        f1f2_r[d-1] = np.sum(af1f2*ringMap)
        f12_r[d-1] = np.sum(af1_2*ringMap)
        f22_r[d-1] = np.sum(af2_2*ringMap)
        #Now the FRC values are calculated as follows
        FRC_values[d-1] = (f1f2_r[d-1]/np.sqrt(f12_r[d-1]*f22_r[d-1]))

    # The NaN-values need to be interpolated out
    FRC_values = pd.DataFrame(FRC_values, columns=['FRC'])
    FRC_values.interpolate(inplace=True, method='linear')

    import matplotlib.pyplot as plt

    #The x-axis (spatial frequency) goes from 0 to 1/(2*pixel_recon_dim)
    spat_freq = np.linspace(0, 1/(2*pixel_recon_dim), len(FRC_values))

    #We find the FRC resolution
    FRCresolutionID = np.min(np.where(FRC_values<(1/7))[0])
    FRCresolution = 1/spat_freq[FRCresolutionID]

    #Briefly visualise the drift if wanted:
    if visualisation:
        import matplotlib.pyplot as plt
        #Close all previous plots, assumed max 10
        for _ in range(10):
            plt.close()
        
        #Create a new figure
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs[0].imshow(FRC_im1, cmap='gray')
        axs[0].set_title('Image1')
        axs[1].imshow(FRC_im2, cmap='gray')
        axs[1].set_title('Image2')
        axs[2].imshow(af1f2, cmap='gray')
        axs[2].set_title('af1f2')
        #Plot the FRC val
        axs[3].plot(spat_freq, FRC_values)
        axs[3].axhline(1/7, color='r', linestyle='-')
        axs[3].xlabel('Spatial frequency [1/nm]')
        axs[3].ylabel('FRC')
        axs[3].title(f"FRC resolution: {np.round(FRCresolution, 1)} nm");
        plt.show()

    import logging
    logging.info(f"FRC resolution found: {np.round(FRCresolution, 1)} nm")
    performance_metadata = f"Dummy function ran for seconds."
    print('Function one ran!')

    return resultArray, performance_metadata


def create_vis(xy, rendersize, zoom=1, 
        sigma=1, use_cuda=False, use_conv_hist = False):
    
    from .dme.dme import dme
    from .dme.dme.rcc import findshift_pairs, InterpolatedUnivariateSpline
    
    area = np.array([rendersize[0],rendersize[1]])
    
    imgshape = area*zoom
    image = np.zeros(*imgshape)

    if use_conv_hist == False:
        from .dme.dme.native_api import NativeAPI
        with NativeAPI(use_cuda) as dll:
            img = np.zeros(imgshape,dtype=np.float32)

            spots = np.zeros((len(xy), 5), dtype=np.float32)
            spots[:, 0] = xy[:,0] * zoom
            spots[:, 1] = xy[:,1] * zoom
            spots[:, 2] = sigma * zoom
            spots[:, 3] = sigma * zoom
            spots[:, 4] = 1
            
            if len(spots) == 0:
                raise ValueError(f'no spots in image')

            image = dll.DrawGaussians(img, spots)
    elif use_conv_hist == True:
        maxx = rendersize[0]
        maxy = rendersize[1]
        img = np.zeros(imgshape,dtype=np.float32)

        spots = np.zeros((len(xy), 5), dtype=np.float32)
        spots[:, 0] = xy[:,0] * zoom
        spots[:, 1] = xy[:,1] * zoom
        spots[:, 2] = sigma
        spots[:, 3] = sigma
        spots[:, 4] = 1
        
        if len(spots) == 0:
            raise ValueError(f'no spots in image')
        
        
        histogram_original = np.histogram2d(spots[:, 0], spots[:, 1], range=[[0,maxx], [0,maxy]],
                                        bins=[range(int(maxx*zoom+1)), range(int(maxy*zoom+1))])
        
        kernel = create_kernel(7)
        import scipy
        image = scipy.signal.convolve2d(histogram_original[0], kernel, mode='same').T #Transpose to fix x,y shifting wrt DME
        
    
    return image

def create_kernel(size):
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