import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time
import scipy, logging
# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Histogram_convolution": {
            "required_kwargs": [
                {"name": "PxSize","display_text": "Pixel size (nm)", "description": "Pixel-to-pseudopixel ratio","default":10},
                {"name": "Convolution_kernel","display_text": "Convolution kernel size (nm)", "description": "Convolution pixel size","default":50},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Returns a test dictionary and metadata string, to check if the program is working properly.",
            "display_name": "2D Histogram with circular kernel convolution"
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def create_kernel(size):
    #Check if size is odd:
    if size % 2 == 0:
        size += 1
        logging.info(f'Kerning size was even, made odd and changed to size {size}')
    kernel = np.zeros((size,size))
    
    center = (size/2-.5,size/2-.5)
    
    
    for xx in range(size):
        for yy in range(size):
            dist_to_center = np.ceil(np.sqrt((xx-center[0])**2+(yy-center[1])**2))
            kernel[xx,yy] = max(0,center[0]+1-dist_to_center)
            
    return kernel

def Histogram_convolution(resultArray,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    # Start the timer
    start_time = time.time()
    
    #Get pixels sizes for the histogram and kernel
    pxsizeHist = float(kwargs['PxSize'])
    pxsizeKernel = float(kwargs['Convolution_kernel'])
    
    #Set them to 'magnification' and 'kernel size in px'
    zoomvalue = float(settings['PixelSize_nm']['value'])/pxsizeHist
    KernelWidthPx = int(round(pxsizeKernel/pxsizeHist))
    
    # zoomvalue=float(kwargs['ZoomValue'])
    
    #Idea: create an empty array with the right size, i.e. ZoomValue times bigger than the maximum size of the results.
    #Then simply increase the value of the pixels in that array based on resultArray
    
    #Get the min/max bounds in pixel units:
    xoffset = np.min(resultArray['x']) / float(settings['PixelSize_nm']['value'])
    maxx = np.max(resultArray['x']) / float(settings['PixelSize_nm']['value'])
    yoffset = np.min(resultArray['y']) / float(settings['PixelSize_nm']['value'])
    maxy = np.max(resultArray['y']) / float(settings['PixelSize_nm']['value'])
    #Scale them so that minx, miny = 0:
    minx = 0
    miny = 0
    maxx = maxx - xoffset
    maxy = maxy - yoffset
    
    #Take resultArray and remove all nan-entries:   
    data = resultArray[['x','y']].dropna() / float(settings['PixelSize_nm']['value'])
    data['x'] -= xoffset
    data['y'] -= yoffset
    
    
    histogram_original = np.histogram2d(data['x'], data['y'], range=[[minx, maxx], [miny, maxy]],
                                    bins=[int(maxx*zoomvalue), int(maxy*zoomvalue)])
    
    kernel = create_kernel(KernelWidthPx)
    
    histogram_convolved = scipy.signal.convolve2d(histogram_original[0], kernel, mode='same')
    
    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    #Scale should be the scale of pixel - to - um. E.g. a scale of 0.01 means 100 pixels = 1 um
    scale = (maxx*(float(settings['PixelSize_nm']['value']))/1000)/np.shape(histogram_convolved)[0]

    logging.info('Histogram with convolution created!')

    return histogram_convolved.T, scale
