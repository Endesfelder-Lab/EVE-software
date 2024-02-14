import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import logging
#Obtain eventdistribution functions
from EventDistributions import eventDistributions

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "TwoDTimedelay": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "2D event delays min, max, mean per pixel in candidate cluster.",
            "display_name": "2D event delay plots of the candidate cluster"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def TwoDTimedelay(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    ax_min = figure.add_subplot(131)
    ax_max = figure.add_subplot(132)
    ax_mean = figure.add_subplot(133)
    ax_min.set_title("min delay per pixel")
    ax_max.set_title("max delay per pixel")
    ax_mean.set_title("mean delay per pixel")
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.17)

    min = eventDistributions.MinTimeDiff(findingResult).dist2D
    max = eventDistributions.MaxTimeDiff(findingResult).dist2D
    mean = eventDistributions.AverageTimeDiff(findingResult).dist2D

    x_edges, y_edges = eventDistributions.Hist2d_xy(findingResult).x_edges, eventDistributions.Hist2d_xy(findingResult).y_edges

    # Plot the 2D histograms
    ax_min.pcolormesh(x_edges, y_edges, min)
    ax_min.set_aspect('equal')
    ax_max.pcolormesh(x_edges, y_edges, max)
    ax_max.set_aspect('equal')
    ax_mean.pcolormesh(x_edges, y_edges, mean)
    ax_mean.set_aspect('equal')
    ax_min.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    ax_mean.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    ax_max.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')

    # Add and set labels
    ax_min.set_xlabel('x [px]')
    ax_min.set_ylabel('y [px]')
    ax_max.set_xlabel('x [px]')
    ax_max.set_ylabel('y [px]')
    ax_mean.set_xlabel('x [px]')
    ax_mean.set_ylabel('y [px]')


    # required output none
    return 1
