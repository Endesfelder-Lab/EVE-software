import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import logging
#Obtain eventdistribution functions
from EventDistributions import eventDistributions
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "TwoDTimestamps": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "2D timestamps of first, median, mean event per pixel",
            "display_name": "2D timestamp plots of the candidate cluster"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def format_coord_timestamp(x, y, timedist, x_edges, y_edges):
    """
    Function that formats the coordinates of the mouse cursor in candidate preview xy timeplot
    """
    x_pix = round(x)
    y_pix = round(y)
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    time = timedist[y_bin, x_bin]*1e-3

    if time == np.nan:
        display = f'x={x_pix}, y={y_pix}'
    else:
        display = f'x={x_pix}, y={y_pix}, time={time:.2f} ms'
    return display

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def TwoDTimestamps(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    ax_first = figure.add_subplot(131)
    ax_median = figure.add_subplot(132)
    ax_mean = figure.add_subplot(133)
    ax_first.set_title("first event per pixel")
    ax_median.set_title("median event per pixel")
    ax_mean.set_title("mean event per pixel")
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.17)
    cb = None

    first = eventDistributions.FirstTimestamp(findingResult).dist2D
    median = eventDistributions.MedianTimestamp(findingResult).dist2D
    mean = eventDistributions.AverageTimestamp(findingResult).dist2D

    x_edges, y_edges = eventDistributions.Hist2d_xy(findingResult).x_edges, eventDistributions.Hist2d_xy(findingResult).y_edges

    # Plot the 2D histograms
    first_mesh = ax_first.pcolormesh(x_edges, y_edges, first*1e-3)
    ax_first.set_aspect('equal')
    ax_first.format_coord = lambda x,y:format_coord_timestamp(x,y,first, x_edges, y_edges)
    ax_median.pcolormesh(x_edges, y_edges, median*1e-3)
    ax_median.set_aspect('equal')
    ax_mean.pcolormesh(x_edges, y_edges, mean*1e-3)
    ax_mean.set_aspect('equal')
    ax_first.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    ax_mean.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    ax_median.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')

    # Add and set labels
    ax_first.set_xlabel('x [px]')
    ax_first.set_ylabel('y [px]')
    ax_median.set_xlabel('x [px]')
    ax_median.set_ylabel('y [px]')
    ax_mean.set_xlabel('x [px]')
    ax_mean.set_ylabel('y [px]')

    # Add or update colorbar
    if cb is None:
        divider = make_axes_locatable(ax_first)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = figure.colorbar(first_mesh, cax=cax)
        cb.set_label('time [ms]')
    else: 
        cb.update_normal(first_mesh)


    
    # required output none
    return 1
