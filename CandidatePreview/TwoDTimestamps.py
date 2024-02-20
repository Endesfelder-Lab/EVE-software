import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import logging
#Obtain eventdistribution functions
from EventDistributions import eventDistributions
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

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
    # Create a GridSpec with 1 row and 5 columns, where the 4th column is narrower for the colorbar
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    ax_first = figure.add_subplot(gs[0])
    ax_median = figure.add_subplot(gs[1])
    ax_mean = figure.add_subplot(gs[2])
    ax_cb = figure.add_subplot(gs[3])
    ax_first.set_title("first event per pixel", fontsize=10)
    ax_median.set_title("median event per pixel", fontsize=10)
    ax_mean.set_title("mean event per pixel", fontsize=10)
    figure.subplots_adjust(bottom=0.17, left=0.02, right=0.950, wspace=0.05)

    first = eventDistributions.FirstTimestamp(findingResult).dist2D
    median = eventDistributions.MedianTimestamp(findingResult).dist2D
    mean = eventDistributions.AverageTimestamp(findingResult).dist2D

    x_edges, y_edges = eventDistributions.Hist2d_xy(findingResult).x_edges, eventDistributions.Hist2d_xy(findingResult).y_edges

    # find global min and max times
    t_min = np.min([np.nanmin(first*1e-3), np.nanmin(median*1e-3), np.nanmin(mean*1e-3)])
    t_max = np.max([np.nanmax(first*1e-3), np.nanmax(median*1e-3), np.nanmax(mean*1e-3)])
    # Plot the 2D histograms
    first_mesh = ax_first.pcolormesh(x_edges, y_edges, first*1e-3, vmin=t_min, vmax=t_max)
    ax_first.set_aspect('equal')
    ax_first.format_coord = lambda x,y:format_coord_timestamp(x,y,first, x_edges, y_edges)
    median_mesh = ax_median.pcolormesh(x_edges, y_edges, median*1e-3, vmin=t_min, vmax=t_max)
    ax_median.set_aspect('equal')
    ax_median.format_coord = lambda x,y:format_coord_timestamp(x,y,median, x_edges, y_edges)
    mean_mesh = ax_mean.pcolormesh(x_edges, y_edges, mean*1e-3, vmin=t_min, vmax=t_max)
    ax_mean.set_aspect('equal')
    ax_mean.format_coord = lambda x,y:format_coord_timestamp(x,y,mean, x_edges, y_edges)
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
    axes_list = [ax_first, ax_median, ax_mean]
    cb = figure.colorbar(first_mesh, ax=axes_list, cax=ax_cb)
    cb.set_label('time [ms]')
    figure.tight_layout()
    figure.tight_layout()

    # required output none
    return 1
