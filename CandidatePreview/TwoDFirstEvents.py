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
        "TwoDFirstEvents": {
            "required_kwargs": [
                {"name": "outlier_lower","display_text":"lower bound", "description": "lower bound of outlier removal (in %)", "default":"5"},
                {"name": "outlier_higher", "display_text":"higher bound", "description": "higher bound of outlier removal (in %)","default":"95"},
                ],
            "optional_kwargs": [
            ],
            "help_string": "2D time of first per pixel, outlier removed, weight",
            "display_name": "2D time of first event per pixel and candidate cluster"
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

def format_coord_weights(x, y, dist, x_edges, y_edges):
    """
    Function that formats the coordinates of the mouse cursor in candidate preview xy timeplot
    """
    x_pix = round(x)
    y_pix = round(y)
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    value = dist[y_bin, x_bin]

    if value == np.nan:
        display = f'x={x_pix}, y={y_pix}'
    else:
        display = f'x={x_pix}, y={y_pix}, sigma={value}'
    return display

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def TwoDFirstEvents(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    outlier_lower = float(kwargs['outlier_lower'])
    outlier_higher = float(kwargs['outlier_higher'])
    # Create a GridSpec with 1 row and 5 columns, where the 4th column is narrower for the colorbar
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    ax_first = figure.add_subplot(gs[0])
    ax_outlier = figure.add_subplot(gs[1])
    ax_weights = figure.add_subplot(gs[2])
    ax_cb = figure.add_subplot(gs[3])
    ax_first.set_title("first event per pixel", fontsize=10)
    ax_outlier.set_title("first event (outlier removed)", fontsize=10)
    ax_weights.set_title("number of events per pixel", fontsize=10)
    figure.subplots_adjust(bottom=0.17, left=0.02, right=0.950, wspace=0.05)

    firstTimes = eventDistributions.FirstTimestamp(findingResult)
    first = firstTimes.dist2D
    first_wNan = np.where(np.isnan(first), np.nanmedian(first), first)
    q1 = np.percentile(first_wNan, outlier_lower)
    q3 = np.percentile(first_wNan, outlier_higher)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the outlier range
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace outliers in first with NaN
    mask = (first_wNan < lower_bound) | (first_wNan > upper_bound)
    outlier = np.copy(first)
    outlier[mask] = np.nan
    weights = firstTimes.weights
    max_weight = np.nanmax(weights)
    sigma = (max_weight - weights + 1)/max_weight

    x_edges, y_edges = eventDistributions.Hist2d_xy(findingResult).x_edges, eventDistributions.Hist2d_xy(findingResult).y_edges

    # find global min and max times
    t_min = np.min([np.nanmin(first*1e-3), np.nanmin(outlier*1e-3)])
    t_max = np.max([np.nanmax(first*1e-3), np.nanmax(outlier*1e-3)])
    # Plot the 2D histograms
    first_mesh = ax_first.pcolormesh(x_edges, y_edges, first*1e-3, vmin=t_min, vmax=t_max)
    ax_first.set_aspect('equal')
    ax_first.format_coord = lambda x,y:format_coord_timestamp(x,y,first, x_edges, y_edges)
    ax_outlier.pcolormesh(x_edges, y_edges, outlier*1e-3, vmin=t_min, vmax=t_max)
    ax_outlier.set_aspect('equal')
    ax_outlier.format_coord = lambda x,y:format_coord_timestamp(x,y,outlier, x_edges, y_edges)
    ax_weights.pcolormesh(x_edges, y_edges, sigma)
    ax_weights.set_aspect('equal')
    ax_weights.format_coord = lambda x,y:format_coord_weights(x,y,sigma, x_edges, y_edges)
    ax_first.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    ax_outlier.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    ax_weights.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    
    # Add and set labels
    ax_first.set_xlabel('x [px]')
    ax_first.set_ylabel('y [px]')
    ax_outlier.set_xlabel('x [px]')
    ax_outlier.set_ylabel('y [px]')
    ax_weights.set_xlabel('x [px]')
    ax_weights.set_ylabel('y [px]')

    # Add or update colorbar
    axes_list = [ax_first, ax_outlier]
    cb = figure.colorbar(first_mesh, ax=axes_list, cax=ax_cb)
    cb.set_label('time [ms]')
    figure.tight_layout()
    figure.tight_layout()

    # required output none
    return 1
