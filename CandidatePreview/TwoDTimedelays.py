import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import logging
from matplotlib.gridspec import GridSpec
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
def TwoDTimedelay(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    ax_min = figure.add_subplot(gs[0])
    ax_max = figure.add_subplot(gs[1])
    ax_mean = figure.add_subplot(gs[2])
    ax_cb = figure.add_subplot(gs[3])
    ax_min.set_title("min delay per pixel", fontsize=10)
    ax_max.set_title("max delay per pixel", fontsize=10)
    ax_mean.set_title("mean delay per pixel", fontsize=10)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.17, left=0.02, right=0.950, wspace=0.05)

    min = eventDistributions.MinTimeDiff(findingResult).dist2D
    max = eventDistributions.MaxTimeDiff(findingResult).dist2D
    mean = eventDistributions.AverageTimeDiff(findingResult).dist2D

    x_edges, y_edges = eventDistributions.Hist2d_xy(findingResult).x_edges, eventDistributions.Hist2d_xy(findingResult).y_edges

    # find global min and max times
    t_min = np.min([np.nanmin(min*1e-3), np.nanmin(max*1e-3), np.nanmin(mean*1e-3)])
    t_max = np.max([np.nanmax(min*1e-3), np.nanmax(max*1e-3), np.nanmax(mean*1e-3)])
    # Plot the 2D histograms
    min_mesh = ax_min.pcolormesh(x_edges, y_edges, min*1e-3, vmin=t_min, vmax=t_max)
    ax_min.set_aspect('equal')
    ax_min.format_coord = lambda x,y:format_coord_timestamp(x,y,min, x_edges, y_edges)
    max_mesh = ax_max.pcolormesh(x_edges, y_edges, max*1e-3, vmin=t_min, vmax=t_max)
    ax_max.set_aspect('equal')
    ax_max.format_coord = lambda x,y:format_coord_timestamp(x,y,max, x_edges, y_edges)
    mean_mesh = ax_mean.pcolormesh(x_edges, y_edges, mean*1e-3, vmin=t_min, vmax=t_max)
    ax_mean.set_aspect('equal')
    ax_mean.format_coord = lambda x,y:format_coord_timestamp(x,y,mean, x_edges, y_edges)
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

    axes_list = [ax_min, ax_mean, ax_max]
    cb = figure.colorbar(mean_mesh, ax=axes_list, cax=ax_cb)
    cb.set_label('time [ms]')
    figure.tight_layout()
    figure.tight_layout()

    # required output none
    return 1
