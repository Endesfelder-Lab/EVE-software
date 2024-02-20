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
        "TwoDProjection": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
                {"name": "t_bin_width", "description": "Padding in t direction (in ms)", "default":"10"}
            ],
            "help_string": "Draws xy, xt and yt projections of the candidate cluster.",
            "display_name": "2D projections of candidate cluster"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------
def format_coord_projectionxy(x, y, pos_hist, neg_hist, x_edges, y_edges):
    """
    Function that formats the coordinates of the mouse cursor in candidate preview xy projection
    """
    x_pix = round(x)
    y_pix = round(y)
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    pos = int(pos_hist[x_bin, y_bin])
    neg = int(neg_hist[x_bin, y_bin])

    display = f'x={x_pix}, y={y_pix}, events[pos,neg]=[{pos}, {neg}]'
    return display

def format_coord_projectiontx(x, y, pos_hist, neg_hist, t_edges, x_edges):
    """
    Function that formats the coordinates of the mouse cursor in candidate preview xt projection
    """
    time = round(x)
    x_pix = round(y)
    x_bin = np.digitize(x, t_edges) - 1
    y_bin = np.digitize(y, x_edges) - 1
    pos = int(pos_hist[x_bin, y_bin])
    neg = int(neg_hist[x_bin, y_bin])

    display = f't={time:.2f} ms, x={x_pix}, events[pos,neg]=[{pos}, {neg}]'
    return display

def format_coord_projectionty(x, y, pos_hist, neg_hist, t_edges, y_edges):
    """
    Function that formats the coordinates of the mouse cursor in candidate preview yt projection
    """
    time = round(x)
    y_pix = round(y)
    x_bin = np.digitize(x, t_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    pos = int(pos_hist[x_bin, y_bin])
    neg = int(neg_hist[x_bin, y_bin])

    display = f't={time:.2f} ms, y={y_pix}, events[pos,neg]=[{pos}, {neg}]'
    return display
#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def TwoDProjection(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    t_bin_width = float(kwargs['t_bin_width'])

    ax_xy = figure.add_subplot(121)
    ax_xt = figure.add_subplot(222)
    ax_yt = figure.add_subplot(224)
    # ax.tick_params(axis="y", pad=0.5)
    # ax.tick_params(axis="x", pad=0.5)
    figure.tight_layout()
    figure.subplots_adjust(top=0.955,bottom=0.190)

    hist_xy = eventDistributions.Hist2d_xy(findingResult)
    hist_tx = eventDistributions.Hist2d_tx(findingResult, t_bin_width=t_bin_width)
    hist_ty = eventDistributions.Hist2d_ty(findingResult, t_bin_width=t_bin_width)

    x_edges, y_edges, t_edges = hist_xy.x_edges, hist_xy.y_edges, hist_tx.x_edges

    # Compute the 2D histograms (pos)
    hist_xy_pos = hist_xy(findingResult[findingResult['p'] == 1])[0]
    hist_tx_pos = hist_tx(findingResult[findingResult['p'] == 1])[0]
    hist_ty_pos = hist_ty(findingResult[findingResult['p'] == 1])[0]
    # Compute the 2D histograms (neg)
    hist_xy_neg = hist_xy(findingResult[findingResult['p'] == 0])[0]
    hist_tx_neg = hist_tx(findingResult[findingResult['p'] == 0])[0]
    hist_ty_neg = hist_ty(findingResult[findingResult['p'] == 0])[0]

    # Set goodlooking aspect ratio depending on nr of xyt-bins
    aspectty = 3. * (len(t_edges)-1) / (len(y_edges)-1)
    aspecttx = 3. * (len(t_edges)-1) / (len(x_edges)-1)

    # Plot the 2D histograms
    ax_xy.pcolormesh(x_edges, y_edges, hist_xy.dist2D)
    ax_xy.set_aspect('equal')
    ax_xy.format_coord = lambda x,y:format_coord_projectionxy(x,y,hist_xy_pos.T, hist_xy_neg.T, x_edges, y_edges)
    ax_xt.pcolormesh(t_edges, x_edges, hist_tx.dist2D)
    ax_xt.set_aspect(aspecttx)
    ax_xt.format_coord = lambda x,y:format_coord_projectiontx(x,y,hist_tx_pos.T, hist_tx_neg.T, t_edges, x_edges)
    ax_yt.pcolormesh(t_edges, y_edges, hist_ty.dist2D)
    ax_yt.set_aspect(aspectty)
    ax_yt.format_coord = lambda x,y:format_coord_projectionty(x,y,hist_ty_pos.T, hist_ty_neg.T, t_edges, y_edges)
    ax_xy.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    ax_xt.plot(fittingResult['t'], fittingResult['x']/pixel_size, marker='x', c='red')
    ax_yt.plot(fittingResult['t'], fittingResult['y']/pixel_size, marker='x', c='red')

    # Add and set labels
    ax_xy.set_xlabel('x [px]')
    ax_xy.set_ylabel('y [px]')
    ax_xt.set_ylabel('x [px]')
    ax_yt.set_ylabel('y [px]')
    ax_yt.set_xlabel('t [ms]')
    
    # required output none
    return 1
