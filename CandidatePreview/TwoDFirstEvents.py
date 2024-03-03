import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
from scipy import optimize
import logging
#Obtain eventdistribution functions
from EventDistributions import eventDistributions
from TemporalFitting import timeFitting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "TwoDFirstEvents": {
            "required_kwargs": [
                {"name": "show_fits","display_text":"show fits", "description": "Show 2D Gaussian fit.", "default":"True"}
            ],
            "optional_kwargs": [
                {"name": "use_weights", "display_text":"use weights", "description": "Weigh first events per pixel with number of events/pixel","default":"True"}
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
        display = f'x={x_pix}, y={y_pix}, sigma={value:.2f}'
    return display

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def TwoDFirstEvents(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    show_fits = utilsHelper.strtobool(kwargs['show_fits'])
    use_weights = utilsHelper.strtobool(kwargs['use_weights'])

    # Create a GridSpec with 1 row and 5 columns, where the 4th column is narrower for the colorbar
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax_first = figure.add_subplot(gs[0])
    ax_gaussfit = figure.add_subplot(gs[1], projection='3d')
    ax_weights = figure.add_subplot(gs[2])

    ax_first.set_title("time of first event/pixel", fontsize=10)
    ax_gaussfit.set_title("first events + 2D Gaussian fit", fontsize=10)
    ax_weights.set_title("sigma/pixel", fontsize=10)
    figure.subplots_adjust(bottom=0.17, left=0.02, right=0.950, wspace=0.05)

    firstTimes = eventDistributions.FirstTimestamp(findingResult)
    first = firstTimes.dist2D

    first_events = pd.DataFrame()
    first_events = firstTimes.get_smallest_t(findingResult)

    x_edges, y_edges = eventDistributions.Hist2d_xy(findingResult).x_edges, eventDistributions.Hist2d_xy(findingResult).y_edges

    # find global min and max times
    t_min = np.nanmin(first*1e-3)
    t_max = np.nanmax(first*1e-3)
    # Plot the 2D histograms
    first_mesh = ax_first.pcolormesh(x_edges, y_edges, first*1e-3, vmin=t_min, vmax=t_max) # , cmap='cividis'
    ax_first.set_aspect('equal')
    ax_first.format_coord = lambda x,y:format_coord_timestamp(x,y,first, x_edges, y_edges)
    ax_first.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red', label='Localization(s)')

    # scatter first events and do 2d Gaussian fit
    ax_gaussfit.scatter(first_events['x'], first_events['y'], first_events['t']*1e-3, color='C2')#, label='First events')
    
    gaussian_fit = timeFitting.TwoDGaussianFirstTime()
    opt_loc = [(fittingResult['x']/pixel_size-np.min(findingResult['x'])).iloc[0], (fittingResult['y']/pixel_size-np.min(findingResult['y'])).iloc[0]]
    t, del_t, fit_info, opt = gaussian_fit(findingResult, opt_loc, use_weights=use_weights)
    if show_fits == True and not np.isnan(opt[0]):
        x = np.linspace(0, gaussian_fit.fit.xlim, 20)
        y = np.linspace(0, gaussian_fit.fit.ylim, 20)
        X, Y = np.meshgrid(x, y)
        X.ravel()
        Y.ravel()
        fit = gaussian_fit.fit.func((X,Y), *opt)
        fit = fit.reshape(len(y), len(x))
        X += np.min(first_events['x'])
        Y += np.min(first_events['y'])
        X = X.reshape(len(y), len(x))
        Y = Y.reshape(len(y), len(x))

        ax_gaussfit.plot_surface(X,Y,fit, color='black', alpha=0.7, cmap='binary', zorder=-1)
        time_surface = ax_gaussfit.plot_surface(X,Y,np.ones_like(fit)*t, color='maroon', alpha=0.3, label='Fitted time')
        time_surface._edgecolors2d = time_surface._edgecolor3d
        time_surface._facecolors2d = time_surface._facecolor3d
    
    ax_gaussfit.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, fittingResult['t'], marker='x', c='red', label='Localization(s)')

    # Plot the sigma map
    weights_mesh = ax_weights.pcolormesh(x_edges, y_edges, gaussian_fit.fit.sigma_tot.reshape(gaussian_fit.fit.ylim, gaussian_fit.fit.xlim))#,cmap='cividis')
    ax_weights.set_aspect('equal')
    ax_weights.format_coord = lambda x,y:format_coord_weights(x,y,gaussian_fit.fit.sigma_tot.reshape(gaussian_fit.fit.ylim, gaussian_fit.fit.xlim), x_edges, y_edges)
    ax_weights.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, marker='x', c='red')
    
    # Add legend
    ax_gaussfit.legend(ncol=2, bbox_to_anchor=(1.5, -0.25))

    # Add and set labels
    ax_first.set_xlabel('x [px]')
    ax_first.set_ylabel('y [px]')
    ax_gaussfit.set_xlabel('x [px]')
    ax_gaussfit.set_ylabel('y [px]')
    ax_weights.set_xlabel('x [px]')
    ax_weights.set_ylabel('y [px]')
    ax_gaussfit.invert_zaxis()

    # Add or update colorbar
    divider = make_axes_locatable(ax_first)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb_first = figure.colorbar(first_mesh, cax=cax)
    cb_first.set_label('time [ms]')

    divider = make_axes_locatable(ax_weights)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb_weights = figure.colorbar(weights_mesh, cax=cax)
    cb_weights.set_label('sigma [a.u.]')

    figure.tight_layout()
    figure.tight_layout()

    # required output none
    return 1

