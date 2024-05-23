import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import fit
import logging
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
#Obtain eventdistribution functions
from EventDistributions import eventDistributions
from TemporalFitting import timeFitting

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "OneDProjection": {
            "required_kwargs": [
                {"name": "t_bin_width","display_text":"temporal bin width", "description": "Temporal bin width (in ms)", "default":"1"},
                {"name": "show_first", "display_text":"show first events", "description": "Plot the first events per pixel","default":"False"},
            ],
            "optional_kwargs": [
                {"name": "weigh_first", "display_text":"weigh first events", "description": "Weigh first events per pixel with number of events/pixel","default":"False"}
            ],
            "help_string": "Draws temporal trace of the candidate cluster.",
            "display_name": "Temporal trace of candidate cluster"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def OneDProjection(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    if not kwargs['t_bin_width'].isdigit():
        bins = str(kwargs['t_bin_width'])
        t_bin_width = None
    else:
        t_bin_width = float(kwargs['t_bin_width'])
    show_first = utilsHelper.strtobool(kwargs['show_first'])
    weigh_first = utilsHelper.strtobool(kwargs['weigh_first'])

    ax = figure.add_subplot(111)
    figure.tight_layout()
    figure.subplots_adjust(top=0.95,bottom=0.190,left=0.080)

    if t_bin_width is not None:
        hist_t = eventDistributions.Hist1d_t(findingResult, t_bin_width=t_bin_width)
        hist_t_dist, hist_t_edges = hist_t.dist1D, hist_t.t_edges
    else: 
        hist_t_dist, hist_t_edges = np.histogram(findingResult['t'].values*1e-3, bins=bins)
        t_bin_width = hist_t_edges[1]-hist_t_edges[0]
    
    t_min = None
    text = ''

    # plot 1D histogram and fit
    # all events
    if len(findingResult[findingResult['p'] == 0]) != 0 and len(findingResult[findingResult['p'] == 1]) != 0:
        ax.bar(hist_t_edges[:-1], hist_t_dist, width=t_bin_width,  label='Histogram (mix)', color='olive', alpha=0.5, align='edge')

    # positiv events
    elif len(findingResult[findingResult['p'] == 1]) != 0 and len(findingResult[findingResult['p'] == 0]) == 0:
        ax.bar(hist_t_edges[:-1], hist_t_dist, width=t_bin_width,  label='Histogram (pos)', color='C0', alpha=0.5, align='edge')

    # negative events
    else: # len(findingResult[findingResult['p'] == 0]) != 0:
        ax.bar(hist_t_edges[:-1], hist_t_dist, width=t_bin_width,  label='Histogram (neg)', color='C1', alpha=0.5, align='edge')

    # first events
    if show_first==True:
        if weigh_first!=True:
            # get first events
            firstTimes = eventDistributions.FirstTimestamp(findingResult)
            first_events = firstTimes.get_smallest_t(findingResult)
            first_events = first_events.sort_values(by='t')
            hist_t_first, t_edges = np.histogram(first_events['t'].values*1e-3, bins=hist_t_edges)
            
            # Add a zero bin to the beginning and end of hist_t_first for plot with step function
            hist_t_first = np.insert(hist_t_first, 0, 0)  # Add a zero bin at the beginning
            hist_t_first = np.append(hist_t_first, 0)  # Add a zero bin at the end
            t_edges = np.insert(t_edges, 0, t_edges[0]-(t_edges[1]-t_edges[0]))  # Add a zero bin at the beginning
            t_edges = np.append(t_edges, t_edges[-1]+(t_edges[1]-t_edges[0]))  # Add a zero bin at the end

            bincentres = (t_edges[1:]-t_edges[:-1])/2.+t_edges[:-1] 
            ax.step(bincentres,hist_t_first,where='mid',color='C2', label='Histogram (first)',linewidth=1.5)
            
        else:
            # get first events
            firstTimes = eventDistributions.FirstTimestamp(findingResult)
            first_events = firstTimes.get_smallest_t(findingResult)
            first_events = first_events.sort_values(by='t')
            hist_t_first, t_edges = np.histogram(first_events['t']*1e-3, bins=hist_t_edges, weights=first_events['weight'])
            
            # Add a zero bin to the beginning and end of hist_t_first for plot with step function
            hist_t_first = np.insert(hist_t_first, 0, 0)  # Add a zero bin at the beginning
            hist_t_first = np.append(hist_t_first, 0)  # Add a zero bin at the end
            t_edges = np.insert(t_edges, 0, t_edges[0]-(t_edges[1]-t_edges[0]))  # Add a zero bin at the beginning
            t_edges = np.append(t_edges, t_edges[-1]+(t_edges[1]-t_edges[0]))  # Add a zero bin at the end

            bincentres = (t_edges[1:]-t_edges[:-1])/2.+t_edges[:-1] 
            ax.step(bincentres,hist_t_first,where='mid',color='C2', label='Histogram (first, weighted)',linewidth=1.5)

    for index, row in fittingResult.iloc[:-1].iterrows():
        ax.axvline(x=row['t'], color='red')
    ax.axvline(x=fittingResult.iloc[-1]['t'], color='red', label='Localization(s)')
    if not t_min==None:
        t_min = np.min([fittingResult.iloc[0]['t'], t_min])
    else:
        t_min = fittingResult.iloc[0]['t']

    if text != '':
        props = dict(boxstyle='round', facecolor='white', edgecolor='None', alpha=0.8)
        figure.text(0.05, 0.05, s=text, fontsize=10, color='red', verticalalignment='bottom', horizontalalignment='left', bbox=props, wrap=True)

    # Add and set labels
    if not t_min==None:
        ax.set_xlim(np.nanmin([hist_t_edges[0], t_min-2]), hist_t_edges[-1])
    else:
        ax.set_xlim(hist_t_edges[0], hist_t_edges[-1])
   
    ax.set_xlabel('t [ms]')
    ax.set_ylabel('number of events')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    
    # required output none
    return 1
