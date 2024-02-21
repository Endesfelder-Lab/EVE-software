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
        "ThreeDPointCloud": {
            "required_kwargs": [
                {"name": "show_first", "display_text":"show first events", "description": "Label the first events per pixel","default":"False"},
                {"name": "show_surrounding", "display_text":"show surrounding", "description": "Show surrounding events if set to true, only supported in preview mode","default":"False"},
            ],
            "optional_kwargs": [
                {"name": "xy_padding", "display_text":"xy-padding", "description": "Padding in x and y direction (in px)", "default":"0"},
                {"name": "t_padding", "display_text":"t-padding", "description": "Padding in t direction (in ms)", "default":"0"}
            ],
            "help_string": "Draws a 3D point cloud of the candidate cluster.",
            "display_name": "3D point cloud of the candidate cluster"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def get_surrounding(events, candidate_events, x_padding, y_padding, t_padding):
    xlim = [np.min(candidate_events['x'])-x_padding, np.max(candidate_events['x'])+x_padding]
    ylim = [np.min(candidate_events['y'])-y_padding, np.max(candidate_events['y'])+y_padding]
    tlim = [np.min(candidate_events['t'])-t_padding*1e3, np.max(candidate_events['t'])+t_padding*1e3]
    mask = ((events['x'] >= xlim[0]) & (events['x'] <= xlim[1]) & (events['y'] >= ylim[0]) & (events['y'] <= ylim[1]) & (events['t'] >= tlim[0]) & (events['t'] <= tlim[1]))
    all_events = pd.DataFrame(events[mask])
    surrounding = all_events.merge(candidate_events, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    return surrounding


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def ThreeDPointCloud(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    show_first = utilsHelper.strtobool(kwargs['show_first'])
    plot_surrounding = utilsHelper.strtobool(kwargs['show_surrounding'])
    xy_padding = int(kwargs['xy_padding'])
    t_padding = float(kwargs['t_padding'])

    surrounding = pd.DataFrame()
    if len(previewEvents) != 0 and plot_surrounding==True:
        surrounding = get_surrounding(previewEvents, findingResult, xy_padding, xy_padding, t_padding)

    eventsFiltered = findingResult
    first_events = pd.DataFrame()
    if show_first==True:
        first_events = eventDistributions.FirstTimestamp(eventsFiltered).get_smallest_t(eventsFiltered)
        eventsFiltered = eventsFiltered.merge(first_events, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    # figure.suptitle("3D pointcloud of candidate cluster")
    ax = figure.add_subplot(111, projection='3d')
    ax.tick_params(axis="y", pad=0.5)
    ax.tick_params(axis="z", pad=0.5)
    ax.tick_params(axis="x", pad=0.5)
    figure.tight_layout()
    figure.subplots_adjust(top=1.0,bottom=0.140)

    pos_events = eventsFiltered[eventsFiltered['p'] == 1]
    neg_events = eventsFiltered[eventsFiltered['p'] == 0]
    # Do a 3d scatterplot of the event data
    if not len(first_events)==0:
        ax.scatter(first_events['x'], first_events['y'], first_events['t']*1e-3, label='First events', color='C2')
    if not len(pos_events)==0:
        ax.scatter(pos_events['x'], pos_events['y'], pos_events['t']*1e-3, label='Positive events', color='C0')
    if not len(neg_events)==0:
        ax.scatter(neg_events['x'], neg_events['y'], neg_events['t']*1e-3, label='Negative events', color='C1')
    if not len(surrounding)==0:
        ax.scatter(surrounding['x'], surrounding['y'], surrounding['t']*1e-3, label='Surrounding events', color='black')
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    ax.set_zlabel('t [ms]')
    ax.invert_zaxis()

    # Plot the localization(s) of the candidate
    ax.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size, fittingResult['t'], marker='x', c='red', label='Localization(s)')
    ax.legend(loc='upper right', bbox_to_anchor=(2.0, 1))

    
    # required output none
    return 1
