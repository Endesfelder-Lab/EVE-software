import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import logging
#from scipy import optimize
import warnings
#from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", UserWarning)
#from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Spline1D": {
            "required_kwargs": [
                 {"name": "smoothing_factor", "display_text":"smoothing factor", "description": "Smoothing factor controls interplay of smoothness and approximation quality of the fit, increasing leads to smoother fits ","default":0.07},
                 {"name": "localization_sampling", "display_text":"temporal sampling", "description": "Time intervals to sample localizations from spline fit (in ms)","default":10.},
            ],
            "optional_kwargs": [
                {"name": "degree", "description": "Degree of smoothing spline, must be in the range 1 to 5.", "default":3},
            ],
            "help_string": "Makes a 1D spline fit to determin the localization parameters.",
            "display_name": "Tracking: 1D Spline"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def candidate_spline(candidate, candidate_id, smoothing_factor, localization_sampling, pixel_size, **kwargs):
    candidate['events'] = candidate['events'].sort_values(by=['t'])
    s = smoothing_factor * len(candidate['events']['x'])
    mean_polarity = candidate['events']['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
    try:    
        spx = UnivariateSpline(candidate['events']['t'], candidate['events']['x'], s=s, **kwargs)
        spy = UnivariateSpline(candidate['events']['t'], candidate['events']['y'], s=s, **kwargs)
        t = np.arange(np.min(candidate['events']['t']), np.max(candidate['events']['t']), localization_sampling*1000.)
        if t[-1] < np.max(candidate['events']['t']):
            t = np.append(t, np.max(candidate['events']['t']))
        locx = spx(t) * pixel_size # in nm
        locy = spy(t) * pixel_size # in nm
        residual_x = spx.get_residual() * pixel_size # in nm
        residual_y = spy.get_residual() * pixel_size # in nm
        spline_fit_info = ''
        localization_id = np.arange(0, len(t), 1)
        loc_df = pd.DataFrame({'candidate_id': candidate_id, 'localization_id': localization_id, 'x': locx, 'y': locy, 'residual_x': residual_x, 'residual_y': residual_y, 'p': p, 't': t/1000., 'fit_info': spline_fit_info})
    except UserWarning as warning:
        spline_fit_info = 'UserWarning: ' + str(warning)
        localization_id = 0.
        locx = np.nan
        locy = np.nan
        residual_x = np.nan
        residual_y = np.nan
        t = np.nan
        loc_df = pd.DataFrame({'candidate_id': candidate_id, 'localization_id': localization_id, 'x': locx, 'y': locy, 'residual_x': residual_x, 'residual_y': residual_y, 'p': p, 't': t, 'fit_info': spline_fit_info}, index=[0])
    return loc_df


def spline_fit_candidates(i, candidate_dic, smoothing_factor, localization_sampling, pixel_size, **kwargs):
    print('Fitting splines (thread ' + str(i) + ')...')
    localizations = []
    fails = pd.DataFrame()  # Initialize fails as an empty DataFrame
    fail = pd.DataFrame()
    for candidate_id, candidate in candidate_dic.items():
        localization = candidate_spline(candidate, candidate_id, smoothing_factor, localization_sampling, pixel_size, **kwargs)
        localizations.append(localization)
        if localization['fit_info'][0] != '':
            fail['candidate_id'] = localization['candidate_id']
            fail['fit_info'] = localization['fit_info'].str.split(':').str[0][0]
            fails = pd.concat([fails, fail], ignore_index=True)
    if localizations == []:
        localizations = pd.DataFrame()
    else:
        localizations = pd.concat(localizations, ignore_index=True)
    print('Fitting splines (thread ' + str(i) + ') done!')
    return localizations, fails


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

# 1D Spline fit of event "snake" (left by a moving particle)
def Spline1D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    smoothing_factor = float(kwargs['smoothing_factor'])
    localization_sampling = float(kwargs['localization_sampling'])

    # Load the optional kwargs
    k = int(kwargs['degree'])

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(spline_fit_candidates)(i, data_split[i], smoothing_factor, localization_sampling, pixel_size, k=k) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list, ignore_index=True)

    fail_list = [res[1] for res in RES]
    fails = pd.concat(fail_list, ignore_index=True)
    
    # Fit performance information
    spline_fit_info = utilsHelper.info(localizations, fails)

    return localizations, spline_fit_info