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
                 {"name": "smoothing_factor", "description": "Smoothing factor controls interplay of smoothness and approximation quality of the fit, increasing leads to smoother fits ","default":0.07},
                 {"name": "localization_sampling", "description": "Time intervals to sample localizations from spline fit (in ms)","default":10.},
            ],
            "optional_kwargs": [
                {"name": "degree", "description": "Degree of smoothing spline, must be in the range 1 to 5.", "default":3},
                {"name": "multithread","description": "True to use multithread parallelization; False not to.","default":True},
            ],
            "help_string": "Makes a 1D spline fit to determin the localization parameters.",
            "display_name": "1D Spline"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

# calculate number of jobs on CPU
def nb_jobs(candidate_dic, num_cores):
    nb_candidates = len(candidate_dic)
    if nb_candidates < num_cores or num_cores == 1:
        njobs = 1
        num_cores = 1
    elif nb_candidates/num_cores > 100:
        njobs = np.int64(np.ceil(nb_candidates/100.))
    else:
        njobs = num_cores
    return njobs, num_cores

def get_subdictionary(main_dict, start_key, end_key):
    sub_dict = {k: main_dict[k] for k in range(start_key, end_key + 1) if k in main_dict}
    return sub_dict

# Slice data to distribute the computation on several cores
def slice_data(candidate_dic, nb_slices):
    slice_size=1.*len(candidate_dic)/nb_slices
    slice_size=np.int64(np.ceil(slice_size))
    data_split=[]
    last_key = list(candidate_dic.keys())[-1]
    for k in np.arange(nb_slices):
        keys = [k*slice_size, min((k+1)*slice_size-1,last_key)]
        data_split.append(get_subdictionary(candidate_dic, keys[0], keys[1]))
    return data_split

def candidate_spline(candidate, candidate_id, smoothing_factor, localization_sampling, pixel_size, **kwargs):
    candidate['events'] = candidate['events'].sort_values(by=['t'])
    s = smoothing_factor * len(candidate['events']['x'])
    try:    
        spx = UnivariateSpline(candidate['events']['t'], candidate['events']['x'], s=s, **kwargs)
        spy = UnivariateSpline(candidate['events']['t'], candidate['events']['y'], s=s, **kwargs)
        t = np.arange(np.min(candidate['events']['t']), np.max(candidate['events']['t']), localization_sampling*1000.)
        locx = spx(t) * pixel_size # in nm
        locy = spy(t) * pixel_size # in nm
        residual_x = spx.get_residual() * pixel_size # in nm
        residual_y = spy.get_residual() * pixel_size # in nm
        spline_fit_info = ''
        localization_id = np.arange(0, len(t), 1)
        mean_polarity = candidate['events']['p'].mean()
        p = np.ones(len(t))*(int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2)
        loc_df = pd.DataFrame({'candidate_id': candidate_id, 'localization_id': localization_id, 'x': locx, 'y': locy, 'residual_x': residual_x, 'residual_y': residual_y, 'p': p, 't': t/1000.})
    except UserWarning as warning:
        spline_fit_info = 'Candidate cluster ' + str(candidate_id) + ' could not be fitted, a warning occurred:'
        spline_fit_info += str(warning)
        spline_fit_info += '\n\n'
        loc_df = pd.DataFrame()
    return loc_df, spline_fit_info


def spline_fit_candidates(i, candidate_dic, smoothing_factor, localization_sampling, pixel_size, **kwargs):
    print('Fitting splines (thread ' + str(i) + ')...')
    dfs = []
    fitting_info = ''
    for candidate_id, candidate in candidate_dic.items():
        loc_df, spline_fit_info = candidate_spline(candidate, candidate_id, smoothing_factor, localization_sampling, pixel_size, **kwargs)
        dfs.append(loc_df)
        fitting_info += spline_fit_info
    if dfs == []:
        df = pd.DataFrame()
    else:
        df = pd.concat(dfs, ignore_index=True)
    print('Fitting splines (thread ' + str(i) + ') done!')
    return df, fitting_info


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

# 2D Gaussian
def Spline1D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    smoothing_factor = float(kwargs['smoothing_factor'])
    localization_sampling = float(kwargs['localization_sampling'])

    # Load the optional kwargs
    k = int(kwargs['degree'])
    multithread = utilsHelper.strtobool(kwargs['multithread'])

    # Initializations - general
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = nb_jobs(candidate_dic, num_cores)
    data_split = slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(spline_fit_candidates)(i, data_split[i], smoothing_factor, localization_sampling, pixel_size, k=k) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list)
    
    # Fit performance information
    nb_fails = len(candidate_dic)-len(localizations['candidate_id'].unique())
    spline_fit_info = f'Spline fitting failed for {nb_fails} candidate cluster(s).'
    logging.info(spline_fit_info)
    spline_fit_info += ''.join([res[1] for res in RES])

    return localizations, spline_fit_info