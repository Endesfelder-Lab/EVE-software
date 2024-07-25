
try:
    from eve_smlm.Utils import utilsHelper
    from eve_smlm.TemporalFitting import timeFitting
except ImportError:
    from Utils import utilsHelper
    from TemporalFitting import timeFitting
import pandas as pd
import numpy as np
import logging

from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "AverageXYpos": {
            "time_kwarg" : {"base": "TemporalFits", "description": "Temporal fitting routine to get time estimate.", "default_option": "LognormCDFFirstEvents_weighted"},
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Gives the average xy-position of each found localization.",
            "display_name": "Mean X,Y position"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def localize_candidate(candidate, candidate_id, time_fit, pixel_size):
    mean_polarity = candidate['events']['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
    x = np.mean(candidate['events']['x'])* pixel_size #X position in nm
    del_x = np.std(candidate['events']['x'])* pixel_size #X position in nm
    y = np.mean(candidate['events']['y'])* pixel_size #Y position in nm
    del_y = np.std(candidate['events']['y'])* pixel_size #Y position in nm
    opt_loc = [(x/pixel_size-np.min(candidate['events']['x'])), (y/pixel_size-np.min(candidate['events']['y']))]
    t, del_t, t_fit_info, opt_t = time_fit(candidate['events'], opt_loc)
    loc_df = pd.DataFrame({'candidate_id': candidate_id, 'x': x, 'y': y, 'del_x': del_x, 'del_y': del_y, 'p': p, 't': t, 'del_t': del_t, 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][0], 'y_dim': candidate['cluster_size'][1], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': t_fit_info}, index=[0])
    return loc_df

# perform localization for part of candidate dictionary
def localize_canditates2D(i, candidate_dic, time_fit, pixel_size):
    logging.info('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    fails = pd.DataFrame()  # Initialize fails as an empty DataFrame
    fail = pd.DataFrame()
    for candidate_id in list(candidate_dic):
        localization = localize_candidate(candidate_dic[candidate_id], candidate_id, time_fit, pixel_size)
        localizations.append(localization)
        if localization['fit_info'][0] != '':
            fail['candidate_id'] = localization['candidate_id']
            fail['fit_info'] = localization['fit_info'].str.split(':').str[0][0]
            fails = pd.concat([fails, fail], ignore_index=True)
    if localizations == []:
        localizations = pd.DataFrame()
    else:
        localizations = pd.concat(localizations, ignore_index=True)
    logging.info('Localizing PSFs (thread '+str(i)+') done!')
    return localizations, fails

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def AverageXYpos(candidate_dic,settings,**kwargs):

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    time_fit = getattr(timeFitting, kwargs['time_kwarg'])()

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], time_fit, pixel_size) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    if len(localization_list) == 0:
        localization_list = [pd.DataFrame()]
    localizations = pd.concat(localization_list, ignore_index=True)

    fail_list = [res[1] for res in RES]
    if len(fail_list) == 0:
        fail_list = [pd.DataFrame()]
    fails = pd.concat(fail_list, ignore_index=True)
    
    # Fit performance information
    average_fit_info = utilsHelper.info(localizations, fails)

    return localizations, average_fit_info