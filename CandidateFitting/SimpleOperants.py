
try:
    from eve_smlm.Utils import utilsHelper
    from eve_smlm.EventDistributions import eventDistributions
    from eve_smlm.TemporalFitting import timeFitting
except ImportError:
    from Utils import utilsHelper
    from EventDistributions import eventDistributions
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
            "dist_kwarg" : {"base": "XYDist", "description": "Two-dimensional event-distribution to fit to get x,y-localization.", "default_option": "Hist2d_xy"},
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

# return the weighted average and standard deviation
def weighted_avg_and_std(dist_shape, weights, min_value, pixel_size):
    average = np.average(np.arange(1,dist_shape+1), weights=weights)
    variance = np.average((np.arange(1,dist_shape+1)-average)**2, weights=weights)
    average = (average-1.+min_value)*pixel_size
    std = np.sqrt(variance)*pixel_size
    return (average, std)

def localize_candidate(candidate, candidate_id, time_fit, pixel_size, distfunc):
    dist = distfunc(candidate['events']).dist2D
    if np.isnan(dist).all():
        xmean = np.nan
        ymean = np.nan
        xstd = np.nan
        ystd = np.nan
        t = np.nan
        del_t = np.nan
        fit_info = 'ValueError: No data to fit.'
    else:
        xmean, xstd = weighted_avg_and_std(dist.shape[1], np.nansum(dist, axis=0), np.min(candidate['events']['x']), pixel_size) # = (np.average(np.arange(1,dist.shape[1]+1), weights=np.nansum(dist, axis=0))-1.+np.min(candidate['events']['x']))*pixel_size # sum in y-direction
        ymean, ystd = weighted_avg_and_std(dist.shape[0], np.nansum(dist, axis=1), np.min(candidate['events']['y']), pixel_size) # (np.average(np.arange(1,dist.shape[0]+1), weights=np.nansum(dist, axis=1))-1.+np.min(candidate['events']['y']))*pixel_size # sum in x-direction
        opt_loc = [(xmean/pixel_size-np.min(candidate['events']['x'])), (ymean/pixel_size-np.min(candidate['events']['y']))]
        t, del_t, fit_info, opt_t = time_fit(candidate['events'], opt_loc)
    mean_polarity = candidate['events']['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
    loc_df = pd.DataFrame({'candidate_id': candidate_id, 'x': xmean, 'y': ymean, 'del_x': xstd, 'del_y': ystd, 'p': p, 't': t, 'del_t': del_t, 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][1], 'y_dim': candidate['cluster_size'][0], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': fit_info}, index=[0])
    return loc_df

# perform localization for part of candidate dictionary
def localize_canditates2D(i, candidate_dic, time_fit, pixel_size, distfunc):
    logging.info('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    fails = pd.DataFrame()  # Initialize fails as an empty DataFrame
    fail = pd.DataFrame()
    for candidate_id in list(candidate_dic):
        localization = localize_candidate(candidate_dic[candidate_id], candidate_id, time_fit, pixel_size, distfunc)
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
    dist_func = getattr(eventDistributions, kwargs['dist_kwarg'])
    time_fit = getattr(timeFitting, kwargs['time_kwarg'])()

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs) 
    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], time_fit, pixel_size, dist_func) for i in range(len(data_split)))
    
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