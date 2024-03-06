import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
from scipy.ndimage import convolve
from EventDistributions import eventDistributions
from TemporalFitting import timeFitting
import logging

from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "RadialSym2D": {
            "time_kwarg" : {"base": "TemporalFits", "description": "Temporal fitting routine to get time estimate.", "default_option": "TwoDGaussianFirstTime"},
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Determines localization parameters via radial symmetry.",
            "display_name": "Radial Symmetry 2D"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def radialcenter(distfunc, candidateID, candidate, time_fit, pixel_size):
    I = distfunc(candidate['events']).dist2D
    # Number of grid points
    Ny, Nx = I.shape

    # grid coordinates are -n:n, where Nx (or Ny) = 2*n+1
    # grid midpoint coordinates are -n+0.5:n-0.5;
    # The two lines below replace
    #    xm = repmat(-(Nx-1)/2.0+0.5:(Nx-1)/2.0-0.5,Ny-1,1);
    # and are faster (by a factor of >15 !)
    # -- the idea is taken from the repmat source code
    xm_onerow = np.arange(-(Nx-1)/2.0+0.5, (Nx-1)/2.0-0.5+1)
    xm = np.tile(xm_onerow, (Ny-1, 1))
    # similarly replacing
    #    ym = repmat((-(Ny-1)/2.0+0.5:(Ny-1)/2.0-0.5)', 1, Nx-1);
    ym_onecol = np.arange(-(Ny-1)/2.0+0.5, (Ny-1)/2.0-0.5+1)[:, np.newaxis] #Note that y increases "downward"
    ym = np.tile(ym_onecol, (1, Nx-1))

    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    # Note that y increases "downward" (increasing row number) -- we'll deal
    # with this when calculating "m" below.
    dIdu = I[:-1, 1:] - I[1:, :-1]
    dIdv = I[:-1, :-1] - I[1:, 1:]

    # Smoothing --
    h = np.ones((3, 3)) / 9.0  #simple 3x3 averaging filter
    fdu = convolve(dIdu, h, mode='constant')
    fdv = convolve(dIdv, h, mode='constant')
    dImag2 = fdu**2 + fdv**2  # gradient magnitude, squared

    # Slope of the gradient .  Note that we need a 45 degree rotation of 
    # the u,v components to express the slope in the x-y coordinate system.
    # The negative sign "flips" the array to account for y increasing
    # "downward" 
    m = -(fdv + fdu) / (fdu - fdv)

    # *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
    # zero.  In this case, replace with the un-smoothed gradient.
    NNanm = np.sum(np.isnan(m))
    if NNanm > 0:
        unsmoothm = (dIdv + dIdu) / (dIdu - dIdv)
        m[np.isnan(m)] = unsmoothm[np.isnan(m)]

    # If it's still NaN, replace with zero. (Very unlikely.)
    NNanm = np.sum(np.isnan(m))
    if NNanm > 0:
        m[np.isnan(m)] = 0



    # Almost as rarely, an element of m can be infinite if the smoothed u and v
    # derivatives are identical.  To avoid NaNs later, replace these with some
    # large number -- 10x the largest non-infinite slope.  The sign of the
    # infinity doesn't matter
    try:
        m[np.isinf(m)] = 10 * np.max(m[~np.isinf(m)])
    except:
        # if this fails, it's because all the elements are infinite.  Replace
        # with the unsmoothed derivative.  There's probably a more elegant way
        # to do this.
        m = (dIdv + dIdu) / (dIdu - dIdv)



    # Shorthand "b", which also happens to be the
    # y intercept of the line of slope m that goes through each grid midpoint
    b = ym - m * xm

    # Weighting: weight by square of gradient magnitude and inverse 
    # distance to gradient intensity centroid.
    sdI2 = np.sum(dImag2)
    xcentroid = np.sum(np.sum(dImag2 * xm)) / sdI2
    ycentroid = np.sum(np.sum(dImag2 * ym)) / sdI2
    w = dImag2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # least-squares minimization to determine the translated coordinate
    # system origin (xc, yc) such that lines y = mx+b have
    # the minimal total distance^2 to the origin:
    # See function lsradialcenterfit (below)
    xc, yc = lsradialcenterfit(m, b, w)



    
    # Return output relative to upper left coordinate
    xc += (Nx -2) / 2.0
    yc += (Ny - 2) / 2.0

    # A rough measure of the particle width.
    # Not at all connected to center determination, but may be useful for tracking applications; 
    # could eliminate for (very slightly) greater speed
    Isub = I - np.min(I)
    px, py = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    xoffset = px - xc
    yoffset = py - yc
    r2 = xoffset**2 + yoffset**2
    sigma = np.sqrt(np.sum(np.sum(Isub * r2)) / np.sum(Isub)) / 2  # second moment is 2*Gaussian width

    mean_polarity = candidate['events']['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
    opt = [xc, yc]
    t, del_t, t_fit_info, opt_t = time_fit(candidate['events'], opt)
    xc = (xc + np.min(candidate['events']['x']))*pixel_size # in nm
    yc = (yc + np.min(candidate['events']['y']))*pixel_size # in nm

    loc_df = pd.DataFrame({'candidate_id': candidateID, 'x': xc, 'y': yc, 'p': p, 't': t, 'del_t': del_t, 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][0], 'y_dim': candidate['cluster_size'][1], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': t_fit_info}, index=[0])
    return loc_df


def lsradialcenterfit(m, b, w):
    # Least squares solution to determine the radial symmetry center

    # inputs m, b, w are defined on a grid
    # w are the weights for each point
    wm2p1 = w / (m**2 + 1)
    sw = np.sum(np.sum(wm2p1))
    smmw = np.sum(np.sum(m**2 * wm2p1))
    smw = np.sum(np.sum(m * wm2p1))
    smbw = np.sum(np.sum(m * b * wm2p1))
    sbw = np.sum(np.sum(b * wm2p1))
    det = smw**2 - smmw * sw
    xc = (smbw * sw - smw * sbw) / det  # relative to image center
    yc = (smbw * smw - smmw * sbw) / det  # relative to image center

    return xc, yc

# perform localization for part of candidate dictionary
def localize_canditates2D(i, candidate_dic, distfunc, time_fit, pixel_size):
    logging.info('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    fails = pd.DataFrame()  # Initialize fails as an empty DataFrame
    fail = pd.DataFrame()
    for candidate_id in list(candidate_dic):
        localization = radialcenter(distfunc, candidate_id, candidate_dic[candidate_id], time_fit, pixel_size)
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

# 2D radial symmetry
def RadialSym2D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    # maybe add a smoothing parameter here

    # Load the optional kwargs

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm

    distfunc = getattr(eventDistributions, 'Hist2d_xy')
    time_fit = getattr(timeFitting, kwargs['time_kwarg'])()

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], distfunc, time_fit, pixel_size) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list, ignore_index=True)

    fail_list = [res[1] for res in RES]
    fails = pd.concat(fail_list, ignore_index=True)
    
    # Fit performance information
    radSym_fit_info = utilsHelper.info(localizations, fails)

    return localizations, radSym_fit_info
