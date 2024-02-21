import inspect
from Utils import utilsHelper
from CandidateFitting.TemporalFitting import timeDistributions
from EventDistributions import eventDistributions
import pandas as pd
import numpy as np
import time, logging
from scipy import optimize
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)

from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Gaussian2D": {
            "dist_kwarg" : {"base": "XYDist", "description": "Two-dimensional event-distribution to run fitting on.", "default_option": "Hist2d_xy"},
            "required_kwargs": [
                {"name": "expected_width", "display_text":"expected width", "description": "Expected width of Gaussian fit (in nm)","default":150.},
                {"name": "fitting_tolerance", "display_text":"fitting tolerance", "description": "Discard localizations with uncertainties larger than this value times the pixel size. ","default":1.},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Makes a 2D gaussian fit (via least squares) to determine the localization parameters.",
            "display_name": "2D Gaussian"
        }, 
        "LogGaussian2D": {
            "dist_kwarg" : {"base": "XYDist", "description": "Two-dimensional event-distribution to run fitting on.", "default_option": "Hist2d_xy"},
            "required_kwargs": [
                {"name": "expected_width", "display_text":"expected width", "description": "Expected width of log-Gaussian fit (in nm)","default":150.},
                {"name": "fitting_tolerance", "display_text":"fitting tolerance", "description": "Discard localizations with uncertainties larger than this value times the pixel size. ","default":1.},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Makes a 2D log-gaussian fit (via least squares) to determine the localization parameters.",
            "display_name": "2D LogGaussian"
        }, 
        "Gaussian3D": {
            "dist_kwarg" : {"base": "XYDist", "description": "Two-dimensional event-distribution to run fitting on.", "default_option": "Hist2d_xy"},
            "required_kwargs": [
                {"name": "theta", "display_text":"rotation angle", "description": "Rotation angle (in degrees) of the Gaussian","default":0},
                {"name": "expected_width", "display_text":"expected width", "description": "Expected width of Gaussian fit (in nm)","default":150.},
                {"name": "fitting_tolerance", "display_text":"expected width", "description": "Discard localizations with uncertainties larger than this value times the pixel size. ","default":1.},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Makes a 2D gaussian fit with rotation angle theta to determine the 3D localization parameters.",
            "display_name": "3D Gaussian: astigmatism"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

class fit:

    def __init__(self, dist, candidateID):
        self.candidateID = candidateID
        self.ylim, self.xlim = dist.dist2D.shape
        self.xmean = np.average(np.arange(1,self.xlim+1), weights=np.nansum(dist.dist2D, axis=0))-1. 
        self.ymean = np.average(np.arange(1,self.ylim+1), weights=np.nansum(dist.dist2D, axis=1))-1.
        self.image = dist.dist2D.ravel()
        self.imstats = [np.nanmax(self.image), np.nanmedian(self.image)]
        self.mesh = self.meshgrid()
        self.fit_info = ''
    
    def meshgrid(self):
        x = np.arange(self.xlim)
        y = np.arange(self.ylim)
        X,Y = np.meshgrid(x,y)
        return X.ravel(),Y.ravel()
    
    def __call__(self, func, **kwargs):
        try:
            popt, pcov = optimize.curve_fit(func, self.mesh, self.image, nan_policy='omit', **kwargs) #, gtol=1e-4,ftol=1e-4
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError as warning:
            self.fit_info += 'RuntimeError: ' + str(warning)
            popt = np.full(6, np.nan)
            perr = np.full(6, np.nan)
        except ValueError as warning:
            self.fit_info += 'ValueError: ' + str(warning)
            popt = np.full(6, np.nan)
            perr = np.full(6, np.nan)
        except OptimizeWarning as warning:
            self.fit_info += 'OptimizeWarning: ' + str(warning)
            popt = np.full(6, np.nan)
            perr = np.full(6, np.nan)
        return popt, perr

# 2D gaussian fit
class gauss2D(fit):

    def __init__(self, dist, candidateID, width, fitting_tolerance, pixel_size):
        super().__init__(dist, candidateID)
        self.bounds = self.bounds()
        self.p0 = self.p0(width)
        self.fitting_tolerance = fitting_tolerance
        self.pixel_size = pixel_size

    def bounds(self):
        bounds = ([0., 0., 0., 0., 0., 0.], [self.xlim, self.ylim, np.inf, np.inf, np.inf, np.inf])
        return bounds
    
    def p0(self, width):
        p0 = (self.xmean, self.ymean, width, width, self.imstats[0], self.imstats[1])
        return p0
    
    def func(self, XY, x0, y0, sigma_x, sigma_y, amplitude, offset):
        X, Y = XY
        g = offset + amplitude * np.exp( - ((X-x0)**2/(2*sigma_x**2) + (Y-y0)**2/(2*sigma_y**2)))
        return g
    
    def __call__(self, events, **kwargs):
        opt, err = super().__call__(self.func, bounds=self.bounds, p0=self.p0, **kwargs)
        if self.fit_info != '':
            x = np.nan
            y = np.nan
            del_x = np.nan
            del_y = np.nan
            p = np.nan
            t = np.nan
        else: 
            x = (opt[0]+np.min(events['x']))*self.pixel_size # in nm
            y = (opt[1]+np.min(events['y']))*self.pixel_size # in nm
            del_x = err[0]*self.pixel_size # in nm
            del_y = err[1]*self.pixel_size # in nm
            if del_x > self.fitting_tolerance*self.pixel_size or del_y > self.fitting_tolerance*self.pixel_size:
                self.fit_info = 'ToleranceWarning: Fitting uncertainties exceed the tolerance.'
                x = np.nan
                y = np.nan
                del_x = np.nan
                del_y = np.nan
                t = np.nan
            else:
                time_fit = timeDistributions.rayleigh(events['t']*1e-3, bins='auto')
                time_fit_results = time_fit(events['t']*1e-3)
                t = time_fit_results[0]
            mean_polarity = events['p'].mean()
            p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
        loc_df = pd.DataFrame({'candidate_id': self.candidateID, 'x': x, 'y': y, 'del_x': del_x, 'del_y': del_y, 'p': p, 't': t, 'fit_info': self.fit_info}, index=[0])
        return loc_df

# 2d log gaussian fit
class loggauss2D(gauss2D):

    def __init__(self, dist, candidateID, width, fitting_tolerance, pixel_size):
        super().__init__(dist, candidateID, width, fitting_tolerance, pixel_size)
        self.update_p0()
    
    def update_p0(self):
        mod_p0 = list(self.p0)
        mod_p0[4] = np.exp(self.imstats[0])
        mod_p0[5] = np.exp(self.imstats[1])
        self.p0 = tuple(mod_p0)
    
    def func(self, XY, x0, y0, sigma_x, sigma_y, amplitude, offset):
        X, Y = XY
        g = np.log(offset + amplitude * np.exp( - ((X-x0)**2/(2*sigma_x**2) + (Y-y0)**2/(2*sigma_y**2))))
        return g
    
# 3d gaussian fit
class gauss3D(gauss2D):

    def __init__(self, dist, candidateID, width, fitting_tolerance, pixel_size, theta):
        super().__init__(dist, candidateID, width, fitting_tolerance, pixel_size)
        self.theta = theta

    # 2D Gaussian with rotation (angle theta in [rad])
    def func(self, XY, x0, y0, sigma_x, sigma_y, amplitude, offset):  
        X, Y = XY
        a = (np.cos(self.theta)**2)/(2*sigma_x**2) + (np.sin(self.theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*self.theta))/(4*sigma_x**2) + (np.sin(2*self.theta))/(4*sigma_y**2)
        c = (np.sin(self.theta)**2)/(2*sigma_x**2) + (np.cos(self.theta)**2)/(2*sigma_y**2)
        g = offset + amplitude * np.exp( - (a*((X-x0)**2) + 2*b*(X-x0)*(Y-y0) 
                                + c*((Y-y0)**2)))
        return g


# perform localization for part of candidate dictionary
def localize_canditates2D(i, candidate_dic, func, distfunc, *args, **kwargs):
    logging.info('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    fails = pd.DataFrame()  # Initialize fails as an empty DataFrame
    fail = pd.DataFrame()
    for candidate_id in list(candidate_dic):
        dist = distfunc(candidate_dic[candidate_id]['events'])
        fitting = func(dist, candidate_id, *args)
        localization = fitting(candidate_dic[candidate_id]['events'], **kwargs)
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

# 2D Gaussian
def Gaussian2D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    expected_width = float(kwargs['expected_width'])
    fitting_tolerance = float(kwargs['fitting_tolerance'])

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    expected_width = expected_width/pixel_size
    fit_func = gauss2D
    dist_func = getattr(eventDistributions, kwargs['dist_kwarg'])
    params = expected_width, fitting_tolerance, pixel_size

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], fit_func, dist_func, *params) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list, ignore_index=True)

    fail_list = [res[1] for res in RES]
    fails = pd.concat(fail_list, ignore_index=True)
    
    # Fit performance information
    gaussian_fit_info = utilsHelper.info(localizations, fails)

    return localizations, gaussian_fit_info

# 2D LogGaussian
def LogGaussian2D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    expected_width = float(kwargs['expected_width'])
    fitting_tolerance = float(kwargs['fitting_tolerance'])

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    expected_width = expected_width/pixel_size
    fit_func = loggauss2D
    dist_func = getattr(eventDistributions, kwargs['dist_kwarg'])
    params = expected_width, fitting_tolerance, pixel_size

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], fit_func, dist_func, *params) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list)

    fail_list = [res[1] for res in RES]
    fails = pd.concat(fail_list)
    
    # Fit performance information
    gaussian_fit_info = utilsHelper.info(localizations, fails)

    return localizations, gaussian_fit_info

# ToDo: Include calibration file to transform sigma_x/sigma_y to z height
def Gaussian3D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    expected_width = float(kwargs['expected_width'])
    fitting_tolerance = float(kwargs['fitting_tolerance'])

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    expected_width = expected_width/pixel_size
    theta = np.radians(float(kwargs['theta']))
    fit_func = gauss3D
    dist_func = getattr(eventDistributions, kwargs['dist_kwarg'])
    params = expected_width, fitting_tolerance, pixel_size, theta

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], fit_func, dist_func, *params) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list)

    fail_list = [res[1] for res in RES]
    fails = pd.concat(fail_list)
    
    # Fit performance information
    gaussian_fit_info = utilsHelper.info(localizations, fails)

    return localizations, gaussian_fit_info