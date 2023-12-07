import inspect
from Utils import utilsHelper
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
            "required_kwargs": [
                 {"name": "expected_width", "description": "Expected width of Gaussian fit (in nm)","default":150.},
                 {"name": "fitting_tolerance", "description": "Discard localizations with uncertainties larger than this value times the pixel size. ","default":1.},
            ],
            "optional_kwargs": [
                {"name": "multithread","description": "True to use multithread parallelization; False not to.","default":True},
            ],
            "help_string": "Makes a 2D gaussian fit (via least squares) to determine the localization parameters.",
            "display_name": "2D Gaussian"
        }, 
        "Gaussian3D": {
            "required_kwargs": [
                {"name": "theta", "description": "Rotation angle (in degrees) of the Gaussian","default":0},
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

    def __init__(self, events, candidateID):
        self.candidateID = candidateID
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.ylim = [np.min(events['y']), np.max(events['y'])]
        self.xstats = [np.mean(events['x']), np.std(events['x'])]
        self.ystats = [np.mean(events['y']), np.std(events['y'])]
        self.image = self.hist2d(events)
        self.imstats = [np.max(self.image), np.median(self.image)]
        self.mesh = self.meshgrid()
        self.fit_info = ''

    def hist2d(self, events):
        image=np.zeros((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))
        for index, event in events.iterrows():
            image[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]+=1
        image = image.ravel()
        return image
    
    def meshgrid(self):
        x = np.arange(self.xlim[1]-self.xlim[0]+1)
        y = np.arange(self.ylim[1]-self.ylim[0]+1)
        X,Y = np.meshgrid(x,y)
        return X,Y
    
    def __call__(self, func, **kwargs):
        try:
            popt, pcov = optimize.curve_fit(func, self.mesh, self.image, **kwargs) #, gtol=1e-4,ftol=1e-4
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:
            self.fit_info += f'RuntimeError encountered during fit. No localization generated for candidate cluster {self.candidateID}.\n'
            popt = np.zeros(6)
            perr = np.zeros(6)
        except ValueError:
            self.fit_info += f'ValueError encountered during fit. No localization generated for candidate cluster {self.candidateID}.\n'
            popt = np.zeros(6)
            perr = np.zeros(6)
        except OptimizeWarning:
            self.fit_info += f'OptimizeWarning encountered during fit. No localization generated for candidate cluster {self.candidateID}.\n'
            popt = np.zeros(6)
            perr = np.zeros(6)
        return popt, perr

class gauss2D(fit):

    def __init__(self, events, candidateID, width, fitting_tolerance, pixel_size):
        super().__init__(events, candidateID)
        self.bounds = self.bounds()
        self.p0 = self.p0(width)
        self.fitting_tolerance = fitting_tolerance
        self.pixel_size = pixel_size

    def bounds(self):
        bounds = ([0., 0., 0., 0., 0., 0.], [self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0], np.inf, np.inf, np.inf, np.inf])
        return bounds
    
    def p0(self, width):
        p0 = (self.xstats[0]-self.xlim[0], self.ystats[0]-self.ylim[0], width, width, self.imstats[0], self.imstats[1])
        return p0
    
    def func(self, XY, x0, y0, sigma_x, sigma_y, amplitude, offset):
        X, Y = XY
        g = offset + amplitude * np.exp( - ((X-x0)**2/(2*sigma_x**2) + (Y-y0)**2/(2*sigma_y**2)))
        return g.ravel()
    
    def __call__(self, events, **kwargs):
        opt, err = super().__call__(self.func, bounds=self.bounds, p0=self.p0, **kwargs)
        x = (opt[0]+self.xlim[0])*self.pixel_size # in nm
        y = (opt[1]+self.ylim[0])*self.pixel_size # in nm
        del_x = err[0]*self.pixel_size # in nm
        del_y = err[1]*self.pixel_size # in nm
        if del_x > self.fitting_tolerance*self.pixel_size or del_y > self.fitting_tolerance*self.pixel_size:
            self.fit_info = f'Fitting uncertainties exceed the tolerance. No localization generated for candidate cluster {self.candidateID}.\n'
        t = np.mean(events['t'])/1000. # in ms
        mean_polarity = events['p'].mean()
        p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
        return np.array([self.candidateID, x, y, del_x, del_y, p, t]), self.fit_info


# perform localization for part of candidate dictionary
def localize_canditates2D(i, candidate_dic, func, df_cols, *args, **kwargs):
    print('Localizing PSFs (thread '+str(i)+')...')
    nr_of_localizations = len(candidate_dic)
    localizations = pd.DataFrame(index=range(nr_of_localizations), columns = df_cols)
    index = 0
    nb_fails = 0
    info = ''
    for candidate_id in list(candidate_dic):
        fitting = func(candidate_dic[candidate_id]['events'], candidate_id, *args)
        localization, fitting_info = fitting(candidate_dic[candidate_id]['events'], **kwargs)
        if fitting_info != '':
            info += fitting_info
            nb_fails +=1
        else:
            localizations.loc[index] = localization
            index += 1
    localizations = localizations.drop(localizations.tail(nb_fails).index)
    print('Localizing PSFs (thread '+str(i)+') done!')
    return localizations, info

# 2D Gaussian with rotation (angle theta in [rad])
def gauss2d_theta(XY, amplitude, x0, y0, sigma_x, sigma_y, offset, theta):  
    X, Y = XY
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp( - (a*((X-x0)**2) + 2*b*(X-x0)*(Y-y0) 
                            + c*((Y-y0)**2)))
    return g.ravel()

# Wrapperfunction for constant variable (theta) in Gaussian 2D fit
def const_theta(theta0):
    def const_theta_func(XY, amplitude, x0, y0, sigma_x, sigma_y, offset, theta=theta0):
        return gauss2d_theta(XY, amplitude, x0, y0, sigma_x, sigma_y, offset, theta)
    return const_theta_func

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

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

# 2D Gaussian
def Gaussian2D(candidate_dic,settings,**kwargs):
    # Start the timer
    start_time = time.time()

    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    expected_width = float(kwargs['expected_width'])
    fitting_tolerance = float(kwargs['fitting_tolerance'])

    # Load the optional kwargs
    multithread = utilsHelper.strtobool(kwargs['multithread'])

    # Initializations - general
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    expected_width = expected_width/pixel_size
    fit_func = gauss2D
    params = expected_width, fitting_tolerance, pixel_size
    df_cols = ['candidate_id', 'x', 'y', 'del_x', 'del_y', 'p', 't']

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # calculate number of jobs on CPU
    nb_candidates = len(candidate_dic)
    if nb_candidates < num_cores or num_cores == 1:
        njobs = 1
        num_cores = 1
    elif nb_candidates/num_cores > 100:
        njobs = np.int64(np.ceil(nb_candidates/100.))
    else:
        njobs = num_cores

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    data_split = slice_data(candidate_dic, njobs)
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], fit_func, df_cols, *params) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list)
    
    # Fit performance information
    nb_fails = len(candidate_dic)-len(localizations)
    n_loc_info = f'Number of localizations found: {len(localizations)}'
    logging.info(n_loc_info)
    n_fails_info = f'Gaussian fitting failed for {nb_fails} candidate cluster(s).'
    logging.info(n_fails_info)
    gaussian_fit_info = ''
    gaussian_fit_info += ''.join([res[1] for res in RES])
    logging.info(gaussian_fit_info)

    # Stop the timer
    end_time = time.time()
    timing_info = f'Gaussian fitting took {end_time-start_time} seconds.'
    logging.info(timing_info)
    gaussian_fit_info = timing_info + '\n' + n_loc_info + '\n' + n_fails_info + '\n' + gaussian_fit_info

    return localizations, gaussian_fit_info

# ToDo: Modify for 3D to make it functional
def Gaussian3D(candidate_dic,settings,**kwargs):
    
    localizations = pd.DataFrame(index=range(candidate_dic), columns=['x','y','p','t'])
    Gaussian_fit_info = ''
    
    return localizations, Gaussian_fit_info