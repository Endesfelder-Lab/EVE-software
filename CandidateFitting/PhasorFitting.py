import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time, logging
from scipy import optimize
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "PhasorFitting": {
            "required_kwargs": [            ],
            "optional_kwargs": [
                {"name": "multithread","description": "True to use multithread parallelization; False not to.","default":True},
            ],
            "help_string": "3D phasor.",
            "display_name": "Phasor"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------


def hist2d(self, events):
    image=np.zeros((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))
    for index, event in events.iterrows():
        image[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]+=1
    image = image.ravel()
    return image
    
def phasor_xy(events):
    
    fit_info = ''
    t = np.mean(events['t'])/1000. # in ms
    mean_polarity = events['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
    
    loc_df = pd.DataFrame({'candidate_id': self.candidateID, 'x': x, 'y': y, 'mse': mse, 'del_x': del_x, 'del_y': del_y, 'p': p, 't': t}, index=[0]) #return np.array([self.candidateID, x, y, mse, del_x, del_y, p, t]), self.fit_info
    return loc_df, self.fit_info
    
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


class phasor(fit):
    
    def __init__(self, events, candidateID, pixel_size):
        super().__init__(events, candidateID)
        self.bounds = self.bounds()
        self.msscale = 5
        self.time_hist = self.hist1d_time(events,msscale=self.msscale)
        self.pixel_size = pixel_size
        
    def bounds(self):
        bounds = ([0., 0., 0., 0., 0., 0.], [self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0], np.inf, np.inf, np.inf, np.inf])
        return bounds
    
    def hist1d_time(self, events, msscale = 10):
        self.tlim = [np.min(events['t']), np.max(events['t'])]
        self.tlim_scaled = [np.floor(self.tlim[0]/(msscale*1000)), np.ceil(self.tlim[1]/(msscale*1000))]
        t_hist = np.zeros((int(self.tlim_scaled[1]-self.tlim_scaled[0]+1),1))
        for index, event in events.iterrows():
            t_hist[(int(event['t']/(msscale*1000)-self.tlim_scaled[0]))]+=1
        t_hist = t_hist.ravel()
        return t_hist
    
    def __call__(self, events, **kwargs):
        
        #Perform 2D Fourier transform over the xy ROI
        self.image_sq = self.image.reshape(((self.xlim[1]-self.xlim[0]+1), (self.ylim[1]-self.ylim[0]+1)))
        ROI_F = np.fft.fft2(self.image_sq)
        #We have to calculate the phase angle of array entries [0,1] and [1,0] for 
        #the sub-pixel x and y values, respectively
        #This phase angle can be calculated as follows:
        xangle = np.arctan(ROI_F[0,1].imag/ROI_F[0,1].real) - np.pi
        #Correct in case it's positive
        if xangle > 0:
            xangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionX = abs(xangle)/(2*np.pi/((self.xlim[1]-self.xlim[0]+1)+1));
        
        yangle = np.arctan(ROI_F[1,0].imag/ROI_F[1,0].real) - np.pi
        #Correct in case it's positive
        if yangle > 0:
            yangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionY = abs(yangle)/(2*np.pi/((self.ylim[1]-self.ylim[0]+1)+1));
        
        ROI_FFT_t = np.fft.fft(self.time_hist)
        tangle = np.arctan(ROI_FFT_t[1].imag/ROI_FFT_t[1].real) - np.pi
        #Correct in case it's positive
        if tangle > 0:
            tangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionT = (abs(tangle)/(2*np.pi))*((self.tlim_scaled[1]-self.tlim_scaled[0]+1)+1)
        
        x = (PositionX+self.xlim[0])*self.pixel_size # in nm
        y = (PositionY+self.ylim[0])*self.pixel_size # in nm
        t = ((PositionT+self.tlim_scaled[0])*self.msscale) # in ms
        
        mean_polarity = events['p'].mean()
        p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
        loc_df = pd.DataFrame({'candidate_id': self.candidateID, 'x': x, 'y': y, 'p': p, 't': t}, index=[0])
        
        self.fit_info = ''
        return loc_df, self.fit_info

# perform localization for part of candidate dictionary
def localize_canditates2D(i, candidate_dic, func, *args, **kwargs):
    print('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
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
            localizations.append(localization)
            index += 1
    if localizations == []:
        localizations = pd.DataFrame()
    else:
        localizations = pd.concat(localizations, ignore_index=True)
    print('Localizing PSFs (thread '+str(i)+') done!')
    return localizations, info

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

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

# Phasor fitting
def PhasorFitting(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Initializations - general
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    fit_func = phasor
    params = [pixel_size]
    
    # Load the optional kwargs
    multithread = utilsHelper.strtobool(kwargs['multithread'])

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = nb_jobs(candidate_dic, num_cores)
    data_split = slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    localize_canditates2D(0, data_split[0], fit_func, *params)

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], fit_func, *params) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list)
    

    phasor_fit_info = ''
    phasor_fit_info += ''.join([res[1] for res in RES])
    logging.info(phasor_fit_info)

    return localizations, phasor_fit_info

# 2D LogGaussian
def LogGaussian2D(candidate_dic,settings,**kwargs):
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
    fit_func = loggauss2D
    params = expected_width, fitting_tolerance, pixel_size

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = nb_jobs(candidate_dic, num_cores)
    data_split = slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], fit_func, *params) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list)
    
    # Fit performance information
    nb_fails = len(candidate_dic)-len(localizations)
    n_fails_info = f'Gaussian fitting failed for {nb_fails} candidate cluster(s).'
    logging.info(n_fails_info)
    gaussian_fit_info = ''
    gaussian_fit_info += ''.join([res[1] for res in RES])
    logging.info(gaussian_fit_info)

    gaussian_fit_info = n_fails_info + '\n' + gaussian_fit_info

    return localizations, gaussian_fit_info

# ToDo: Include calibration file to transform sigma_x/sigma_y to z height
def Gaussian3D(candidate_dic,settings,**kwargs):
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
    theta = np.radians(float(kwargs['theta']))
    fit_func = gauss3D
    params = expected_width, fitting_tolerance, pixel_size, theta

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = nb_jobs(candidate_dic, num_cores)
    data_split = slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], fit_func, *params) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list)
    
    # Fit performance information
    nb_fails = len(candidate_dic)-len(localizations)
    n_fails_info = f'Gaussian fitting failed for {nb_fails} candidate cluster(s).'
    logging.info(n_fails_info)
    gaussian_fit_info = ''
    gaussian_fit_info += ''.join([res[1] for res in RES])
    logging.info(gaussian_fit_info)

    gaussian_fit_info = n_fails_info + '\n' + gaussian_fit_info

    return localizations, gaussian_fit_info