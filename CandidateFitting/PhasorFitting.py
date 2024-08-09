import inspect
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
import time, logging
from scipy import optimize
import warnings
from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "PhasorFitting": {
            "dist_kwarg" : {"base": "XYDist", "description": "Two-dimensional event-distribution to fit to get x,y-localization.", "default_option": "Hist2d_xy"},
            "required_kwargs": [            ],
            "optional_kwargs": [],
            "help_string": "3D phasor.",
            "display_name": "Phasor"
        },
        "PhasorFitting_customTimeFit": {
            "dist_kwarg" : {"base": "XYDist", "description": "Two-dimensional event-distribution to fit to get x,y-localization.", "default_option": "Hist2d_xy"},
            "time_kwarg" : {"base": "TemporalFits", "description": "Temporal fitting routine to get time estimate.", "default_option": "LognormCDFFirstEvents_weighted"},
            "required_kwargs": [            ],
            "optional_kwargs": [],
            "help_string": "2D phasor with custom time fit routine",
            "display_name": "Phasor - custom time fit"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------
class fit:
    def __init__(self, events, candidateID):
        self.candidateID = candidateID
        self.image = self.hist2d(events)
        self.imstats = [np.max(self.image), np.median(self.image)]
        self.fit_info = ''

    def hist2d(self, events):
        image=np.zeros((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))
        for index, event in events.iterrows():
            image[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]+=1
        image = image.ravel()
        return image

class phasor():
    def __init__(self, events, distfunc, pixel_size):
        self.dist = distfunc(events).dist2D
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.ylim = [np.min(events['y']), np.max(events['y'])]
        self.fit_info = ''
        self.image = self.dist
        self.imstats = [np.max(self.image), np.median(self.image)]
        self.pixel_size = pixel_size
        self.tlim = [np.min(events['t']), np.max(events['t'])]
        self.msscale = 5
        self.tlim_scaled = [np.floor(self.tlim[0]/(self.msscale*1000)), np.ceil(self.tlim[1]/(self.msscale*1000))]
        self.t_hist = np.zeros((int(self.tlim_scaled[1]-self.tlim_scaled[0]+1),1))
        for index, event in events.iterrows():
            self.t_hist[(int(event['t']/(self.msscale*1000)-self.tlim_scaled[0]))]+=1
        self.t_hist = self.t_hist.ravel()

    def __call__(self, candidate, time_fit, candidate_id, **kwargs):#(self, candidate, time_func, **kwargs):
        
        #Perform 2D Fourier transform over the xy ROI
        self.image_sq = self.image.reshape(((self.ylim[1]-self.ylim[0]+1), (self.xlim[1]-self.xlim[0]+1)))
        ROI_F = np.fft.fft2(self.image_sq)
        #We have to calculate the phase angle of array entries [0,1] and [1,0] for 
        #the sub-pixel x and y values, respectively
        #This phase angle can be calculated as follows:
        xangle = np.arctan(ROI_F[0,1].imag/ROI_F[0,1].real) - np.pi
        #Correct in case it's positive
        if xangle > 0:
            xangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionX = abs(xangle)/(2*np.pi/((self.xlim[1]-self.xlim[0]+1)+1))
        
        yangle = np.arctan(ROI_F[1,0].imag/ROI_F[1,0].real) - np.pi
        #Correct in case it's positive
        if yangle > 0:
            yangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionY = abs(yangle)/(2*np.pi/((self.ylim[1]-self.ylim[0]+1)+1))
        
        ROI_FFT_t = np.fft.fft(self.t_hist)
        tangle = np.arctan(ROI_FFT_t[1].imag/ROI_FFT_t[1].real) - np.pi
        #Correct in case it's positive
        if tangle > 0:
            tangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionT = (abs(tangle)/(2*np.pi))*((self.tlim_scaled[1]-self.tlim_scaled[0]+1)+1)
        
        x = (PositionX+self.xlim[0])*self.pixel_size # in nm
        y = (PositionY+self.ylim[0])*self.pixel_size # in nm
        t = ((PositionT+self.tlim_scaled[0])*self.msscale) # in ms
        
        mean_polarity = candidate['events']['p'].mean()
        p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
        loc_df = pd.DataFrame({'candidate_id': candidate_id, 'x': x, 'y': y, 'p': p, 't': t, 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][1], 'y_dim': candidate['cluster_size'][0], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': ''}, index=[0])
        return loc_df

class phasor_customTimeFit(fit):
    def __init__(self, events, distfunc, pixel_size):
        self.dist = distfunc(events).dist2D
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.ylim = [np.min(events['y']), np.max(events['y'])]
        self.fit_info = ''
        self.image = self.dist
        self.imstats = [np.max(self.image), np.median(self.image)]
        self.pixel_size = pixel_size
        self.tlim = [np.min(events['t']), np.max(events['t'])]

    def __call__(self, candidate, time_fit, candidate_id, **kwargs):#(self, candidate, time_func, **kwargs):
        
        #Perform 2D Fourier transform over the xy ROI
        self.image_sq = self.image.reshape(((self.ylim[1]-self.ylim[0]+1), (self.xlim[1]-self.xlim[0]+1)))
        ROI_F = np.fft.fft2(self.image_sq)
        #We have to calculate the phase angle of array entries [0,1] and [1,0] for 
        #the sub-pixel x and y values, respectively
        #This phase angle can be calculated as follows:
        xangle = np.arctan(ROI_F[0,1].imag/ROI_F[0,1].real) - np.pi
        #Correct in case it's positive
        if xangle > 0:
            xangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionX = abs(xangle)/(2*np.pi/((self.xlim[1]-self.xlim[0]+1)+1))
        
        yangle = np.arctan(ROI_F[1,0].imag/ROI_F[1,0].real) - np.pi
        #Correct in case it's positive
        if yangle > 0:
            yangle -= 2*np.pi
        #Calculate position based on the ROI size
        PositionY = abs(yangle)/(2*np.pi/((self.ylim[1]-self.ylim[0]+1)+1))
        
        t, del_t, t_fit_info, opt_t = time_fit(candidate['events'], np.array([np.nan])) # t, del_t in ms
        if t_fit_info != '':
            self.fit_info = t_fit_info
        
        x = (PositionX+self.xlim[0])*self.pixel_size # in nm
        y = (PositionY+self.ylim[0])*self.pixel_size # in nm
        
        mean_polarity = candidate['events']['p'].mean()
        p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
        loc_df = pd.DataFrame({'candidate_id': candidate_id, 'x': x, 'y': y, 'p': p, 't': t, 'del_t': del_t, 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][1], 'y_dim': candidate['cluster_size'][0], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': ''}, index=[0])
        return loc_df

# perform localization for part of candidate dictionary
def phasor2D(i, candidate_dic, distfunc, time_fit, pixel_size):
    print('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    for candidate_id in list(candidate_dic):
        fitting = phasor(candidate_dic[candidate_id]['events'], distfunc, pixel_size)
        localization = fitting(candidate_dic[candidate_id], time_fit, candidate_id)
        localizations.append(localization)
    if localizations == []:
        localizations = pd.DataFrame()
    else:
        localizations = pd.concat(localizations, ignore_index=True)
    print('Localizing PSFs (thread '+str(i)+') done!')
    return localizations

# perform localization for part of candidate dictionary
def phasor2D_customTime(i, candidate_dic, distfunc, time_fit, pixel_size):
    print('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    for candidate_id in list(candidate_dic):
        fitting = phasor_customTimeFit(candidate_dic[candidate_id]['events'], distfunc, pixel_size)
        localization = fitting(candidate_dic[candidate_id], time_fit, candidate_id)
        localizations.append(localization)
    if localizations == []:
        localizations = pd.DataFrame()
    else:
        localizations = pd.concat(localizations, ignore_index=True)
    print('Localizing PSFs (thread '+str(i)+') done!')
    return localizations

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

# Phasor fitting
def PhasorFitting(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    distfunc = getattr(eventDistributions, kwargs['dist_kwarg'])
    time_fit = None

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(phasor2D)(i, data_split[i], distfunc, time_fit, pixel_size) for i in range(len(data_split)))

    localization_list = [res for res in RES]
    localizations = pd.concat(localization_list)
    
    phasor_fit_info = ''

    return localizations, phasor_fit_info

# Phasor fitting
def PhasorFitting_customTimeFit(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    distfunc = getattr(eventDistributions, kwargs['dist_kwarg'])
    time_fit = getattr(timeFitting, kwargs['time_kwarg'])()

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(phasor2D_customTime)(i, data_split[i], distfunc, time_fit, pixel_size) for i in range(len(data_split)))

    localization_list = [res for res in RES]
    localizations = pd.concat(localization_list)
    
    phasor_fit_info = ''

    return localizations, phasor_fit_info