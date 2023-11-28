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

# perform localization for part of candidate dictionary
def localize_canditates2D(candidate_dic, expected_width, pixel_size):
    nr_of_localizations = len(candidate_dic)
    localizations = pd.DataFrame(index=range(nr_of_localizations), columns=['x','y','p','t'])
    index = 0
    nb_fails = 0
    Gaussian_fit_info = ''
    for candidate_id in list(candidate_dic):
        localization, fitting_info = localization2D(candidate_dic[candidate_id]['events'], candidate_id, expected_width, pixel_size)
        if fitting_info != '':
            Gaussian_fit_info += fitting_info
            nb_fails +=1
        else:
            localizations.loc[index] = localization
            index += 1
    localizations = localizations.drop(localizations.tail(nb_fails).index)
    return localizations, Gaussian_fit_info

# 2d localization via gaussian fit
def localization2D(sub_events, candidate_id, expected_width, pixel_size):
    opt, err, fitting_info = gaussian_fitting(sub_events, candidate_id, expected_width)
    x = (opt[0]+np.min(sub_events['x']))*pixel_size * bool(err[0]) # in nm
    y = (opt[1]+np.min(sub_events['y']))*pixel_size * bool(err[1]) # in nm
    t = np.mean(sub_events['t'])/1000. # in ms
    mean_polarity = sub_events['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
    return np.array([x,y,p,t]), fitting_info

# gaussian fit via scipy.optimize.curve_fit with bounds
def gaussian_fitting(sub_events, candidate_id, expected_width):
    min_x = np.min(sub_events['x'])
    min_y = np.min(sub_events['y'])
    sub_image=np.zeros((np.max(sub_events['y'])-min_y+1,np.max(sub_events['x'])-min_x+1))
    for index, event in sub_events.iterrows():
        sub_image[int(event['y']-min_y),int(event['x']-min_x)]+=1
    bounds = ([0., 0., 0., 0., 0., 0.], [np.max(sub_events['x'])-np.min(sub_events['x']), np.max(sub_events['y'])-np.min(sub_events['y']), np.inf, np.inf, np.inf, np.inf])
    p0 = (np.mean(sub_events['x'])-np.min(sub_events['x']), np.mean(sub_events['y'])-np.min(sub_events['y']), expected_width, expected_width, np.max(sub_image), np.median(sub_image))
    x = np.arange(np.max(sub_events['x'])-np.min(sub_events['x'])+1)
    y = np.arange(np.max(sub_events['y'])-np.min(sub_events['y'])+1)
    X,Y = np.meshgrid(x,y)
    fitting_info = ''
    try:
        popt, pcov = optimize.curve_fit(gauss2d, (X,Y), sub_image.ravel(), p0=p0, gtol=1e-4,ftol=1e-4, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        fitting_info += f'RuntimeError encountered during fit. No localization generated for candidate cluster {candidate_id}.\n'
        popt = np.zeros(6)
        perr = np.zeros(6)
    except ValueError:
        fitting_info += f'ValueError encountered during fit. No localization generated for candidate cluster {candidate_id}.\n'
        popt = np.zeros(6)
        perr = np.zeros(6)
    except OptimizeWarning:
        fitting_info += f'OptimizeWarning encountered during fit. No localization generated for candidate cluster {candidate_id}.\n'
        popt = np.zeros(6)
        perr = np.zeros(6)
    return popt, perr, fitting_info

# normal 2D Gaussian without rotation
def gauss2d(XY, x0, y0, sigma_x, sigma_y, amplitude, offset):
    X, Y = XY
    g = offset + amplitude * np.exp( - ((X-x0)**2/(2*sigma_x**2) + (Y-y0)**2/(2*sigma_y**2)))
    return g.ravel()

def localization3D(sub_events,pixel_size):
    opt, err, fitting_info = gaussian_fitting_theta(sub_events, 150./pixel_size)
    x = (opt[0]+np.min(sub_events['x']))*pixel_size * bool(err[0]) # in nm
    y = (opt[1]+np.min(sub_events['y']))*pixel_size * bool(err[1]) # in nm
    t = np.mean(sub_events['t'])/1000. # in ms
    p = sub_events['p'][0]
    return np.array([x,y,p,t]), fitting_info

def gaussian_fitting_theta(sub_events, expected_width):
    sub_image=np.zeros((np.max(sub_events['y'])-np.min(sub_events['y'])+1,np.max(sub_events['x'])-np.min(sub_events['x'])+1))
    for k in np.arange(len(sub_events)):
        sub_image[sub_events['y'][k]-np.min(sub_events['y']),sub_events['x'][k]-np.min(sub_events['x'])]+=1
    bounds = ([0., 0., 0., 0., 0., 0.], [np.max(sub_events['x'])-np.min(sub_events['x']), np.max(sub_events['y'])-np.min(sub_events['y']), np.inf, np.inf, np.inf, np.inf])
    p0 = (np.mean(sub_events['x'])-np.min(sub_events['x']),np.mean(sub_events['y'])-np.min(sub_events['y']),np.std(sub_events['x']),np.std(sub_events['y']),np.max(sub_image),np.median(sub_image))
    x = np.arange(np.max(sub_events['x'])-np.min(sub_events['x'])+1)
    y = np.arange(np.max(sub_events['y'])-np.min(sub_events['y'])+1)
    X,Y = np.meshgrid(x,y)
    try:
        popt, pcov = optimize.curve_fit(gauss2d, (X,Y), sub_image.ravel(), p0=p0, gtol=1e-4,ftol=1e-4, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        fitting_info += 'RuntimeError encountered during fit. No localization generated for this candidate cluster.\n'
        popt = np.zeros(6)
        perr = np.zeros(6)
    except ValueError:
        fitting_info += 'ValueError encountered during fit. No localization generated for this candidate cluster.\n'
        popt = np.zeros(6)
        perr = np.zeros(6)
    except OptimizeWarning:
        fitting_info += 'OptimizeWarning encountered during fit. No localization generated for this candidate cluster.\n'
        popt = np.zeros(6)
        perr = np.zeros(6)
    return popt, perr, fitting_info

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
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    expected_width = float(kwargs['expected_width'])

    # Load the optional kwargs
    multithread = bool(kwargs['multithread'])

    # Initializations - general
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    expected_width = expected_width/pixel_size

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    logging.info("Candidate fitting split on "+str(num_cores)+" cores.")
    
    # Determine all localizations
    data_split = slice_data(candidate_dic, num_cores)
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(data_split[i], expected_width, pixel_size) for i in range(len(data_split)))
    
    LOC = [res[0] for res in RES]
    localizations = pd.concat(LOC)
    
    # Fit performance information
    nb_fails = len(candidate_dic)-len(localizations)
    INFO=""
    if nb_fails == 0:
        INFO += "Gaussian fitting was successful for all candidate clusters."
    elif nb_fails == 1:
        INFO += "Gaussian fitting failed for 1 candidate cluster:\n"
    else:
        INFO += f"Gaussian fitting failed for {nb_fails} candidate clusters:\n"
    INFO += ''.join([res[1] for res in RES])
    logging.info(INFO)

    return localizations, INFO

# ToDo: Modify for 3D to make it functional
def Gaussian3D(candidate_dic,settings,**kwargs):
    
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    theta = float(kwargs['Theta_deg']['value'])*2.*np.pi/360. # in rad
    nr_of_localizations = len(candidate_dic)
    localizations = pd.DataFrame(index=range(nr_of_localizations), columns=['x','y','p','t'])

    index = 0
    Gaussian_fit_info = ''
    for i in np.unique(list(candidate_dic)):
        localization, fitting_info = localization3D(candidate_dic[i]['events'], pixel_size)
        if fitting_info != '':
            Gaussian_fit_info += fitting_info
        else:
            localizations.loc[index] = localization
            index += 1
    
    logging.info(Gaussian_fit_info)
    
    return localizations, Gaussian_fit_info