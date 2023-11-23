import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time, logging
from scipy import optimize
import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Gaussian2D": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
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

def localization2D(sub_events,pixel_size):
    opt, err, fitting_info = gaussian_fitting(sub_events, 150./pixel_size)
    x = (opt[0]+np.min(sub_events['x']))*pixel_size * bool(err[0]) # in nm
    y = (opt[1]+np.min(sub_events['y']))*pixel_size * bool(err[1]) # in nm
    t = np.mean(sub_events['t'])/1000. # in ms
    p = sub_events['p'][0]
    return np.array([x,y,p,t]), fitting_info

def gaussian_fitting(sub_events, expected_width):
    sub_image=np.zeros((np.max(sub_events['y'])-np.min(sub_events['y'])+1,np.max(sub_events['x'])-np.min(sub_events['x'])+1))
    for k in np.arange(len(sub_events)):
        sub_image[sub_events['y'][k]-np.min(sub_events['y']),sub_events['x'][k]-np.min(sub_events['x'])]+=1
    bounds = ([0., 0., 0., 0., 0., 0.], [np.max(sub_events['x'])-np.min(sub_events['x']), np.max(sub_events['y'])-np.min(sub_events['y']), np.inf, np.inf, np.inf, np.inf])
    p0 = (np.mean(sub_events['x'])-np.min(sub_events['x']),np.mean(sub_events['y'])-np.min(sub_events['y']),np.std(sub_events['x']),np.std(sub_events['y']),np.max(sub_image),np.median(sub_image))
    x = np.arange(np.max(sub_events['x'])-np.min(sub_events['x'])+1)
    y = np.arange(np.max(sub_events['y'])-np.min(sub_events['y'])+1)
    X,Y = np.meshgrid(x,y)
    fitting_info = ''
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

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def Gaussian2D(candidate_dic,settings,**kwargs):

    # Start the timer
    start_time = time.time()
    
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    nr_of_localizations = len(candidate_dic)
    localizations = pd.DataFrame(index=range(nr_of_localizations), columns=['x','y','p','t'])

    index = 0
    nb_fails = 0
    Gaussian_fit_info = ''
    for i in np.unique(list(candidate_dic)):
        localization, fitting_info = localization2D(candidate_dic[i]['events'], pixel_size)
        if fitting_info != '':
            Gaussian_fit_info += fitting_info
            nb_fails +=1
        else:
            localizations.loc[index] = localization
            index += 1
    localizations = localizations.drop(localizations.tail(nb_fails).index)
    # if fitting_info != '':
    logging.info(Gaussian_fit_info)
    # Stop the timer
    end_time = time.time()
 
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    logging.info(f"2D Gaussian fit ran for {elapsed_time} seconds.")

    return localizations, Gaussian_fit_info

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