import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time
from scipy import optimize

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Gaussian2D_new": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Makes a 2D gaussian fit to determine the localization parameters.",
            "display_name": "2D Gaussian new"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def localization_PSF(sub_events,pixel_size):
    opt,err=gaussian_fitting(sub_events)
    x = (opt[0]+np.min(sub_events['x']))*pixel_size # in nm
    y = (opt[1]+np.min(sub_events['y']))*pixel_size # in nm
    t = np.mean(sub_events['t'])/1000. # in ms
    p = sub_events['p'][0]
    return np.array([x,y,p,t])

def gaussian_fitting(sub_events):
    sub_image=np.zeros((np.max(sub_events['y'])-np.min(sub_events['y'])+1,np.max(sub_events['x'])-np.min(sub_events['x'])+1))
    for k in np.arange(len(sub_events)):
        sub_image[sub_events['y'][k]-np.min(sub_events['y']),sub_events['x'][k]-np.min(sub_events['x'])]+=1
    initial_guess=np.mean(sub_events['x'])-np.min(sub_events['x']),np.mean(sub_events['y'])-np.min(sub_events['y']),np.std(sub_events['x']),np.std(sub_events['y']),np.max(sub_image),np.median(sub_image)
    xrange=np.arange(np.max(sub_events['x'])-np.min(sub_events['x'])+1)
    yrange=np.arange(np.max(sub_events['y'])-np.min(sub_events['y'])+1)
    errorfunction=lambda p:np.ravel(gaussian_symmetrical(xrange, yrange, *p)-sub_image)
    p_full = optimize.leastsq(errorfunction,initial_guess,gtol=1e-4,ftol=1e-4,full_output=True)
    p=p_full[0]
    fv=p_full[2]['fvec']
    ss_err=(fv**2).sum()
    ss_tot=((sub_image-sub_image.mean())**2).sum()
    err=ss_err/ss_tot
    return p,err

def gaussian_symmetrical(xrange, yrange, x0, y0, xwidth, ywidth, height, offset):
    fX=np.exp(-(xrange-x0)**2/(2.*xwidth**2))
    fY=np.exp(-(yrange-y0)**2/(2.*ywidth**2))
    fY=fY.reshape(len(fY),1)
    return offset+height*fY*fX


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def Gaussian2D_new(candidate_dic,settings,**kwargs):
    
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    nr_of_localizations = len(candidate_dic)
    localizations = pd.DataFrame(index=range(nr_of_localizations), columns=['x','y','p','t'])

    for i in np.unique(list(candidate_dic)):
        localizations.loc[i] = localization_PSF(candidate_dic[i]['events'], pixel_size)
    
    return localizations, ''