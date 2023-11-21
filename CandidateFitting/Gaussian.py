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
        "Gaussian2D": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Makes a 2D gaussian fit to determine the localization parameters.",
            "display_name": "2D Gaussian"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def localization_PSF(sub_events,pixel_size):
    initial_guess_w = 150./pixel_size # expected width of Gaussian in px
    candidate_radius = int(np.round(((np.max(sub_events['y'])-np.min(sub_events['y']))/2. + (np.max(sub_events['x'])-np.min(sub_events['x']))/2.)/2.))+1
    x_mean = np.mean(sub_events['x'])
    y_mean = np.mean(sub_events['y'])
    edge=[int(np.round(y_mean))-candidate_radius,int(np.round(x_mean))-candidate_radius] # I don't see why this is needed...
    opt,err=gaussian_fitting(sub_events,edge, candidate_radius, initial_guess_w)
    y = (opt[0]+edge[0])*pixel_size # in nm
    x = (opt[1]+edge[1])*pixel_size # in nm
    t = np.mean(sub_events['t'])/1000. # in ms
    p = sub_events['p'][0]
    return np.array([x,y,p,t])

def gaussian_fitting(sub_events, edge, candidate_radius, initial_guess_w):
    sub_image=np.zeros((candidate_radius*2+1,candidate_radius*2+1))
    for k in np.arange(len(sub_events)):
        sub_image[sub_events['y'][k]-edge[0],sub_events['x'][k]-edge[1]]+=1
    initial_guess=candidate_radius,candidate_radius,initial_guess_w,np.max(sub_image),np.median(sub_image)
    xrange=np.arange(2*candidate_radius+1)*1.
    yrange=np.arange(2*candidate_radius+1)*1.
    errorfunction=lambda p:np.ravel(gaussian_symmetrical(xrange, yrange, *p)-sub_image)
    p_full = optimize.leastsq(errorfunction,initial_guess,gtol=1e-4,ftol=1e-4,full_output=True)
    p=p_full[0]
    fv=p_full[2]['fvec']
    ss_err=(fv**2).sum()
    ss_tot=((sub_image-sub_image.mean())**2).sum()
    err=ss_err/ss_tot    
    return p,err

def gaussian_symmetrical(xrange, yrange, y0, x0, width, height, offset):
    fX=np.exp(-(xrange-x0)**2/(2.*width**2))
    fY=np.exp(-(yrange-y0)**2/(2.*width**2))
    fY=fY.reshape(len(fY),1)
    return offset+height*fY*fX


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def Gaussian2D(candidate_dic,settings,**kwargs):
    
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    nr_of_localizations = len(candidate_dic)
    localizations = pd.DataFrame(index=range(nr_of_localizations), columns=['x','y','p','t'])

    for i in np.unique(list(candidate_dic)):
        localizations.loc[i] = localization_PSF(candidate_dic[i]['events'], pixel_size)
    
    return localizations, ''