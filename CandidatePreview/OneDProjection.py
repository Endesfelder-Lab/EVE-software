import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import fit
import logging
#Obtain eventdistribution functions
from EventDistributions import eventDistributions

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "OneDProjection": {
            "required_kwargs": [
                {"name": "t_bin_width", "description": "Temporal bin width (in ms)", "default":"1"},
                {"name": "show_first", "description": "Plot the first events per pixel","default":"False"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Draws temporal trace of the candidate cluster.",
            "display_name": "Temporal trace of candidate cluster"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

# Define the modified shifted Rayleigh function with an amplitude parameter
def rayleigh_distribution(x, shift, sigma, amplitude, offset):
    result = amplitude * (x - shift) * np.exp(-(x - shift)**2 / (2 * sigma**2))
    result[x < shift] = 0  # Set the function to zero for x less than the shift
    result += offset  # Add an additional offset for x greater than the shift
    return result

from scipy.stats import rv_continuous
from scipy.stats import rayleigh
# define discrete rayleigh function for mle fit
class rayleigh_gen(rv_continuous):
    "Rayleigh distribution"
    def _pdf(self, t, shift, sigma, amplitude, offset): # t in µs
        return rayleigh_distribution(t, shift, sigma, amplitude, offset)


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def OneDProjection(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    pixel_size = float(settings['PixelSize_nm']['value']) # in nm
    t_bin_width = float(kwargs['t_bin_width'])
    show_first = utilsHelper.strtobool(kwargs['show_first'])
    

    first_events = pd.DataFrame()
    if show_first==True:
        first_events = eventDistributions.FirstTimestamp(findingResult).get_smallest_t(findingResult)

    ax = figure.add_subplot(111)
    figure.tight_layout()
    figure.subplots_adjust(top=0.95,bottom=0.190,left=0.080)

    hist_t = eventDistributions.Hist1d_t(findingResult)
    hist_t.set_t_bin_width(t_bin_width, findingResult)
    
    bincentres = (hist_t.t_edges[1:]-hist_t.t_edges[:-1])/2.0 + hist_t.t_edges[:-1]
    # try a Rayleigh fit
    mean = np.average(bincentres, weights=hist_t.dist1D)
    std = np.sqrt(np.average((bincentres-mean)**2, weights=hist_t.dist1D))
    shift = mean - 2*std
    sigma = np.average(bincentres, weights=hist_t.dist1D) - shift
    amplitude = np.sqrt(np.exp(1))*np.amax(hist_t.dist1D)*sigma # *(np.average(bincentres, weights=hist_t.dist1D)-bincentres[0])
    p0 = [shift, sigma, amplitude, 0]
    print(f"p0 is {p0}")
    popt, pcov = curve_fit(rayleigh_distribution, bincentres, hist_t.dist1D, p0=p0, bounds=(0, np.inf), method='dogbox')
    perr = np.sqrt(np.diag(pcov))
    print(popt, perr)
    ax.plot(bincentres, rayleigh_distribution(bincentres, *popt), label='Rayleigh fit', color='black')

    # try a MLE Rayleigh fit
    np.save("/home/laura/PhD/Event_Based_Sensor_Project/GUI_tests/MLE_fit/data.npy", findingResult['t'])
    # custom_rayleigh = rayleigh_gen("custom_rayleigh")
    # bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]
    # mle_rayleigh = fit(custom_rayleigh, findingResult['t'], guess=p0, bounds=bounds)
    # print(mle_rayleigh.params)
    # mle_rayleigh.plot()


    mean = bincentres[0]
    shift = mean
    sigma = np.average(bincentres, weights=hist_t.dist1D) - shift
    amplitude = np.sqrt(np.exp(1))*np.amax(hist_t.dist1D)*sigma # *(np.average(bincentres, weights=hist_t.dist1D)-bincentres[0])
    p0 = [shift, sigma, amplitude, 0]
    print(f"p0 is {p0}")
    popt, pcov = curve_fit(rayleigh_distribution, bincentres, hist_t.dist1D, p0=p0, bounds=(0, np.inf), method='dogbox')
    perr = np.sqrt(np.diag(pcov))
    print(popt, perr)
    ax.plot(bincentres, rayleigh_distribution(bincentres, *popt), label='Rayleigh fit different inital guess', color='red')

    
    # Plot the 2D histograms
    if not len(findingResult[findingResult['p'] == 1]) == 0:
        hist_t_pos = hist_t(findingResult[findingResult['p'] == 1])[0]
        ax.bar(hist_t.t_edges[:-1], hist_t_pos, width=hist_t.t_bin_width,  label='Positive events', color='C0', alpha=0.5, align='edge')
    if not len(findingResult[findingResult['p'] == 0]) == 0:
        hist_t_neg = hist_t(findingResult[findingResult['p'] == 0])[0]
        ax.bar(hist_t.t_edges[:-1], hist_t_neg, width=hist_t.t_bin_width,  label='Negative events', color='C1', alpha=0.5, align='edge')
    if not len(first_events) == 0:
        hist_t_first = hist_t(first_events)[0]
        # Add a zero bin to the beginning and end of hist_t_first
        # try rayleigh fit of hist_t_first
        shift = bincentres[0]
        sigma = np.average(bincentres, weights=hist_t_first) - shift
        amplitude = np.sqrt(np.exp(1))*np.amax(hist_t_first)*sigma # *(np.average(bincentres, weights=hist_t_first)-bincentres[0])
        p0 = [shift, sigma, amplitude, 0]
        popt, pcov = curve_fit(rayleigh_distribution, bincentres, hist_t_first, p0=p0, bounds=(0, np.inf))
        perr = np.sqrt(np.diag(pcov))
        print(popt, perr)
        ax.plot(bincentres, rayleigh_distribution(bincentres, *popt), label='Rayleigh fit first events', color='darkgreen')

        hist_t_first = np.insert(hist_t_first, 0, 0)  # Add a zero bin at the beginning
        hist_t_first = np.append(hist_t_first, 0)  # Add a zero bin at the end
        t_edges = np.insert(hist_t.t_edges, 0, hist_t.t_edges[0]-hist_t.t_bin_width)  # Add a zero bin at the beginning
        t_edges = np.append(t_edges, hist_t.t_edges[-1]+hist_t.t_bin_width)  # Add a zero bin at the end

        bincentres = (t_edges[1:]-t_edges[:-1])/2.+t_edges[:-1] 
        ax.step(bincentres,hist_t_first,where='mid',color='C2', label='First events',linewidth=1.5)
        # ax.bar(hist_t.t_edges[:-1], hist_t_first, width=hist_t.t_bin_width,  label='First events', color='C2', alpha=0.5, align='edge')


    # Add and set labels
    ax.set_xlim(hist_t.t_edges[0], hist_t.t_edges[-1])
    ax.set_xlabel('t [ms]')
    ax.set_ylabel('number of events')
    ax.legend(loc='upper right')
    
    # required output none
    return 1
