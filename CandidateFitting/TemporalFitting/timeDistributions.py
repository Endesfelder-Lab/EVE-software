import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
import pandas as pd

class hist_fit:

    def __init__(self, times, **kwargs):
        self.hist, self.hist_edges = np.histogram(times,**kwargs)
        bin_width = self.hist_edges[1] - self.hist_edges[0]
        self.bincentres = (self.hist_edges[1:]-self.hist_edges[:-1])/2.0 + self.hist_edges[:-1]
        self.fit_info = ''
    
    def __call__(self, func, **kwargs):
        try:
            popt, pcov = curve_fit(func, self.bincentres, self.hist, nan_policy='omit', **kwargs)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError as warning:
            self.fit_info += 'RuntimeError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except ValueError as warning:
            self.fit_info += 'ValueError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except OptimizeWarning as warning:
            self.fit_info += 'OptimizeWarning: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        return popt, perr

# rayleigh fit
class rayleigh(hist_fit):

    def __init__(self, times, **kwargs):
        super().__init__(times, **kwargs)
        self.bounds = self.bounds()
        self.p0 = self.p0(times)

    def bounds(self):
        bounds = ([-np.inf, np.finfo(np.float64).tiny, 0.5*np.sqrt(np.finfo(np.float64).tiny), np.finfo(np.float64).tiny], [np.inf, np.inf, np.inf, np.inf])
        return bounds
    
    def p0(self, times):
        shift = np.percentile(times, 5)
        sigma = np.max([np.mean(times) - shift, 0])
        amplitude = np.sqrt(np.exp(1))*np.amax(self.hist)*sigma
        p0 = [shift, sigma, amplitude, np.finfo(np.float64).tiny]
        return p0
    
    def func(self, t, shift, sigma, amplitude, offset):
        result = amplitude * (t - shift)/(sigma**2) * np.exp(-(t - shift)**2 / (2 * sigma**2))
        result[t < shift] = 0  # Set the function to zero for x less than the shift
        result += offset # Add an additional offset for x greater than the shift
        return result
    
    def __call__(self, times, **kwargs):
        opt, err = super().__call__(self.func, bounds=self.bounds, p0=self.p0, **kwargs)
        if self.fit_info != '':
            t = np.mean(times)
        else: 
            times_std = np.std(times)
            if err[0] > times_std:
                self.fit_info += 'TimeToleranceWarning: Temporal fitting uncertainties exceed the tolerance.'
                t = np.mean(times)
            else:
                t = opt[0]
        return t, self.fit_info