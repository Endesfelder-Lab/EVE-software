import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
import pandas as pd
from EventDistributions import eventDistributions
from scipy.special import erf, erfinv

class fit:
    def __init__(self):
        self.fit_info = ''

    # get R^2 value for fit
    def get_R2(self, data, prediction):
        residuals = data - prediction
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data-np.mean(data))**2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def __call__(self, func, i, f_i,**kwargs):
        try:
            popt, pcov = curve_fit(func, i, f_i, **kwargs)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError as warning:
            self.fit_info += 'TimeRuntimeError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except ValueError as warning:
            self.fit_info += 'TimeValueError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except OptimizeWarning as warning:
            self.fit_info += 'TimeOptimizeWarning: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        return popt, perr

class TwoDfit(fit):

    def __init__(self, dist, use_weights=True):
        super().__init__()
        self.ylim, self.xlim = dist.dist2D.shape
        self.xmean = np.average(np.arange(1,self.xlim+1), weights=np.nansum(dist.dist2D, axis=0))-1. 
        self.ymean = np.average(np.arange(1,self.ylim+1), weights=np.nansum(dist.dist2D, axis=1))-1.
        self.dist = dist # I don't need this
        self.image = dist.dist2D.ravel()
        if hasattr(dist, 'weights') and use_weights==True:
            weights = dist.weights.ravel()
            max_weight = np.nanmax(weights)
            self.sigma = (max_weight - weights + 1)/max_weight
            self.sigma_tot = self.sigma
            mask = ~np.isnan(self.sigma)
            self.sigma = self.sigma[mask] # all nans have to be removed in sigma
        else:
            weights = np.ones_like(self.image)
            self.sigma_tot = weights
            mask = ~np.isnan(self.image)
            self.sigma = weights[mask]
        self.imstats = [np.nanpercentile(self.image, 90), np.nanpercentile(self.image, 10)]
        self.mesh = self.meshgrid()
    
    def meshgrid(self):
        x = np.arange(self.xlim)
        y = np.arange(self.ylim)
        X,Y = np.meshgrid(x,y)
        return X.ravel(),Y.ravel()
    
    def __call__(self, func, **kwargs):
        popt, perr = super().__call__(func, self.mesh, self.image, sigma=self.sigma, nan_policy='omit', **kwargs)
        return popt, perr

# 2D gaussian fit
class gauss2D(TwoDfit):

    def __init__(self, dist, **kwargs):
        if hasattr(dist, 'trafo_gauss'):
            dist.trafo_gauss()
        super().__init__(dist, **kwargs)
        self.width_bounds = [0.5, 2.0]
        x_std = np.sqrt(np.average((np.arange(1, self.xlim + 1) - self.xmean)**2, weights=np.nansum(dist.dist2D, axis=0)))
        y_std = np.sqrt(np.average((np.arange(1, self.ylim + 1) - self.ymean)**2, weights=np.nansum(dist.dist2D, axis=1)))
        self.width = np.mean([x_std, y_std])
        self.ratio = y_std/x_std
        self.ratio = np.max([self.width_bounds[0],self.ratio])
        self.ratio = np.min([self.width_bounds[1],self.ratio])
        self.bounds = self.bounds()
        self.p0 = self.p0()
        self.offset = 0.0

    def bounds(self):
        bounds = ([-0.5, -0.5, 0., self.width_bounds[0], 0.], [self.xlim-0.5, self.ylim-0.5, np.inf, self.width_bounds[1], np.inf]) # allow borders of pixels
        return bounds
    
    def p0(self):
        p0 = (self.xmean, self.ymean, self.width, self.ratio, self.imstats[0])
        return p0
    
    def func(self, XY, x0, y0, sigma, sigma_xy_ratio, amplitude):
        X, Y = XY
        g = amplitude * np.exp( - ((X-x0)**2/(2*sigma**2) + (Y-y0)**2/(2*(sigma*sigma_xy_ratio)**2))) + self.offset
        return g
    
    def __call__(self, dist, events, opt_loc, **kwargs):
        opt, err = super().__call__(self.func, bounds=self.bounds, p0=self.p0, **kwargs)
        t = np.mean(events['t']*1e-3) # in ms
        del_t = np.std(events['t']*1e-3) # in ms
        if self.fit_info == '':
            xy_threshold = 2. # threshold in px
            if abs(opt_loc[0]-opt[0])>xy_threshold or abs(opt_loc[1]-opt[1])>xy_threshold: 
                self.fit_info += 'TimeToleranceWarning: Temporal fit result exceeds the tolerance.'
                opt = np.array([np.nan])
            else:
                t_estimated = self.func((opt[0], opt[1]), *opt)
                t_estimated = dist.undo_trafo_gauss(t_estimated)*1e-3 # in ms
                t_min = np.min(events['t']*1e-3)-2.* del_t
                t_max = np.max(events['t']*1e-3)
                if t_estimated<t_min or t_estimated>t_max:
                    self.fit_info += 'TimeToleranceWarning: Estimated time out of bounds.'
                else: 
                    t = t_estimated
                    del_t = np.sqrt(err[4]**2)*1e-3 # in ms # + err[5]**2
                    opt[4] = (-1)*opt[4]*1e-3 # in ms
                    self.offset = dist.undo_trafo_gauss(self.offset)*1e-3 # in ms
        return t, del_t, self.fit_info, opt
    
class cumsum_fit(fit):
    def __init__(self, events, **kwargs):
        super().__init__()
        self.cumsum = np.cumsum(np.ones_like(events['t'].values))-1. # cumsum should start at 1
        self.times = events['t'].values*1e-3 # in ms
        # make sure that self.times is sorted in time
        self.times = np.sort(self.times)
        self.t0 = events['t'].values[0]*1e-3 # in ms
    
    def __call__(self, func, **kwargs):
        popt, perr = super().__call__(func, self.times, self.cumsum, **kwargs)
        return popt, perr
    
class lognormal_cdf(cumsum_fit):
    def __init__(self, events, **kwargs):
        super().__init__(events, **kwargs)
        self.bounds = self.bounds()
        self.p0 = self.p0()
    
    def bounds(self):
        bounds = ((0., np.finfo(np.float64).tiny/np.sqrt(2.), self.t0-2*np.std(self.times), 0.0, 0.0, 0.0), (np.inf, np.inf, self.times[-1], np.inf, np.inf, np.inf))
        return bounds
    
    def p0(self):
        max_cumsum = np.max(self.cumsum)
        i_half_max = np.argmax(self.cumsum >= 0.5 * max_cumsum)
        shift = np.percentile(self.times, 5)
        mu = np.max([np.log(np.max([self.times[i_half_max]-shift, 1])),0])
        sigma = 1 # idea: set sigma to np.sqrt(2*(np.log(np.mean(times))-mu)), but this seems to be generally to high
        scale = max_cumsum
        slope = 0.1 # expected background event rate
        offset = 0.
        p0 = [mu, sigma, shift, scale, slope, offset]
        return p0
    
    def get_time(self, opt, err, mean_t, std_t):
        mu, sigma, shift, scale, slope, offset = opt
        err_mu, err_sigma, err_shift, err_scale, err_slope, err_offset = err
        t_fit_info = ''
        warnings.simplefilter("error", RuntimeWarning)
        try: 
            alpha = alpha = 0.5*(1.+erf(-sigma/np.sqrt(2)))
            x_hat = np.exp(mu-sigma**2)+shift
            a_fac = np.exp(sigma**2/2.-mu)/(np.sqrt(2*np.pi)*sigma)
            a = scale*a_fac
            b = scale*(alpha-a_fac*x_hat)
            t_intersect = x_hat-alpha/a_fac
            # calculate error of t_intersect with Gaussian error propagation
            fac_shift = 1.
            fac_mu = np.exp(mu-sigma**2)-alpha/a_fac
            fac_sigma = np.exp(mu-sigma**2)/sigma - np.sqrt(np.pi/2.)*sigma*(1-erf(sigma/np.sqrt(2))*np.exp(mu-sigma**2/2.))
            del_t_intersect = np.sqrt((fac_shift*err_shift)**2+(fac_mu*err_mu)**2+(fac_sigma*err_sigma)**2)
        except RuntimeWarning: # for very steep lognormal cdfs a approaches inf (a vertical line) and is thus not feasible
            a = np.inf
            b = np.inf
            try:
                x_hat = np.exp(mu-sigma**2)+shift
                t_intersect = x_hat
                fac_shift = 1.
                fac_mu = np.exp(mu-sigma**2)
                fac_sigma = -2.*sigma*np.exp(mu-sigma**2)
                del_t_intersect = np.sqrt((fac_shift*err_shift)**2+(fac_mu*err_mu)**2+(fac_sigma*err_sigma)**2)
            except RuntimeWarning:
                t_intersect = mean_t
                del_t_intersect = std_t
                t_fit_info = 'TimeRuntimeWarning: Fit yields unphysical results.'
        warnings.simplefilter("ignore", RuntimeWarning)
        return t_intersect, del_t_intersect, a, b, t_fit_info
    
    def func(self, t, mu, sigma, shift, scale, slope, offset):
        shift_t = (t - shift) # shift t
        res = np.zeros_like(t)
        condition = t>shift
        shift_t_condition = shift_t[condition]
        res[condition] = scale * (0.5 + 0.5 * erf((np.log(shift_t_condition)-mu) / (np.sqrt(2) * sigma)))
        res += slope*(t-self.t0) + offset
        return res
    
    def __call__(self, **kwargs): # event-threshold defines how to get the fitted time
        opt, err = super().__call__(self.func, p0=self.p0, bounds=self.bounds, **kwargs) # maxfev=5e4
        t = np.mean(self.times)
        del_t = np.std(self.times)
        if self.fit_info == '':
            time_fit_results = self.get_time(opt, err, t, del_t)
            self.fit_info += time_fit_results[4]
            tolerance_del_t = 1e9 # 1 s as upper limit for time uncertainty
            if time_fit_results[0]<self.bounds[0][2] or time_fit_results[0]>self.bounds[1][2]:
                self.fit_info += 'TimeToleranceWarning: Estimated time out of bounds.'
            elif time_fit_results[1] > tolerance_del_t:
                self.fit_info += 'TimeToleranceWarning: Estimated time uncertainty exceeds tolerance.'
            else:
                t, del_t = time_fit_results[0:2]
        return t, del_t, self.fit_info, opt

class TemporalFits:
    def __init__(self):
        self.fit_info = ''

class AverageTime(TemporalFits):
    display_name = "Average time"
    description = "Average time of all events."
    def __init__(self):
        super().__init__()

    def __call__(self, events, opt_loc, **kwargs):
        opt = []
        t = np.mean(events['t']*1e-3)
        del_t = np.std(events['t']*1e-3)
        return t, del_t, self.fit_info, opt

class TwoDGaussianFirstTime(TemporalFits):
    display_name = "2D Gaussian (first events)"
    description = "2D Gaussian fit of the first events per pixel."
    def __init__(self):
        super().__init__()
        self.fit = None

    def __call__(self, events, opt_loc, use_weights = True, **kwargs):
        firstTimes = eventDistributions.FirstTimestamp(events)
        first = firstTimes.dist2D
        first_events = firstTimes.get_smallest_t(events)
        self.fit = gauss2D(firstTimes, use_weights=use_weights)
        t, del_t, self.fit_info, opt = self.fit(firstTimes, first_events, opt_loc, **kwargs)
        return t, del_t, self.fit_info, opt
    
class LognormCDFAllEvents(TemporalFits):
    display_name = "Lognormal CDF (all events)"
    description = "Lognormal CDF fit of cumulative sum of all events."
    def __init__(self):
        super().__init__()
        self.fit = None

    def __call__(self, events, opt_loc, **kwargs):
        # np.save("/home/laura/PhD/Event_Based_Sensor_Project/GUI_tests/MLE_fit/data.npy", events)
        # events.to_pickle("/home/laura/PhD/Event_Based_Sensor_Project/GUI_tests/MLE_fit/data.pkl")
        self.fit = lognormal_cdf(events)
        t, del_t, self.fit_info, opt = self.fit()
        return t, del_t, self.fit_info, opt
    
class LognormCDFFirstEvents(TemporalFits):
    display_name = "Lognormal CDF (first events)"
    description = "Lognormal CDF fit of cumulative sum of all first events per pixel."
    def __init__(self):
        super().__init__()
        self.fit = None

    def __call__(self, events, opt_loc, **kwargs):
        firstTimes = eventDistributions.FirstTimestamp(events)
        first_events = firstTimes.get_smallest_t(events)
        first_events = first_events.sort_values(by='t')
        self.fit = lognormal_cdf(first_events)
        t, del_t, self.fit_info, opt = self.fit()
        return t, del_t, self.fit_info, opt
    
class LognormCDFFirstEvents_weighted(TemporalFits):
    display_name = "Lognormal CDF (first events, weighted)"
    description = "Lognormal CDF fit of cumulative sum of all events, each event is weighted by the number of events/pixel."
    def __init__(self):
        super().__init__()
        self.fit = None
        self.sigma = None

    def __call__(self, events, opt_loc, **kwargs):
        firstTimes = eventDistributions.FirstTimestamp(events)
        first_events = firstTimes.get_smallest_t(events)
        first_events = first_events.sort_values(by='t')
        max_weight = np.max(first_events['weight'])
        self.sigma = (max_weight - first_events['weight'] + 1.)/max_weight
        self.fit = lognormal_cdf(first_events)
        t, del_t, self.fit_info, opt = self.fit(sigma=self.sigma)
        return t, del_t, self.fit_info, opt

