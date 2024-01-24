import warnings
import pandas as pd
import numpy as np

def argumentChecking(dictionaryInfo, function_name, given_kwargs):
    #Checks whether named required and optional arguments are given or not, and gives errors when things are missing.

    #Get required kwargs and throw error if any are missing
    required_args = [req_kwarg["name"] for req_kwarg in dictionaryInfo[function_name]["required_kwargs"]]
    missing_req_args = [arg for arg in required_args if arg not in given_kwargs]
    if missing_req_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_req_args)} in function {str(function_name)}")
    
    #Get optional kwargs and throw warning if any are missing
    optional_args = [req_kwarg["name"] for req_kwarg in dictionaryInfo[function_name]["optional_kwargs"]]
    missing_opt_args = [arg for arg in optional_args if arg not in given_kwargs]
    provided_opt_args = [arg for arg in optional_args if arg in given_kwargs]
    if missing_opt_args:
        warnings.warn(f"Unused optional arguments: {', '.join(missing_opt_args)} in function {str(function_name)}")
    
    #Return the provided and missing optional arguments
    return [provided_opt_args, missing_opt_args]

# Convert a string representation of truth to true (1) or false (0), or raise an exception
def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true','on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))
    
class Hist2d_tx():
    def __init__(self, candidate, **kwargs):
        self.xlim = [np.min(candidate['events']['x']), np.max(candidate['events']['x'])]
        self.ylim = [np.min(candidate['events']['y']), np.max(candidate['events']['y'])]
        self.tlim = [np.min(candidate['events']['t']), np.max(candidate['events']['t'])]
        self.x_bin_width = 1. # in px
        self.t_bin_width = 10. # in ms
        self.dist2D, self.x_edges, self.y_edges = self(candidate, **kwargs)

    def range(self):
        xrange = self.tlim
        yrange = [self.xlim[0]-0.5, self.xlim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int(np.ceil((self.tlim[1]-self.tlim[0])/self.t_bin_width))
        ybins = int((self.xlim[1]-self.xlim[0]+1)/self.x_bin_width)
        self.tlim[1] = self.tlim[0]+xbins*self.t_bin_width
        return (xbins, ybins)

    def __call__(self, candidate, **kwargs):
        hist_tx, x_edges, y_edges = np.histogram2d(candidate['events']['t']*1e-3, candidate['events']['x'], bins = self.bins(), range = self.range(), **kwargs)
        return hist_tx.T, x_edges, y_edges
    

class Hist2d_ty():
    def __init__(self, candidate, **kwargs):
        self.xlim = [np.min(candidate['events']['x']), np.max(candidate['events']['x'])]
        self.ylim = [np.min(candidate['events']['y']), np.max(candidate['events']['y'])]
        self.tlim = [np.min(candidate['events']['t']), np.max(candidate['events']['t'])]
        super().__init__(candidate)
        self.y_bin_width = 1. # in px
        self.t_bin_width = 10. # in ms
        self.dist2D, self.x_edges, self.y_edges = self(candidate, **kwargs)

    def range(self):
        xrange = self.tlim
        yrange = [self.ylim[0]-0.5, self.ylim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int(np.ceil((self.tlim[1]-self.tlim[0])/self.t_bin_width))
        ybins = int((self.ylim[1]-self.ylim[0]+1)/self.y_bin_width)
        self.tlim[1] = self.tlim[0]+xbins*self.t_bin_width
        return (xbins, ybins)

    def __call__(self, candidate, **kwargs):
        hist_ty, x_edges, y_edges = np.histogram2d(candidate['events']['t']*1e-3, candidate['events']['y'], bins = self.bins(), range = self.range(), **kwargs)
        return hist_ty.T, x_edges, y_edges

class Dist2d():
    def __init__(self, candidate):
        self.xlim = [np.min(candidate['events']['x']), np.max(candidate['events']['x'])]
        self.ylim = [np.min(candidate['events']['y']), np.max(candidate['events']['y'])]
        self.tlim = [np.min(candidate['events']['t']), np.max(candidate['events']['t'])]

class Hist2d_xy(Dist2d):
    description = "2D histogram of x, y position of all events."
    def __init__(self, candidate, **kwargs):
        super().__init__(candidate)
        self.xy_bin_width = 1. # in px
        self.dist2D, self.x_edges, self.y_edges = self(candidate, **kwargs)

    def range(self):
        xrange = [self.xlim[0]-0.5, self.xlim[1]+0.5]
        yrange = [self.ylim[0]-0.5, self.ylim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int((self.xlim[1]-self.xlim[0]+1)/self.xy_bin_width)
        ybins = int((self.ylim[1]-self.ylim[0]+1)/self.xy_bin_width)
        return (xbins, ybins)

    def __call__(self, candidate, **kwargs):
        hist_xy, x_edges, y_edges = np.histogram2d(candidate['events']['x'], candidate['events']['y'], bins = self.bins(), range = self.range(), **kwargs)
        return hist_xy.T, x_edges, y_edges
    
class FirstTimestamp(Dist2d):
    description = "The timestamp of the first event for each pixel."
    def __init__(self, candidate):
        super().__init__(candidate)
        self.dist2D = self(candidate)

    def get_smallest_t(self, candidate):
        smallest_t = candidate['events'].groupby(['x', 'y'])['t'].min().reset_index()
        return smallest_t
    
    def __call__(self, candidate):
        smallest_t = self.get_smallest_t(candidate)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in smallest_t.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d

class AverageTimestamp(Dist2d):
    description = "The average timestamp of all events for each pixel."
    def __init__(self, candidate):
        super().__init__(candidate)
        self.dist2D = self(candidate)

    def get_average_t(self, candidate):
        average_t = candidate['events'].groupby(['x', 'y'])['t'].mean()
        return average_t
    
    def __call__(self, candidate):
        average_t = self.get_average_t(candidate)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in average_t.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        print(dist2d.shape)
        return dist2d
    
class MedianTimestamp(Dist2d):
    description = "The median timestamp of all events for each pixel."
    def __init__(self, candidate):
        super().__init__(candidate)
        self.dist2D = self(candidate)

    def get_median_t(self, candidate):
        median_t = candidate['events'].groupby(['x', 'y'])['t'].median()
        return median_t
    
    def __call__(self, candidate):
        median_t = self.get_median_t(candidate)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in median_t.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d
    
class AverageTimeDiff(Dist2d):
    description = "The average time difference between all events for each pixel."
    def __init__(self, candidate):
        super().__init__(candidate)
        self.dist2D = self(candidate)
    
    def get_averageTimeDiff(self, candidate):
        averageTimeDiff = candidate['events'].groupby(['x', 'y'])['t'].diff().mean()
        return averageTimeDiff
    
    def __call__(self, candidate):
        averageTimeDiff = self.get_averageTimeDiff(candidate)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in averageTimeDiff.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        # Maybe use this code instead (suggestion of Codeium)
        # y_indices = (averageTimeDiff['y'] - self.ylim[0]).astype(int)
        # x_indices = (averageTimeDiff['x'] - self.xlim[0]).astype(int)
        # dist2d[y_indices, x_indices] = averageTimeDiff['t']
        return dist2d

