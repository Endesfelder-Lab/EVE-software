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
    def __init__(self, events, **kwargs):
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.tlim = [np.min(events['t'])*1e-3, np.max(events['t'])*1e-3]
        self.x_bin_width = 1. # in px
        self.t_bin_width = 10. # in ms
        self.dist2D, self.x_edges, self.y_edges = self(events, **kwargs)

    def set_t_bin_width(self, t_bin_width):
        self.t_bin_width = t_bin_width

    def range(self):
        xrange = self.tlim
        yrange = [self.xlim[0]-0.5, self.xlim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int(np.ceil((self.tlim[1]-self.tlim[0])/self.t_bin_width))
        ybins = int((self.xlim[1]-self.xlim[0]+1)/self.x_bin_width)
        self.tlim[1] = self.tlim[0]+xbins*self.t_bin_width
        return (xbins, ybins)

    def __call__(self, events, **kwargs):
        hist_tx, x_edges, y_edges = np.histogram2d(events['t']*1e-3, events['x'], bins = self.bins(), range = self.range(), **kwargs)
        return hist_tx.T, x_edges, y_edges
    

class Hist2d_ty():
    def __init__(self, events, **kwargs):
        self.ylim = [np.min(events['y']), np.max(events['y'])]
        self.tlim = [np.min(events['t'])*1e-3, np.max(events['t'])*1e-3]
        self.y_bin_width = 1. # in px
        self.t_bin_width = 10. # in ms
        self.dist2D, self.x_edges, self.y_edges = self(events, **kwargs)

    def set_t_bin_width(self, t_bin_width):
        self.t_bin_width = t_bin_width

    def range(self):
        xrange = self.tlim
        yrange = [self.ylim[0]-0.5, self.ylim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int(np.ceil((self.tlim[1]-self.tlim[0])/self.t_bin_width))
        ybins = int((self.ylim[1]-self.ylim[0]+1)/self.y_bin_width)
        self.tlim[1] = self.tlim[0]+xbins*self.t_bin_width
        return (xbins, ybins)

    def __call__(self, events, **kwargs):
        hist_ty, x_edges, y_edges = np.histogram2d(events['t']*1e-3, events['y'], bins = self.bins(), range = self.range(), **kwargs)
        return hist_ty.T, x_edges, y_edges

class Dist2d():
    def __init__(self, events):
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.ylim = [np.min(events['y']), np.max(events['y'])]
        self.tlim = [np.min(events['t']), np.max(events['t'])]

class Hist2d_xy(Dist2d):
    description = "2D histogram of x, y position of all events."
    def __init__(self, events, **kwargs):
        super().__init__(events)
        self.xy_bin_width = 1. # in px
        self.dist2D, self.x_edges, self.y_edges = self(events, **kwargs)

    def range(self):
        xrange = [self.xlim[0]-0.5, self.xlim[1]+0.5]
        yrange = [self.ylim[0]-0.5, self.ylim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int((self.xlim[1]-self.xlim[0]+1)/self.xy_bin_width)
        ybins = int((self.ylim[1]-self.ylim[0]+1)/self.xy_bin_width)
        return (xbins, ybins)

    def __call__(self, events, **kwargs):
        hist_xy, x_edges, y_edges = np.histogram2d(events['x'], events['y'], bins = self.bins(), range = self.range(), **kwargs)
        return hist_xy.T, x_edges, y_edges
    
class SumPolarity(Dist2d):
    description = "The sum of the polarities of all events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.xy_bin_width = 1. # in px
        self.dist2D, self.x_edges, self.y_edges = self(events)
        
    def range(self):
        xrange = [self.xlim[0]-0.5, self.xlim[1]+0.5]
        yrange = [self.ylim[0]-0.5, self.ylim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int((self.xlim[1]-self.xlim[0]+1)/self.xy_bin_width)
        ybins = int((self.ylim[1]-self.ylim[0]+1)/self.xy_bin_width)
        return (xbins, ybins)
    
    def __call__(self, events):
        #Create a histogram of positive events
        pos_events = events[events['p']==1]
        histPos_xy, x_edges, y_edges = np.histogram2d(pos_events['x'], pos_events['y'], bins = self.bins(), range = self.range())
        #Create a histogram of negative events
        neg_events = events[events['p']==0]
        histNeg_xy, x_edges, y_edges = np.histogram2d(neg_events['x'], neg_events['y'], bins = self.bins(), range = self.range())
        #The sum of the polarities of all events for each pixel
        histxy = histPos_xy-histNeg_xy
        
        return histxy.T, x_edges, y_edges

class FirstTimestamp(Dist2d):
    description = "The timestamp of the first event for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)

    def get_smallest_t(self, events):
        smallest_t = events.groupby(['x', 'y'])['t'].min().reset_index()
        return smallest_t
    
    def __call__(self, events):
        smallest_t = self.get_smallest_t(events)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in smallest_t.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d

class AverageTimestamp(Dist2d):
    description = "The average timestamp of all events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)

    def get_average_t(self, events):
        average_t = events.groupby(['x', 'y'])['t'].mean().reset_index()
        return average_t
    
    def __call__(self, events):
        average_t = self.get_average_t(events)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in average_t.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d
    
class MedianTimestamp(Dist2d):
    description = "The median timestamp of all events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)

    def get_median_t(self, events):
        median_t = events.groupby(['x', 'y'])['t'].median().reset_index()
        return median_t
    
    def __call__(self, events):
        median_t = self.get_median_t(events)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in median_t.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d
    
class AverageTimeDiff(Dist2d):
    description = "The average time difference between events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)
    
    def get_averageTimeDiff(self, events):
        TimeDiff = events.groupby(['x', 'y'])['t'].diff().to_frame()
        TimeDiff['x'] = events['x']
        TimeDiff['y'] = events['y']
        averageTimeDiff = TimeDiff.groupby(['x', 'y'])['t'].mean().reset_index()
        return averageTimeDiff
    
    def __call__(self, events):
        averageTimeDiff = self.get_averageTimeDiff(events)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in averageTimeDiff.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d
    
class MinTimeDiff(Dist2d):
    description = "The minimum time difference between events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)
    
    def get_minTimeDiff(self, events):
        TimeDiff = events.groupby(['x', 'y'])['t'].diff().to_frame()
        TimeDiff['x'] = events['x']
        TimeDiff['y'] = events['y']
        minTimeDiff = TimeDiff.groupby(['x', 'y'])['t'].min().reset_index()
        return minTimeDiff
    
    def __call__(self, events):
        minTimeDiff = self.get_minTimeDiff(events)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in minTimeDiff.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d
    
class MaxTimeDiff(Dist2d):
    description = "The maximum time difference between events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)
    
    def get_maxTimeDiff(self, events):
        TimeDiff = events.groupby(['x', 'y'])['t'].diff().to_frame()
        TimeDiff['x'] = events['x']
        TimeDiff['y'] = events['y']
        maxTimeDiff = TimeDiff.groupby(['x', 'y'])['t'].max().reset_index()
        return maxTimeDiff
    
    def __call__(self, events):
        maxTimeDiff = self.get_maxTimeDiff(events)
        # The timestamp of pixels missing in the cluster is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in maxTimeDiff.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event['t']
        return dist2d


def removeCandidatesWithLargeBoundingBox(candidates,xymax,tmax):
    npopped = 0
    for candidate in sorted(candidates, reverse=True):
        if candidates[candidate]['cluster_size'][0] > float(xymax) or candidates[candidate]['cluster_size'][1] > float(xymax) or candidates[candidate]['cluster_size'][2] > float(tmax):
            candidates.pop(candidate)
            #set correct numbering of remaining candidates
            candidates = {index: value for index, value in enumerate(candidates.values(), start=0)}
            npopped += 1
    return candidates, npopped
