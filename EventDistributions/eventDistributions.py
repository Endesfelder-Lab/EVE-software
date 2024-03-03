import numpy as np
import pandas as pd

#-------------------------------------------------------------------------------------------------------------------------------
# Base class definitions
#-------------------------------------------------------------------------------------------------------------------------------
class Dist1d():
    def __init__(self) -> None:
        pass

class TDist(Dist1d):
    def __init__(self, events):
        self.tlim = [np.min(events['t'])*1e-3, np.max(events['t'])*1e-3]

class Dist2d():
    def __init__(self) -> None:
        pass

class XYDist(Dist2d):
    def __init__(self, events):
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.ylim = [np.min(events['y']), np.max(events['y'])]

    def measure(self, xyEventMeasure, measure='t'):
        # The value of pixels missing xyMeasure is set to np.nan
        dist2d=np.ones((self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))*np.nan
        for index, event in xyEventMeasure.iterrows():
            dist2d[int(event['y']-self.ylim[0]),int(event['x']-self.xlim[0])]=event[measure]
        return dist2d

class XTDist(Dist2d):
    def __init__(self, events):
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.tlim = [np.min(events['t'])*1e-3, np.max(events['t'])*1e-3]

class YTDist(Dist2d):
    def __init__(self, events):
        self.ylim = [np.min(events['y']), np.max(events['y'])]
        self.tlim = [np.min(events['t'])*1e-3, np.max(events['t'])*1e-3]

# ToDo: Write a function that checks if a class is in the correct format and all used functions are defined

#-------------------------------------------------------------------------------------------------------------------------------
# Callable derived classes
#-------------------------------------------------------------------------------------------------------------------------------
class Hist1d_t(TDist):
    def __init__(self, events, t_bin_width = 10., **kwargs):
        super().__init__(events)
        self.t_bin_width = t_bin_width # in ms
        self.range = self.range()
        self.bins = self.bins()
        self.dist1D, self.t_edges = self(events, **kwargs)
    
    def set_t_bin_width(self, t_bin_width, events, **kwargs):
        self.t_bin_width = t_bin_width
        self.bins = self.bins()
        self.dist1D, self.t_edges = self(events, **kwargs)

    def range(self):
        return [self.tlim[0], self.tlim[1]]
    
    def bins(self):
        tbins = int(np.ceil((self.tlim[1]-self.tlim[0])/self.t_bin_width))
        self.range[1] = self.tlim[0]+tbins*self.t_bin_width
        return tbins

    def __call__(self, events, **kwargs):
        self.dist1D, self.t_edges = np.histogram(events['t']*1e-3, bins = self.bins, range = tuple(self.range), **kwargs)
        return self.dist1D, self.t_edges

class Hist2d_tx(XTDist):
    def __init__(self, events, t_bin_width = 10.,**kwargs):
        super().__init__(events)
        self.x_bin_width = 1. # in px
        self.t_bin_width = t_bin_width # in ms
        self.range = self.range()
        self.bins = self.bins()
        self.dist2D, self.x_edges, self.y_edges = self(events, **kwargs)

    def set_t_bin_width(self, t_bin_width, events, **kwargs):
        self.t_bin_width = t_bin_width
        self.bins = self.bins()
        self.dist2D, self.x_edges, self.y_edges = self(events, **kwargs)

    def range(self):
        xrange = self.tlim
        yrange = [self.xlim[0]-0.5, self.xlim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int(np.ceil((self.tlim[1]-self.tlim[0])/self.t_bin_width))
        ybins = int((self.xlim[1]-self.xlim[0]+1)/self.x_bin_width)
        self.range[0][1] = self.tlim[0]+xbins*self.t_bin_width
        return (xbins, ybins)

    def __call__(self, events, **kwargs):
        hist_tx, x_edges, y_edges = np.histogram2d(events['t']*1e-3, events['x'], bins = self.bins, range = tuple(self.range), **kwargs)
        return hist_tx.T, x_edges, y_edges
    

class Hist2d_ty(YTDist):
    def __init__(self, events, t_bin_width = 10., **kwargs):
        super().__init__(events)
        self.y_bin_width = 1. # in px
        self.t_bin_width = t_bin_width # in ms
        self.range = self.range()
        self.bins = self.bins()
        self.dist2D, self.x_edges, self.y_edges = self(events, **kwargs)

    def set_t_bin_width(self, t_bin_width, events, **kwargs):
        self.t_bin_width = t_bin_width
        self.bins = self.bins()
        self.dist2D, self.x_edges, self.y_edges = self(events, **kwargs)

    def range(self):
        xrange = self.tlim
        yrange = [self.ylim[0]-0.5, self.ylim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int(np.ceil((self.tlim[1]-self.tlim[0])/self.t_bin_width))
        ybins = int((self.ylim[1]-self.ylim[0]+1)/self.y_bin_width)
        self.range[0][1] = self.tlim[0]+xbins*self.t_bin_width
        return (xbins, ybins)

    def __call__(self, events, **kwargs):
        hist_ty, x_edges, y_edges = np.histogram2d(events['t']*1e-3, events['y'], bins = self.bins, range = tuple(self.range), **kwargs)
        return hist_ty.T, x_edges, y_edges

class Hist2d_xy(XYDist):
    display_name = "2D histogram of x, y positions"
    description = "2D histogram of x, y position of all events."
    def __init__(self, events, **kwargs):
        super().__init__(events)
        self.xy_bin_width = 1. # in px
        self.range = self.range()
        self.bins = self.bins()
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
        hist_xy, x_edges, y_edges = np.histogram2d(events['x'], events['y'], bins = self.bins, range = tuple(self.range), **kwargs)
        return hist_xy.T, x_edges, y_edges
    
class SumPolarity(XYDist):
    display_name = "Sum of polarities"
    description = "The sum of the polarities of all events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.xy_bin_width = 1. # in px
        self.range = self.range()
        self.bins = self.bins()
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
        histPos_xy, x_edges, y_edges = np.histogram2d(pos_events['x'], pos_events['y'], bins = self.bins, range = self.range)
        #Create a histogram of negative events
        neg_events = events[events['p']==0]
        histNeg_xy, x_edges, y_edges = np.histogram2d(neg_events['x'], neg_events['y'], bins = self.bins, range = self.range)
        #The sum of the polarities of all events for each pixel
        histxy = histPos_xy-histNeg_xy
        
        return histxy.T, x_edges, y_edges

class FirstTimestamp(XYDist):
    display_name = "First time/pixel"
    description = "The timestamp of the first event for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.weights = None
        self.dist2D = self(events)
        self.max = 0.

    def get_smallest_t(self, events):
        smallest_t = events.groupby(['x', 'y'])['t'].agg(['min', 'size']).reset_index()
        smallest_t.columns = ['x', 'y', 't', 'weight']
        return smallest_t
    
    def trafo_gauss(self):
        self.max = np.nanmax(self.dist2D)
        self.dist2D = self.dist2D*(-1) + self.max
    
    def undo_trafo_gauss(self, times):
        times = times - self.max
        times = times*(-1)
        return times
    
    def __call__(self, events):
        smallest_t = self.get_smallest_t(events)
        self.weights = super().measure(smallest_t, measure='weight')
        return super().measure(smallest_t)

class AverageTimestamp(XYDist):
    display_name = "Average time/pixel"
    description = "The average timestamp of all events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)

    def get_average_t(self, events):
        average_t = events.groupby(['x', 'y'])['t'].agg(['mean', 'size']).reset_index()
        average_t.columns = ['x', 'y', 't', 'weight']
        return average_t
    
    def __call__(self, events):
        average_t = self.get_average_t(events)
        return super().measure(average_t)
    
class MedianTimestamp(XYDist):
    display_name = "Median time/pixel"
    description = "The median timestamp of all events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)

    def get_median_t(self, events):
        median_t = events.groupby(['x', 'y'])['t'].agg(['median', 'size']).reset_index()
        median_t.columns = ['x', 'y', 't', 'weight']
        return median_t
    
    def __call__(self, events):
        median_t = self.get_median_t(events)
        return super().measure(median_t)
    
class AverageTimeDiff(XYDist):
    display_name = "Average time delay/pixel"
    description = "The average time difference between events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)
    
    def get_averageTimeDiff(self, events):
        TimeDiff = events.groupby(['x', 'y'])['t'].diff().to_frame()
        TimeDiff['x'] = events['x']
        TimeDiff['y'] = events['y']
        averageTimeDiff = TimeDiff.groupby(['x', 'y'])['t'].agg(['mean', 'size']).reset_index()
        averageTimeDiff = averageTimeDiff.rename(columns = {'mean': 't', 'size': 'weight'})
        return averageTimeDiff
    
    def __call__(self, events):
        averageTimeDiff = self.get_averageTimeDiff(events)
        return super().measure(averageTimeDiff)
    
class MinTimeDiff(XYDist):
    display_name = "Minimum time delay/pixel"
    description = "The minimum time difference between events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)
    
    def get_minTimeDiff(self, events):
        TimeDiff = events.groupby(['x', 'y'])['t'].diff().to_frame()
        TimeDiff['x'] = events['x']
        TimeDiff['y'] = events['y']
        minTimeDiff = TimeDiff.groupby(['x', 'y'])['t'].agg(['min', 'size']).reset_index()
        minTimeDiff = minTimeDiff.rename(columns = {'min': 't', 'size': 'weight'})
        return minTimeDiff
    
    def __call__(self, events):
        minTimeDiff = self.get_minTimeDiff(events)
        return super().measure(minTimeDiff)
    
class MaxTimeDiff(XYDist):
    display_name = "Maximum time delay/pixel"
    description = "The maximum time difference between events for each pixel."
    def __init__(self, events):
        super().__init__(events)
        self.dist2D = self(events)
    
    def get_maxTimeDiff(self, events):
        TimeDiff = events.groupby(['x', 'y'])['t'].diff().to_frame()
        TimeDiff['x'] = events['x']
        TimeDiff['y'] = events['y']
        maxTimeDiff = TimeDiff.groupby(['x', 'y'])['t'].agg(['max', 'size']).reset_index()
        maxTimeDiff = maxTimeDiff.rename(columns = {'max': 't', 'size': 'weight'})
        return maxTimeDiff
    
    def __call__(self, events):
        maxTimeDiff = self.get_maxTimeDiff(events)
        return super().measure(maxTimeDiff)