from Utils import utilsHelper
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import inspect
import numpy as np
import time
from scipy import spatial
from tqdm import tqdm
import imageio
import random
from datetime import datetime
# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "DBSCAN_finding": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Finding via DBSCAN"
        }
    }



#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def DBSCAN_finding(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    print('Hi!')
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    
    nr_batches = 1   #!!!
    consec_weight_max = 30
    spatial_radius_outer = 7 #in pixels, 7 seems to work
    spatial_radius_inner = 1 #in pixels, 1 seems to work
    temporal_duration = 50e3 #in microseconds?, 50e3 seems to work
    posneg = 1 #only pos
    ratio_ms_to_pixel = 35
    
    
    print('Loading events...')
    events = npy_array
    
    #Normalise all axes of the events
    events['t'] = events['t']-min(events['t'])
    events['x'] = events['x']-min(events['x'])
    events['y'] = events['y']-min(events['y'])
    
    print('Creating DataFrame...')
    df_events = pd.DataFrame({'x':events['x'],'y':events['y'],'t':events['t'],'p':events['p'], 'w':0})
    
    #sorting by pixel coordinates
    print('Sorting...')
    df_events = df_events.sort_values(by=['x','y'])
    
    print('Reconverting to array...')
    events = np.ndarray(len(df_events), dtype={'names':['x','y','t','p', 'w'], 'formats':['<u2','<u2','<i8','<i2', '<i2']})
    events['x'] = df_events['x']
    events['y'] = df_events['y']
    events['t'] = df_events['t']
    events['p'] = df_events['p']
    events['w'] = df_events['w']
    
    
    print('Assigning weights...')
    #General idea for the weights assigned in this loop:
        #Discard: w = 0
        #Keep: w > 0
        #Assign cumulative weights to consecutive pos/neg events
    for i in range(len(events)):
        if i == 0 or i == len(events)-1: #if it's the first or last event
            events[i]['w'] = 1 #we'll keep it
        
        elif (events[i]['x'],events[i]['y']) != (events[i-1]['x'],events[i-1]['y']): #if it's a new pixel
        #we don't care about what happened before
            if events[i+1]['p'] == events[i]['p']: #if the polarity of the next event is the same
                events[i]['w'] = 1 #this is the first of at least 2 consecutive events, we'll keep it
        
        elif (events[i]['x'],events[i]['y']) != (events[i+1]['x'],events[i+1]['y']): #if it's the last event at this pixel
        #we don't care what happens after
            if events[i-1]['p'] == events[i]['p']: #if the previous polarity is the same as this one
                events[i]['w'] = events[i-1]['w']+1 #these are consecutive events, add weight
        
        else: 
        #we need to look at the next event and the previous one
            if events[i-1]['p'] == events[i]['p']: #if the previous polarity is the same as this one
                events[i]['w'] = events[i-1]['w']+1 #these are consecutive events, add weight
                
            elif events[i+1]['p'] == events[i]['p']: #if the polarity of the next event is the same
                events[i]['w'] = 1 #this is the first of at least 2 consecutive events, we'll keep it
    
    print('Filtering...')
    #discard all events that weren't assigned a weight above
    events = events[events['w']!=0]
    #also discard events that have too many consecutive events (very likely hot pixels)
    events = events[events['w'] <= consec_weight_max]
    
    
    prob = float("inf")
    
    random_reorder_time=True
    print('Starting noise analysis...') 
    [nr_neighbours_noise, high_count_neighbours_noise] = find_localizations_batching(events,spatial_radius_inner,spatial_radius_outer,temporal_duration,posneg,prob,random_reorder_time,1000*1e3)
    #Should the above function be switched to find_localizations_batching_spatiotemporal since it said in the main file that this one was unused? Would it make much of a difference? #???        
    
    #Calculate the 'level of events' required to be a single molecule: 
    prob = np.mean(nr_neighbours_noise)+2.5*np.std(nr_neighbours_noise)
    print('Prob:', prob)
    random_reorder_time=False

    
    print('\nStarting signal analysis...')
    tic = time.time()
    [nr_neighbours, high_count_neighbours, events_per_batch] = find_localizations_batching_spatiotemporal(events,spatial_radius_inner,spatial_radius_outer,nr_batches,temporal_duration,posneg,prob,random_reorder_time,100)
    print("Time to analyse signal: ", time.time() - tic) 
    
    high_count_neighbours = high_count_neighbours[high_count_neighbours[:, 0].argsort()]
    
    hcn = np.zeros((len(high_count_neighbours),5))
    hcn.dtype = {'names':['t','x','y','p','nr_ev'],'formats':[np.int64,np.int64,np.int64,np.int64,np.int64]}
    hcn['t'] = high_count_neighbours[:,0].reshape(-1,1)
    hcn['x'] = high_count_neighbours[:,1].reshape(-1,1)
    hcn['y'] = high_count_neighbours[:,2].reshape(-1,1)
    hcn['p'] = high_count_neighbours[:,3].reshape(-1,1)
    hcn['nr_ev'] = high_count_neighbours[:,4].reshape(-1,1)
    hcn = np.sort(hcn, axis=0, order=['t', 'x', 'y'])
    
    print('\nStarting DBSCAN...')
    tic = time.time()
    high_count_neighbours_DBSCAN = high_count_neighbours*1 #Multiplying by 1 stops the linkages between these variables...
    #Get a ratio of how many milliseconds should be 1 pixel for DBSCAN purposes
    #did some testing and the best parameter combo seems to be eps = 6, min_samples = 15 & ratio_ms_to_pixel = 35 (test 19) 
    high_count_neighbours_DBSCAN[:,0] /= ratio_ms_to_pixel*1000
    hcnDBS = DBSCAN(eps=6, min_samples=15).fit(high_count_neighbours_DBSCAN) #6/15 seems to be good
    #Filter on found clustesr only
    hcnDBS_exp = hcn[hcnDBS.labels_ > -1]
    
    #Make a new dtype field structure thing that we call b *this is the array with cluster id col*
    new_dt = hcnDBS_exp.dtype.descr + [('DBSCAN',np.int64)]
    b = np.zeros(hcnDBS_exp.shape, dtype=new_dt)
    b['t'] = hcnDBS_exp['t']
    b['x'] = hcnDBS_exp['x']
    b['y'] = hcnDBS_exp['y']
    b['p'] = hcnDBS_exp['p']
    b['nr_ev'] = hcnDBS_exp['nr_ev']
    labels = hcnDBS.labels_[hcnDBS.labels_ > -1]+1 #Start count labels at 1, not 0 (for imageJ purposes)
    b['DBSCAN'] = labels.reshape(-1,1)
    print('Time for DBSCAN: ', time.time()-tic)
    
    # generate the candidates in from of a dictionary
    candidates = {}
    for cluster in np.unique(b['DBSCAN']):
        clusterEvents = pd.DataFrame(b[b['DBSCAN']==cluster])
        candidates[cluster] = {}
        candidates[cluster]['events'] = clusterEvents
        candidates[cluster]['cluster_size'] = [np.max(clusterEvents['y'])-np.min(clusterEvents['y']), np.max(clusterEvents['x'])-np.min(clusterEvents['x']), np.max(clusterEvents['t'])-np.min(clusterEvents['t'])]
        candidates[cluster]['N_events'] = len(clusterEvents)
    
    performance_metadata = ''
    return candidates, performance_metadata


def find_localizations_batching(events,spatial_radius_inner,spatial_radius_outer,temporal_duration,posneg,prob,random_reorder_time,event_batch_time_range):
    '''
    Should call find_localizations with a range of events (given by event_nr_lookup), of which the central part (event_nr_analyse) is analysed.

    Parameters
    ----------
    events : Numpy array
        Specially formatted numpy array with x, y, p (polarity -> -1 for negative events and 1 for positive events) and t (time in μs) columns.
    spatial_radius_inner : int
        Small radius (in pixels) we want to ignore when counting neighbouring localizations.
    spatial_radius_outer : int
        Larger radius (in pixels) beyond which we do not want to count neighbours.
    temporal_duration : int
        The span of time (in μs) in which we will look for neighbouring events (only into the future)#??? But is it actually?
    posneg : int
        -1 for negative and 1 for positive events.
    prob : float
        If an event has fewer than this many neighbours, it is considered to be noise and disregarded.
    random_reorder_time : bool
        True or False depending on whether the user wishes to randomly reorder the time (used for prob calculation).
    event_batch_time_range : int
        The amount of time (in μs) we want in each temporal batch.

    Returns
    -------
    nr_neighbours_full : Numpy array
        An array with a single column containing the number of neighbours (exculding the point itself) for each event being analysed.  
    high_count_neighbours_full : Numpy array
        Numpy array with t (time in μs), x, y, p (-1 for negative and 1 for positive events) and nr_ev (the number of neighbours for the event, including itself) columns.

    '''

    #Get the event ID at the center of time
    eventid_center = int(np.floor(len(events)/2))
    #Find the event where the time is at the center - time/2 - temp_duration/2
    eventid_start = events['t']>=(events[eventid_center]['t']-event_batch_time_range*0.5-temporal_duration/2)
    #Find the event where the time is at the center + time/2 + temp_duration/2
    eventid_end = events['t']<=(events[eventid_center]['t']+event_batch_time_range*0.5+temporal_duration/2)
    events = events[eventid_start*eventid_end]
    #eventnrstart = int(np.floor(len(events)/2)-(event_nr_lookup-event_nr_analyse)/2)
    #events = events[eventnrstart:eventnrstart+event_nr_lookup]
    
    random.shuffle(events['t'])
    
    [points, tree, posevents] = make_kdtree(events, posneg, temporal_duration, spatial_radius_outer)
    
    [nr_neighbours_full, high_count_neighbours_full] = find_localizations_kd(points,spatial_radius_inner,spatial_radius_outer,temporal_duration,posneg,prob,random_reorder_time, tree, posevents)

    return nr_neighbours_full, high_count_neighbours_full


def make_kdtree(events, posneg, temporal_duration, spatial_radius_outer):
    '''
    Filters the events by posnegativity, recreates a new structured array and converts the time column from μs to pixels, then creates a KDTree with all events.

    Parameters
    ----------
    events : numpy array
        Specially formatted numpy array with x, y, p (polarity -> -1 for negative events and 1 for positive events) and t (time in μs) columns.
    posneg : int
        -1 for negative and 1 for positive events.
    temporal_duration : int
        The span of time (in μs) in which we will look for neighbouring events (only into the future)#??? But is it actually?
    spatial_radius_outer : int
        Larger radius (in pixels) beyond which we do not want to count neighbours.

    Returns
    -------
    points : Numpy array
        Array of events, each of which is a 1x3 array of the form [x, y, t (in pixels)].
    tree : KDTree
        KDTree of all events in points.
    posevents : Numpy array
        Array of events, each of which is a 1x3 np.void object of the form (x, y, t (in μs)).

    '''
    
    #Filter events on posnegativity
    #events will have time converted to pixels but events_micros will be kept in microseconds
    posevents = events[events['p']==posneg]

    #Remaking an array here because the times will NOT cast to floats no matter what I do
    pos = np.ndarray(shape=(len(posevents),1), dtype={'names':['x','y','p','t'], 'formats':['<i8','<i8','<i8','<f8']})
    pos['x'] = posevents['x'].reshape(-1,1)
    pos['y'] = posevents['y'].reshape(-1,1)
    pos['p'] = posevents['p'].reshape(-1,1)
    pos['t'] = posevents['t'].reshape(-1,1)
    pos = np.sort(pos, axis=0, order=['t', 'x', 'y'])

    #dividing by spatial_radius_outer because that's the number of pixels we're going to search in the KDTree so we need the temporal dimension to be in the same units
    ratio_micros_to_pixel = temporal_duration/spatial_radius_outer
    #converting the time col from microseconds to pixels
    pos['t'] /= ratio_micros_to_pixel

    points = np.c_[pos['x'].ravel(), pos['y'].ravel(), pos['t'].ravel()]
    #now points is an array of 1x3 subarrays, where each subarray is an event coordinate of the form [x y t]
    
    #Making a tree
    tree = spatial.cKDTree(points)
    
    return points, tree, posevents

def find_localizations_kd(search, spatial_radius_inner, spatial_radius_outer, temporal_duration, posneg, prob, random_reorder_time, tree, posevents): 
    '''
    Run over all the events and calculate the number of 'event neighbours' within a given radius and time
    Idea: look at every event, and get the nr of events in the spatiotemporal 'area' provided by spatial_radius (in both directions), and temporal_duration (only in the future) #??? but is it actually?!
    HOWEVER, we do NOT count the same pixel in here - i.e. hot pixels are undetected

    Parameters
    ----------
    search : Numpy array
        Array of events, each of which is a 1x3 array of the form [x, y, t].  These are the events that will be searched for neighbours in the tree.
    spatial_radius_inner : int
        Small radius (in pixels) we want to ignore when counting neighbouring localizations.
    spatial_radius_outer : int
        Larger radius (in pixels) beyond which we do not want to count neighbours.
    temporal_duration : int
        The span of time (in μs) in which we will look for neighbouring events (only into the future)#??? But is it actually?
    posneg : int
        -1 for negative and 1 for positive events.
    prob : float
        If an event has fewer than this many neighbours, it is considered to be noise and disregarded.
    random_reorder_time : bool
        True or False depending on whether the user wishes to randomly reorder the time (used for prob calculation).
    tree : KDTree
        KDTree of all events being analysed in the complete dataset.
    posevents : Numpy array
        Array of events, each of which is a 1x3 np.void object of the form (x, y, t (in μs)).

    Returns
    -------
    nr_neighbours_kd : Numpy array
        Array with a single column containing the number of neighbours (exculding the point itself) for each event in search.  
    high_count_neighbours_kd : Numpy array
        Numpy array with t (time in μs), x, y, p (-1 for negative and 1 for positive events) and nr_ev (the number of neighbours for the event, including itself) columns.

    '''
    ratio_micros_to_pixel = temporal_duration/spatial_radius_outer

    #Performing the actual range searches (one for the outer radius and one for the inner)
    #tic0 = time.time()
    neighbours = tree.query_ball_point(search, spatial_radius_outer, workers = 8)
    #print('Outer search: ', time.time()-tic0)
    #tic1 = time.time()
    neighbours_in = tree.query_ball_point(search, spatial_radius_inner, workers = 8)
    #print('Inner search: ', time.time()-tic1)


    #tic3 = time.time()
    nr_neighbours_kd = np.zeros(len(search))
    high_count_neighbours_kd = np.zeros((len(search), 5))
    hcncounter = 0
    for i in range(len(neighbours)):
        if np.mod(i,5000) == 0:
            if i > 0:
                print(["Currently on event: " + str(i) + " out of " + str(len(search))])
        #First filtering out the neighbours within the inner radius
        neighbours[i] = [x for x in neighbours[i] if x not in neighbours_in[i]]
        #Now counting the remaining events (neighbours within the spatiotemporal doughnut)
        nr_neighbours_kd[i] = len(neighbours[i])
        if nr_neighbours_kd[i] > prob:
            #I'm pretty sure that no temporal information is lost by reconverting to microseconds here
            high_count_neighbours_kd[hcncounter, :] = (int(search[i,2]*ratio_micros_to_pixel), search[i,0], search[i,1], posneg, nr_neighbours_kd[i]+1)
                                                       #time in μs^                                x^           y^                  
            hcncounter += 1
    high_count_neighbours_kd = high_count_neighbours_kd[0:hcncounter,:]
    #print('Formatting output ', time.time()-tic3)


    #print("TOTAL KD LOOP RUN TIME: ", time.time()-tic)

    return nr_neighbours_kd, high_count_neighbours_kd


def find_localizations_batching_spatiotemporal(events, spatial_radius_inner, spatial_radius_outer, nr_batches, temporal_duration, posneg, prob, random_reorder_time, event_batch_xypx_range):
    """
    Should call find_localizations with a range of events (given by event_nr_lookup), of which the central part (event_nr_analyse) is analysed.

    Parameters
    ----------
    events : numpy array
        Specially formatted numpy array with x, y, p (polarity -> -1 for negative events and 1 for positive events) and t (time in μs) columns.
    spatial_radius_inner : int
        Small radius (in pixels) we want to ignore when counting neighbouring localizations.
    spatial_radius_outer : int
        Larger radius (in pixels) beyond which we do not want to count neighbours.
    nr_batches: int
        The number of temporal batches into which the data will be split for the analysis step.
    temporal_duration : int
        The span of time (in μs) in which we will look for neighbouring events (only into the future)#??? But is it actually?
    posneg : int
        -1 for negative and 1 for positive events.
    prob : float
        If an event has fewer than this many neighbours, it is considered to be noise and disregarded.
    random_reorder_time : bool
        True or False depending on whether the user wishes to randomly reorder the time (used for prob calculation).
    event_batch_xypx_range : int
        The number of pixels in a single spatial dimention that are part of a batch.

    Returns
    -------
    nr_neighbours_full : Numpy array
        An array with a single column containing the number of neighbours (exculding the point itself) for each event being analysed.  
    high_count_neighbours_full : Numpy array
        Numpy array with t (time in μs), x, y, p (-1 for negative and 1 for positive events) and nr_ev (the number of neighbours for the event, including itself) columns.

    """
    
    print('Making tree...')
    [points, tree, posevents] = make_kdtree(events, posneg, temporal_duration, spatial_radius_outer)
    #ADD BATCH OVERLAP
    #calculating the number of events per batch
    events_per_batch = int(np.ceil(len(posevents)/nr_batches))
    print(f'{events_per_batch} events per batch')

    nr_neighbours_full = np.zeros((1,1))
    high_count_neighbours_full = np.zeros((1,5))

    for batch_num in range(nr_batches):
        tic = time.time()
        start_index = batch_num*events_per_batch+1 #the +1 avoids double-counting the last end event as the next start event
        #need to add this conditional so we don't miss the 0th event
        if batch_num == 0:
            start_index = 0
        end_index = min(len(posevents), (batch_num+1)*events_per_batch)
        
        #defining the batch of points to be analyzed
        batch = points[start_index:end_index+1] #include end_index
        
        [nr_neighbours, high_count_neighbours] = find_localizations_kd(batch,spatial_radius_inner,spatial_radius_outer,temporal_duration,posneg,prob,random_reorder_time, tree, posevents)
        nr_neighbours_full = np.append(nr_neighbours_full,nr_neighbours)
        high_count_neighbours_full = np.append(high_count_neighbours_full,high_count_neighbours,axis=0)
        print(f"Batch {batch_num+1} Time: ", time.time()-tic)

    #Remove first indices that are zeros
    nr_neighbours_full=nr_neighbours_full[1:]
    high_count_neighbours_full=high_count_neighbours_full[1:]

    return nr_neighbours_full, high_count_neighbours_full, events_per_batch
