
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
    from Utils import utilsHelper
from sklearn.cluster import DBSCAN
import scipy.ndimage
from scipy import spatial
import numpy as np
import time, logging
import pandas as pd
import inspect

import gc
from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "FrameBased_finding": {
            "required_kwargs": [
                {"name": "threshold_detection", "description": "Threshold for wavelet detection","default":3.,"display_text":"Detection threshold"},
                {"name": "exclusion_radius", "description": "Radius of the exclusion area (if two or more PSFs are closer than twice the value, they will all be discarded) (in pixels)","default":4.,"display_text":"Exclusion radius"},
                {"name": "min_diameter", "description": "Minimum radius of the thresholded area (in pixels)","default":1.25,"display_text":"Min. radius"},
                {"name": "max_diameter", "description": "Maximum radius of the thresholded area (in pixels)","default":4.,"display_text":"Max. radius"},
                {"name": "frame_time", "description": "frame time in (ms)", "default":100.,"display_text":"Frame time (ms)"},
                {"name": "candidate_radius", "description": "Radius of the area around the localization (in px)","default":4.,"display_text":"Candidate radius"},
            ],
            "optional_kwargs": [
                {"name": "multithread","description": "True to use multithread parallelization; False not to.","default":True,"display_text":"Multithreading"},
            ],
            "help_string": "Convert event data to frames and perform candidate finding via wavelet detection",
            "display_name": "Frame Based finding"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

class Frame():
    def __init__(self, events):
        self.xlim = [np.min(events['x']), np.max(events['x'])]
        self.ylim = [np.min(events['y']), np.max(events['y'])]
        self.xy_bin_width = 1. # in px
        self.range = self.range()
        self.bins = self.bins()

    def range(self):
        xrange = [self.xlim[0]-0.5, self.xlim[1]+0.5]
        yrange = [self.ylim[0]-0.5, self.ylim[1]+0.5]
        return [xrange, yrange]
    
    def bins(self): 
        xbins = int((self.xlim[1]-self.xlim[0]+1)/self.xy_bin_width)
        ybins = int((self.ylim[1]-self.ylim[0]+1)/self.xy_bin_width)
        return (xbins, ybins)

    def __call__(self, events, times, **kwargs):
        msk=(events['t']>=times[0])*(events['t']<times[1])
        sub_events=events[msk]
        frame, x_edges, y_edges = np.histogram2d(sub_events['x'], sub_events['y'], bins = self.bins, range = self.range, **kwargs)
        return frame.T
    
# wavelet detection
def wavelet_detection(frame, kernel1, kernel2, kernel, threshold_detection):
    """
    Perform difference-of-wavelet detection via scipy's convolve
    """
    V1=scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(frame,kernel1,axis=1),kernel1,axis=0)
    V2=scipy.ndimage.convolve1d(scipy.ndimage.convolve1d(V1,kernel2,axis=1),kernel2,axis=0)
    Wavelet2nd=V1-V2
    Wavelet2nd-=scipy.ndimage.convolve(Wavelet2nd,kernel)
    Wavelet2nd*=(Wavelet2nd>=0)
    image_to_label=(Wavelet2nd>=threshold_detection*np.std(Wavelet2nd))*Wavelet2nd
    return image_to_label

# PSF detection
def detect_PSFs(frame, x0, y0, min_diameter, max_diameter, exclusion_radius, candidate_radius, kernel1, kernel2, kernel, threshold_detection):# verify exceptions for zero detections
    """
    General PSF detection function
    """
    # Generate image to label
    image_to_label=wavelet_detection(frame, kernel1, kernel2, kernel, threshold_detection)
    
    # Create labels
    labels,nb_labels=scipy.ndimage.label(image_to_label)
    label_list=np.arange(nb_labels)+1
    area_sizes=scipy.ndimage.sum((image_to_label>0),labels,index=label_list)
    
    # Filter labels by size
    msk=(area_sizes>(min_diameter*2+1)**2)*(area_sizes<(max_diameter*2+1)**2)
    label_list=label_list[msk]
    labels*=np.isin(labels,label_list)
    nb_labels=len(label_list)
    
    # Calculate center of mass
    coordinates_COM=scipy.ndimage.center_of_mass(image_to_label,labels,label_list)
    coordinates_COM=np.asarray(coordinates_COM)
    if len(np.shape(coordinates_COM))==1:
        if np.shape(coordinates_COM)[0]==0:
            coordinates_COM=np.zeros((0,2))
    
    # Filter coordinates by distance to each other
    msk=np.ones(nb_labels,dtype=bool)
    mgy1,mgy2=np.meshgrid(coordinates_COM[:,0],coordinates_COM[:,0].transpose())
    mgx1,mgx2=np.meshgrid(coordinates_COM[:,1],coordinates_COM[:,1].transpose())
    msk2=(np.abs(mgy1-mgy2)<2*exclusion_radius)*(np.abs(mgx1-mgx2)<2*exclusion_radius)
    np.fill_diagonal(msk2,False)
    ind=np.nonzero(msk2)
    ind=ind[0].tolist()+ind[1].tolist()
    ind=list(set(ind))
    msk[ind]=False
    label_list=label_list[msk]
    labels*=np.isin(labels,label_list)
    nb_labels=len(label_list)
    coordinates_COM=coordinates_COM[msk]
    
    # Filter coordinates by distance to the frame edges
    if nb_labels>0:
        msk=(coordinates_COM[:,0]>candidate_radius*2)*(coordinates_COM[:,1]>candidate_radius*2)*(coordinates_COM[:,0]<np.shape(frame)[0]-candidate_radius*2)*(coordinates_COM[:,1]<np.shape(frame)[1]-candidate_radius*2)
        label_list=label_list[msk]
        labels*=np.isin(labels,label_list)
        nb_labels=len(label_list)
        coordinates_COM=coordinates_COM[msk]
        
    if len(np.shape(coordinates_COM))==1:
        coordinates_COM=np.zeros((0,2))
        
    x=coordinates_COM[:,1]
    y=coordinates_COM[:,0]
    ROIs=np.zeros((np.shape(x)[0],3))
    ROIs[:,0]=y+y0
    ROIs[:,1]=x+x0
    
    return ROIs

# Process frame (get all ROIs in frame)
def process_frame(frame, x0, y0, times, min_diameter, max_diameter, exclusion_radius, candidate_radius, kernel1, kernel2, kernel, threshold_detection):    
    """
    Process a single frame
    """
    ROIs=detect_PSFs(frame, x0, y0, min_diameter, max_diameter, exclusion_radius, candidate_radius, kernel1, kernel2, kernel, threshold_detection)
    ROIs[:,2]=times[0]
    return ROIs


def generate_candidate(events, ROI, frame_time, candidate_radius):
    """
    # Generate single candidate dictionary entry for specified ROI + parameter
    """
    msk=(events['t']>=ROI[2])*(events['t']<ROI[2]+frame_time)
    rho=(events['y']-ROI[0])**2+(events['x']-ROI[1])**2
    msk*=(rho<=candidate_radius**2)
    sub_events=events[msk]
    candidate = {}
    candidate['events'] = pd.DataFrame(sub_events)
    candidate['cluster_size'] = [np.max(sub_events['y'])-np.min(sub_events['y'])+1, np.max(sub_events['x'])-np.min(sub_events['x'])+1, np.max(sub_events['t'])-np.min(sub_events['t'])]
    candidate['N_events'] = len(sub_events)
    return candidate


def nb_jobs(events, num_cores, frame_time):
    """
    # Calculate number of jobs on CPU
    """
    time_tot = events['t'][-1]-events['t'][0]
    time_base = frame_time
    if time_tot < frame_time or num_cores == 1:
        njobs = 1
        num_cores = 1
        time_base = time_tot
    elif time_tot/frame_time > num_cores:
        njobs = np.int64(np.ceil(time_tot/frame_time))
    else:
        njobs = np.int64(np.ceil(time_tot/frame_time))
        num_cores = njobs
    return njobs, num_cores, time_base

def slice_data(events, num_cores, frame_time):
    """
    # Slice data to distribute the computation on several cores
    """
    njobs, num_cores, time_slice = nb_jobs(events, num_cores, frame_time)
    data_split=[]
    t_min = events['t'][0]
    t_max = events['t'][-1]
    for k in np.arange(njobs):
        times=[t_min+k*time_slice, min(t_min+(k+1)*time_slice, t_max)]
        msk = (events['t']>=times[0])*(events['t']<times[1])
        data_split.append(events[msk])
    return data_split, njobs, num_cores

def compute_thread(i, sub_events, frame_time, candidate_radius, min_diameter, max_diameter, exclusion_radius, kernel1, kernel2, kernel, threshold_detection, single_frame):
    """
    # Candidate finding routine on a single core
    """
    print('Finding candidates (thread '+str(i)+')...')
    time_max=np.max(sub_events['t'])
    time_min=np.min(sub_events['t'])
    nb_frames=int(np.ceil((time_max-time_min)/frame_time))
    ROIs=[]
    for k in np.arange(nb_frames):
        times=[time_min+k*frame_time,time_min+(k+1)*frame_time]
        # load all events in frame
        msk=(sub_events['t']>=times[0])*(sub_events['t']<times[1])
        events_loaded=sub_events[msk]
        # Detect PSFs
        frame=single_frame(events_loaded, times)
        x0,y0=single_frame.xlim[0],single_frame.ylim[0]
        ROIs.append(process_frame(frame, x0, y0, times, min_diameter, max_diameter, exclusion_radius, candidate_radius, kernel1, kernel2, kernel, threshold_detection))
    ROIs=np.vstack(ROIs)
    gc.collect()
    
    candidates = {}
    dict_index = 0
    # Generating candidate dictionary
    for k in np.arange(np.shape(ROIs)[0]):
        # Loading all events in time limits
        msk = (sub_events['t']>=ROIs[k][2])*(sub_events['t']<ROIs[k][2]+frame_time)
        events_loaded = sub_events[msk]
        key_list = list(candidates.keys())
        if len(candidates)!=0:
            dict_index = max(key_list)+1
        candidates[dict_index]=generate_candidate(events_loaded, ROIs[k,:], frame_time, candidate_radius)
    print('Finding candidates (thread '+str(i)+') done!')
    return candidates


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def FrameBased_finding(npy_array,settings,**kwargs):
    """
    General FrameBased finding  function
    """
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate finding...")
    # Load the required kwargs
    threshold_detection = float(kwargs['threshold_detection'])
    exclusion_radius = float(kwargs['exclusion_radius'])
    min_diameter = float(kwargs['min_diameter'])
    max_diameter = float(kwargs['max_diameter'])
    frame_time = float(kwargs['frame_time'])*1000.
    candidate_radius = float(kwargs['candidate_radius'])

    # Load the optional kwargs
    multithread = utilsHelper.strtobool(kwargs['multithread'])

    # Initializations - general
    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1

    # Initializations - wavelet
    kernel1=np.array([0.0625,0.25,0.375,0.25,0.0625])
    kernel2=np.array([0.0625,0,0.375,0,0.25,0,0.0625])
    kernel_size=8
    kernel=np.ones((kernel_size,kernel_size))/(kernel_size**2.)
    
    # Find all candidates
    candidates = {}
    index = 0
    candidates_info = ''

    single_frame = Frame(npy_array)
    events_split, njobs, num_cores = slice_data(npy_array, num_cores, frame_time)
    print("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(compute_thread)(i, events_split[i], frame_time, candidate_radius, min_diameter, max_diameter, exclusion_radius, kernel1, kernel2, kernel, threshold_detection, single_frame) for i in range(len(events_split)))
    for i in range(len(RES)):
        for candidate in RES[i].items():
            candidates[index] = candidate[1]
            index+=1
    
    #Remove small/large bounding-box data
    candidates, _, _ = utilsHelper.removeCandidatesWithLargeSmallBoundingBox(candidates,settings)

    performance_metadata = candidates_info
    return candidates, performance_metadata