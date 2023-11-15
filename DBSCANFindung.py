import inspect
from Utils import utilsHelper
import sys
# Add /usr/lib/python3/dist-packages/ to PYTHONPATH to include Metavision libraries
sys.path.append("/usr/lib/python3/dist-packages/") 

import os
import numpy as np
from metavision_core.event_io.raw_reader import RawReader

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "LoadEvents": {
            "required_kwargs": [
                {"name": "filepath", "description": "Path to the .raw or .npy file that should be loaded"},
            ],
            "optional_kwargs": [
                {"name": "buffer_size", "description": "buffer size in µs, increase if input too large to be loaded, decrease if PC out of memory, default=1e8"},
                {"name": "time_batches", "description": "time batches in µs, default=50e3 (50ms)"}
            ],
            "help_string": "Load the events from the specified .raw or .npy file"
        },
        "FunctionTwo": {
            "required_kwargs": [
                {"name": "rkwarg_1", "description": "Value(s) to be converted to score"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Score value(s) based on a gaussian profile with given mean, sigma - maximum score of 1 is possible."
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def LoadEvents(**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    if "buffer_size" in provided_optional_args:
        buffer_size = float(kwargs["buffer_size"])
    else:
        #Default buffer_size value
        buffer_size = 1e8
    
    if "time_batches" in provided_optional_args:
        time_batches = float(kwargs["time_batches"])
    else:
        #Default time_batches value
        time_batches = 50e3
    filepath = kwargs["filepath"]
    print('Loading data...')
    if(os.path.exists(filepath[:-4]+'.npy')):
        events = np.load(filepath[:-4]+'.npy')
    else:
        record_raw = RawReader(filepath)
        sums = 0
        time = 0
        events=np.empty
        while not record_raw.is_done() and record_raw.current_event_index() < buffer_size:
            #Load a batch of events
            events_temp = record_raw.load_delta_t(time_batches)
            sums += events_temp.size
            time += time_batches/1e6
            #Add the events in this batch to the big array
            if sums == events_temp.size:
                events = events_temp
            else:
                events = np.concatenate((events,events_temp))
        record_raw.reset()
        # correct the coordinates and time stamps
        events['x']-=np.min(events['x'])
        events['y']-=np.min(events['y'])
        events['t']-=np.min(events['t'])
        np.save(filepath[:-4]+'.npy',events)
    print('Data loaded')
    return events

def FunctionTwo(**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    print('Function Two ran!')
    return (kwargs["rkwarg_1"])
