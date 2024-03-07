#Quick util that cuts a hdf5 based on specific x/y values
import h5py
import numpy as np


# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "CutHDF_xy": {
            "required_kwargs": [
                {"name": "frame_time_for_dme", "description": "Frame-time used for drift-correction (in ms)","default":100.,"type":float,"display_text":"Frame time used in DME"},
                {"name": "frames_per_bin", "description": "Number of frames in every bin for dme drift correction ","default":50,"type":int,"display_text":"Frames per bin"},
                {"name": "visualisation", "description": "Visualisation of the drift traces.","default":True,"display_text":"Visualisation"},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Drift correction from Cnossen et al.."
        }
    }
    
def CutHDF_xy(**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    filefolder = "//Smi2pc/e/Data/Koen/20240222/DNAPAINT_fullGreen_cont_mirror_TIRF_cleanup561/"
    filename = "output.hdf5"
    dataLocation = filefolder+filename
    xyStretch=(400,450,400,450)
    #Get the time slice from a hdf5 file for index, after running findIndexFromTimeSliceHDF
    print('Starting to read')
    with h5py.File(dataLocation, mode='r') as file:
        events = file['CD']['events']
        #filter out all events that are not in the xyStretch:
        events = events[(events['x'] >= xyStretch[0]) & (events['x'] <= xyStretch[1]) & (events['y'] >= xyStretch[2]) & (events['y'] <= xyStretch[3])]
    
    print('Starting to save')
    #Store these events as a new hdf5:
    newsavename = dataLocation.replace('.hdf5','_xyCut.hdf5')
    with h5py.File(newsavename, mode='r+') as file:
        events = file.create_dataset('CD/events', data=events, compression="gzip")
        
    return events


# print('starting cuthdfxy')
# CutHDF_xy(filefolder+filename,xyStretch=(400,450,400,450))
