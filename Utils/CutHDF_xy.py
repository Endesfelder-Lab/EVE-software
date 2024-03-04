#Quick util that cuts a hdf5 based on specific x/y values
import h5py
import numpy as np


def CutHDF_xy(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
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

filefolder = "//Smi2pc/e/Data/Koen/20240222/DNAPAINT_fullGreen_cont_mirror_TIRF_cleanup561/"
filename = "output.hdf5"
print('starting cuthdfxy')
CutHDF_xy(filefolder+filename,xyStretch=(400,450,400,450))
