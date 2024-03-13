
#Routine to check for hot pixels in the .raw dataset
import h5py
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from metavision_core.event_io.raw_reader import RawReader

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "detectHotPixels": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": ""
        }
    }

def detectHotPixels_run(loadfile,maxTime,outputArea):
    maxTime=eval(maxTime)
    #Check if loadfile ends in .raw:
    if loadfile.endswith('.raw'):
        print('Loading RAW data...')
        buffer_size=1e9
        time_batches=50e3
        curr_time = 0
        record_raw = RawReader(loadfile)
        sums = 0
        while not record_raw.is_done() and record_raw.current_event_index() < buffer_size and curr_time <= maxTime*1000:
            events_temp = record_raw.load_delta_t(time_batches)
            sums += events_temp.size
            print("Loaded batch" + str(sums))
            # print(events.size)
            print(curr_time)
            curr_time += time_batches
            if sums == events_temp.size:
                events = events_temp
            else:
                events = np.concatenate((events,events_temp))
        record_raw.reset()
    elif loadfile.endswith('.hdf5'):
        print('Loading HDF5 data...')
        with h5py.File(loadfile, mode='r') as file:
            events = file['CD']['events']
            #only find events that are in the maxTime:
            events = events[events['t'] <= maxTime*1000]
    else:
        print('File type not supported')
    
    
    #Now that we have the events, lets find out if we have hot pixels
    minx = min(events['x'])
    maxx = max(events['x'])
    miny = min(events['y'])
    maxy = max(events['y'])
    totalEvents = np.zeros((maxx-minx,maxy-miny))

    x_indices = events['x'] - minx
    for x in tqdm(range(minx, maxx)):
        pixelRow = events[(x_indices == x)]
        for y in range(miny, maxy):
            y_indices = pixelRow['y'] - miny
            pixelCol = pixelRow[(y_indices == y)]
            totalEvents[x - minx, y - miny] = len(pixelCol)

    meanTotEvents = np.mean(totalEvents)
    stdTotEvents = np.std(totalEvents)
    #Find all indexes where the total events is superhigh
    hotPixels = np.where(totalEvents > (meanTotEvents+5*stdTotEvents))
    #Print out the x,y of the hot pixels:
    totnrhotpixelevents = 0
    print('--------------------')
    for x,y in zip(hotPixels[0],hotPixels[1]):
        eventsatthispixel = len(events[(events['x']-minx == x) & (events['y']-miny == y)])
        print(f"Hotpixel at: {x+minx},{y+miny} | has {eventsatthispixel} events")
        totnrhotpixelevents+=eventsatthispixel
    
    print(f"Total number of events at hot pixels: {totnrhotpixelevents}")
    print(f"Fraction of events that are in hot pixels: {totnrhotpixelevents/len(events)*100}%")
    
    events
        

def detectHotPixels(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

    from Utils import utils

    window = utils.SmallWindow(parent,windowTitle="Detect Hot Pixels")
    window.addDescription("Detect likely hot pixels")
    loadfileloc = window.addFileLocation()
    
    maxTimeText = window.addTextEdit(labelText="Maximum time (in ms):",preFilledText="5000")
    
    
    outputText = window.addTextEdit(labelText="Output:",preFilledText="Nothing yet, run first")
    
    button = window.addButton("Run")
    button.clicked.connect(lambda: detectHotPixels_run(loadfileloc.text(),maxTimeText.text(),outputText))
    
    window.show()
    pass
