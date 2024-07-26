
#Routine to check for hot pixels in the .raw dataset
import h5py
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import logging

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

def detectHotPixels_run(loadfile,maxTime,stdevString,outputArea,buttonTransfer=None):
    global hotPixels
    from metavision_core.event_io.raw_reader import RawReader
    try:
        float(stdevString)
    except ValueError:
        logging.warning("Stdev should be a float!")
        return
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
    for x in tqdm(range(0, maxx-minx)):
        pixelRow = events[(x_indices == x)]
        for y in range(0, maxy-miny):
            y_indices = pixelRow['y'] - miny
            pixelCol = pixelRow[(y_indices == y)]
            totalEvents[x, y] = len(pixelCol)

    meanTotEvents = np.mean(totalEvents)
    stdTotEvents = np.std(totalEvents)
    #Find all indexes where the total events is superhigh
    hotPixels = np.where(totalEvents > (meanTotEvents+float(stdevString)*stdTotEvents))
    print(hotPixels)
    #Set the hot pixels to be in the output area:
    hotPixels = (hotPixels[0] + minx, hotPixels[1] + miny)
    print(hotPixels)

    #Print out the x,y of the hot pixels:
    totnrhotpixelevents = 0
    outputTextv = ''
    logging.info('--------- Hotpixel utility -----------')
    for x,y in zip(hotPixels[0],hotPixels[1]):
        eventsatthispixel = len(events[(events['x'] == x) & (events['y'] == y)])
        logging.info(f"Hotpixel at: {x},{y} | has {eventsatthispixel} events")
        totnrhotpixelevents+=eventsatthispixel
    
    logging.info(f"Total number of events at hot pixels: {totnrhotpixelevents}")
    logging.info(f"Fraction of events that are in hot pixels: {totnrhotpixelevents/len(events)*100}%")
    
    outputTextv += f"Total events at hot pixels: {totnrhotpixelevents}\n"
    outputTextv += f"% of events in hot pixels: {totnrhotpixelevents/len(events)*100:.1f}%\n"
    for x,y in zip(hotPixels[0],hotPixels[1]):
        eventsatthispixel = len(events[(events['x'] == x) & (events['y'] == y)])
        outputTextv += f"Hotpixel: {x},{y} | {eventsatthispixel} ev ({(eventsatthispixel-meanTotEvents)/stdTotEvents:.2g}x std)\n"
        
    outputArea.setText(outputTextv)
    if buttonTransfer is not None:
        buttonTransfer.setEnabled(True)

def transferOutputToAdvSettings_prewarn(parent):
    """
    Determines the hotpixel fulltext to be put in the globalsettings and double-checks if the user wants to overwrite.
    """
    print('prewarn run')
    try:
        from eve_smlm.Utils import utils
    except ImportError:
        from Utils import utils
        
    global hotPixels
    if hotPixels is not None:
        if len(hotPixels) > 0:
            if len(hotPixels[0]) > 0:
                n_hotpixel = len(hotPixels[0])
                fulltext = ''
                for hotpixel in range(n_hotpixel):
                    fulltext += f'({hotPixels[0][hotpixel]},{hotPixels[1][hotpixel]})'
                    if hotpixel is not (n_hotpixel-1):
                        fulltext += ','
    else:
        fulltext = '()'
    if parent.globalSettings['HotPixelIndexes']['value'] != '':
        #Create a warning for the user.
        warningwindow = utils.SmallWindow(parent,windowTitle="Hot Pixels already set")
        warningwindow.addDescription("Hot Pixels already set! Currently set to: \n"+parent.globalSettings['HotPixelIndexes']['value']+".\n\nAre you sure you want to overwrite it to be \n"+fulltext+"\n It will also store the advanced settings.")
        buttonOK = warningwindow.addButton("OK")
        buttonOK.clicked.connect(lambda: transferOutputToAdvSettings(parent,fulltext,warningwindow))
        buttonCancel = warningwindow.addButton("Cancel")
        buttonCancel.clicked.connect(lambda: warningwindow.close())
        warningwindow.show()
    else:
        transferOutputToAdvSettings(parent,fulltext)
    
def transferOutputToAdvSettings(parent,fulltext,warningwindow=None):
    """
    Actually sets the hotpixelindex to whatever is in fulltext.
    """
    #Set the hotpixel value
    parent.globalSettings['HotPixelIndexes']['value'] = fulltext
    #Store the advanced settings
    parent.advancedSettingsWindow.save_global_settings()
    if warningwindow is not None:
        warningwindow.close()

def detectHotPixels(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

    try:
        from eve_smlm.Utils import utils
    except ImportError:
        from Utils import utils

    global hotPixels
    hotPixels = None

    window = utils.SmallWindow(parent,windowTitle="Detect Hot Pixels")
    window.addDescription("Detect likely hot pixels")
    loadfileloc = window.addFileLocation()
    
    maxTimeText = window.addTextEdit(labelText="Maximum time (in ms):",preFilledText="500000")
    stdevMultText = window.addTextEdit(labelText="Stdev multiplier:",preFilledText="5")
    
    outputText = window.addMultiLineTextEdit(labelText="Output:",preFilledText="Nothing yet, Run first")
    # outputText.setEnabled(False)
    outputText.setStyleSheet("background-color: lightgray;")
    outputText.setReadOnly(True)
    
    
    buttonTransfer = window.addButton("Transfer output to Adv settings")
    buttonTransfer.clicked.connect(lambda: transferOutputToAdvSettings_prewarn(parent))
    buttonTransfer.setEnabled(False)
    
    button = window.addButton("Run")
    button.clicked.connect(lambda: detectHotPixels_run(loadfileloc.text(),maxTimeText.text(),stdevMultText.text(),outputText,buttonTransfer))
    
    window.show()
    
    pass
