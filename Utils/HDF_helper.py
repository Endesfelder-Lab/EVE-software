#Quick util that cuts a hdf5 based on specific x/y values
import h5py
import numpy as np


# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "CutHDF_xy": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "",
            "display_name": "Cut HDF5 in XY"
        },
        "CutHDF_time": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "",
            "display_name": "Cut HDF5 in time"
        }
    }

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog

#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator, QColor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider, QSpacerItem, QTableView, QFrame, QScrollArea, QProgressBar, QMenu, QMenuBar, QColorDialog
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile, QThread, pyqtSignal, QObject
import sys
import typing

def CutHDF_xy_run(loadfile,savefile,xyStretch):
    xyStretch=eval(xyStretch)
    dataLocation = loadfile
    #Get the time slice from a hdf5 file for index, after running findIndexFromTimeSliceHDF
    print('Starting to read')
    with h5py.File(dataLocation, mode='r') as file:
        events = file['CD']['events']
        #filter out all events that are not in the xyStretch:
        events = events[(events['x'] >= xyStretch[0]) & (events['x'] <= xyStretch[1]) & (events['y'] >= xyStretch[2]) & (events['y'] <= xyStretch[3])]
    
    print('Starting to save')
    #Store these events as a new hdf5:
    with h5py.File(savefile, mode='w') as file:
        events = file.create_dataset('CD/events', data=events, compression="gzip")
    
    print('Saved')
        
def CutHDF_time_run(loadfile,savefile,timeStretch):
    timeStretch=eval(timeStretch)
    #ms to us
    timeStretch=[timeStretch[0]*1000,timeStretch[1]*1000]
    dataLocation = loadfile
    #Get the time slice from a hdf5 file for index, after running findIndexFromTimeSliceHDF
    print('Starting to read')
    with h5py.File(dataLocation, mode='r') as file:
        events = file['CD']['events']
        #filter out all events that are not in the timeStretch:
        events = events[(events['t'] >= timeStretch[0]) & (events['t'] <= timeStretch[1])]
    
    print('Starting to save')
    #Store these events as a new hdf5:
    with h5py.File(savefile, mode='w') as file:
        events = file.create_dataset('CD/events', data=events, compression="gzip")
    
    print('Saved')
        

def CutHDF_xy(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

    try:
        from eve_smlm.Utils import utils
    except ImportError:
        from Utils import utils

    window = utils.SmallWindow(parent,windowTitle="Cut HDF5 file in xy")
    window.addDescription("This function allows you to cut an hdf5 file between certain x,y coordinates. Please find the file location, specify the save location, and specify the XY boundaries (e.g. '(100,200,200,250)' to cut 100-200 in x, 200-250 in y)")
    loadfileloc = window.addFileLocation()
    savefileloc = window.addFileLocation(labelText="Save location:", textAddPrePeriod = "_xyCut")
    
    xyStretchText = window.addTextEdit(labelText="XY boundaries:",preFilledText="(0,np.inf,0,np.inf)")
    
    button = window.addButton("Run")
    button.clicked.connect(lambda: CutHDF_xy_run(loadfileloc.text(),savefileloc.text(),xyStretchText.text()))
    
    window.show()
    pass

def CutHDF_time(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

    try:
        from eve_smlm.Utils import utils
    except ImportError:
        from Utils import utils

    window = utils.SmallWindow(parent,windowTitle="Cut HDF5 file in time")
    window.addDescription("This function allows you to cut an hdf5 file between certain time coordinates. Please find the file location, specify the save location, and specify the time boundaries (e.g. '(0,10000)' to cut 0-10000 ms)")
    loadfileloc = window.addFileLocation()
    savefileloc = window.addFileLocation(labelText="Save location:", textAddPrePeriod = "_timeCut")
    
    timeStretchText = window.addTextEdit(labelText="Time boundaries:",preFilledText="(0,10000)")
    
    button = window.addButton("Run")
    button.clicked.connect(lambda: CutHDF_time_run(loadfileloc.text(),savefileloc.text(),timeStretchText.text()))
    
    window.show()
    pass
