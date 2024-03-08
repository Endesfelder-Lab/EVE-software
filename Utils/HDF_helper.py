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
            "help_string": ""
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


class SmallWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Small Window")
        self.resize(300, 200)

        self.parent = parent
        # Set the window icon to the parent's icon
        self.setWindowIcon(self.parent.windowIcon())

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
        

def CutHDF_xy(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:

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
