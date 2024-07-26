#Quick util that cuts a hdf5 based on specific x/y values
import h5py
import numpy as np


# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "Interactive_ROI_selector": {
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

def Interactive_ROI_selector(parent,**kwargs):#(dataLocation,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
    #Create a small pyqt popup window which allows you to load a file:
    """ 
    Idea: show the entire dataset (or a temporal slice from it, based on the GUI parameters), show this in a napari window, and allow the user to draw a rectangle and press OK. This then updates the min/max x/y in run/preview.
    
    """

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
    window.show()
    pass
