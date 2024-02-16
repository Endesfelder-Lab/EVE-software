#General imports
import sys, os, logging, json, argparse, datetime, glob, csv, ast, platform, threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import copy
import appdirs
import pickle
import time
from textwrap import dedent
import h5py
import traceback
import re

#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider, QSpacerItem, QTableView, QFrame, QScrollArea
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile, QThread, pyqtSignal, QObject
import sys
import typing

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QWidget, QGridLayout, QPushButton


from napari import Viewer
from napari.qt import QtViewer
from napari.layers import Image, Shapes
from napari.utils.events import Event
from vispy.color import Colormap

#Custom imports
# Add the folder 2 folders up to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#Import all scripts in the custom script folders
from CandidateFitting import *
from CandidateFinding import *
from Visualisation import *
from PostProcessing import *
from CandidatePreview import *

#Obtain the helperfunctions
from Utils import utils

#Obtain eventdistribution functions
from EventDistributions import eventDistributions

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
class ProcessingThread(QThread):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.FindingEvalText = None
        self.FittingEvalText = None
        self.runFitting = None
        self.storeFinding = None
        self.polarityVal = None
        self.npyData = None
        self.GUIinfo = None
        self.typeOfThread = None

    def run(self):
        # Perform your processing here
        if self.typeOfThread == "Finding":
            result = eval(self.FindingEvalText)
        self.finished.emit()

    def setTypeOfThread(self,typeOfThread):
        self.typeOfThread = typeOfThread
    def setFindingEvalText(self,FindingEvalText):
        self.FindingEvalText = FindingEvalText
    def setFittingEvalText(self,FittingEvalText):
        self.FittingEvalText = FittingEvalText
    def setrunFitting(self,runFitting):
        self.runFitting = runFitting
    def setstoreFinding(self,storeFinding):
        self.storeFinding = storeFinding
    def setpolarityVal(self,polarityVal):
        self.polarityVal = polarityVal
    def setnpyData(self,npyData):
        self.npyData = npyData
    def setGUIinfo(self,GUIinfo):
        self.GUIinfo = GUIinfo
        self.data = self.GUIinfo.data
        self.globalSettings = self.GUIinfo.globalSettings
        self.currentFileInfo = self.GUIinfo.currentFileInfo
        self.logger = self.GUIinfo.logger

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()

    def do_work(self):
        for i in range(1, 11):
            time.sleep(1)  # Simulate time-consuming task
            self.progress.emit(i * 10)  # Emit progress (10%, 20%, ..., 100%)
        self.finished.emit()  # Emit finished signal when the task is done

class MyGUI(QMainWindow):
#GUI class - needs to be initialised and is initalised in gui.py
    def __init__(self):
        """
        Initializes the GUI.
        Sets up the whole GUI and allt abs

        Args:
            None

        Returns:
            None
        """

        #Create parser
        parser = argparse.ArgumentParser(description='EBS fitting - Endesfelder lab - Nov 2023')
        #Look for debug argument
        parser.add_argument('--debug', '-d', action='store_true', help='Enable debug')
        args=parser.parse_args()

        """
        Dictionary creation used throughout GUI
        """
        #Create a dictionary to store the global Settings
        self.globalSettings = self.initGlobalSettings()

        # Create a dictionary to store the entries
        self.entries = {}

        #Create a dictionary that stores info about each file being run (i.e. for metadata)
        self.currentFileInfo = {}

        #Create a dictionary that stores data and passes it between finding,fitting,saving, etc
        self.data = {}
        self.data['refFindingFile'] = ''
        self.data['refFindingFilepos'] = ''
        self.data['refFindingFileneg'] = ''
        self.data['prevFindingMethod'] = 'None'
        self.data['prevNrEvents'] = 0
        self.data['prevNrCandidates'] = 0
        #intialise preview events
        self.previewEvents = []

        """
        Logger creation, both text output and stored to .log file
        """
        # Create a logger with the desired log level
        self.log_file_path = self.globalSettings['LoggingFilePath']['value']
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

        # Create the file handler to log to the file
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)

        # Create the stream handler to log to the debug terminal
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)

        # Add the handlers to the logger
        logging.basicConfig(handlers=[file_handler, stream_handler], level=logging.DEBUG if args.debug else logging.INFO,format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")

        if os.path.exists(self.log_file_path):
            open(self.log_file_path, 'w').close()

        """
        GUI initalisation
        Generally, the GUI consists of QGroupBoxes containing QGridLayouts which contain the input fields and buttons and such
        """
        #Initialisation - title and size
        super().__init__()
        self.setWindowTitle("Eve - alphaVersion")
        self.setMinimumSize(600, 1000)  # Set minimum size for the GUI window

        #Set the central widget that contains everything (self.central_widget)
        #This is a group box that contains a grid layout. We fill everything inside this grid layout
        self.central_widget = QGroupBox()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.polarityDropdownNames = ['All events treated equal','Positive only','Negative only','Pos and neg seperately']
        self.polarityDropdownOrder = [0,1,2,3]

        """
        Global settings group box
        """
        #Create a global settings group box
        self.globalSettingsGroupBox = QGroupBox("Global settings")
        self.globalSettingsGroupBox.setLayout(QGridLayout()) #Give it a grid layout as well

        #Create an advanced settings button that opens a new window
        self.advancedSettingsButton = QPushButton("Advanced settings", self)
        self.advancedSettingsButton.clicked.connect(self.open_advanced_settings)
        self.globalSettingsGroupBox.layout().addWidget(self.advancedSettingsButton, 0, 0,1,2)

        #Initialise (but not show!) advanced window:
        self.advancedSettingsWindow = AdvancedSettingsWindow(self)

        # Create a button to trigger saving the entries
        self.save_button = QPushButton("Save GUI contents", self)
        self.save_button.clicked.connect(self.save_entries_to_json)
        self.globalSettingsGroupBox.layout().addWidget(self.save_button, 0, 3)
        # Create a button to trigger loading the entries
        self.load_button = QPushButton("Load GUI contents", self)
        self.load_button.clicked.connect(self.load_entries_from_json)
        self.globalSettingsGroupBox.layout().addWidget(self.load_button, 0,4)

        #Add the global settings group box to the central widget
        self.layout.addWidget(self.globalSettingsGroupBox, 0, 0)

        """
        Main tab widget (containing processing, post-processing etc tabs)
        """
        #Create a tab widget and add this to the main group box
        self.mainTabWidget = QTabWidget()
        self.mainTabWidget.setTabPosition(QTabWidget.South)
        self.layout.addWidget(self.mainTabWidget, 2, 0)

        #Add the individual tabs
        self.tab_processing = QWidget()
        self.mainTabWidget.addTab(self.tab_processing, "Processing")
        self.tab_postProcessing = QWidget()
        self.mainTabWidget.addTab(self.tab_postProcessing, "Post-processing")
        self.tab_locList = QWidget()
        self.mainTabWidget.addTab(self.tab_locList, "LocalizationList")
        self.tab_visualisation = QWidget()
        self.mainTabWidget.addTab(self.tab_visualisation, "Visualisation")
        self.tab_logfileInfo = QWidget()
        self.mainTabWidget.addTab(self.tab_logfileInfo, "Run info")
        self.tab_previewVis = QWidget()
        self.mainTabWidget.addTab(self.tab_previewVis, "Preview run")
        self.tab_canPreview = QWidget()
        self.mainTabWidget.addTab(self.tab_canPreview, "Candidate preview")

        #Set up the tabs
        self.setup_tab('Processing')
        self.setup_tab('Post-processing')
        self.setup_tab('LocalizationList')
        self.setup_tab('Visualisation')
        self.setup_tab('Run info')
        self.setup_tab('Preview visualisation')
        self.setup_tab('Candidate preview')


        #Loop through all combobox states briefly to initialise them (and hide them)
        self.set_all_combobox_states()

        #Load the GUI settings from last time:
        self.load_entries_from_json()

        #Initialise polarity value thing (only affects very first run I believe):
        self.polarityDropdownChanged()

        #Set up worker information:
        # self.worker = Worker()
        # self.worker_thread = QThread()
        # self.worker.moveToThread(self.worker_thread)
        # # Connect the started signal to the do_work method of the worker
        # self.worker_thread.started.connect(lambda: self.worker.do_work())
        # self.worker.progress.connect(self.QThreadEmitCatcher)
        # self.worker.finished.connect(self.worker_thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        #Initialise empty dictionaries for storage
        self.data['FindingResult'] = {}
        self.data['FindingResult'][0] = {}
        self.data['FindingResult'][1] = []
        self.data['FittingResult'] = {}
        self.data['FittingResult'][0] = {}
        self.data['FittingResult'][1] = []


        self.thread = ProcessingThread()
        self.thread.finished.connect(self.thread.quit)
        # self.thread.progress.connect(self.QThreadEmitCatcher)


        logging.info('Initialisation complete.')

    def QThreadEmitCatcher(self,text):
        #Catches all output from the worker thread
        print(text)
        logging.info(text)

    def setup_tab(self, tab_name):
    #Generic function to set up tabs - basically a look-up to other functions
        tab_mapping = {
            'Processing': self.setup_processingTab,
            'Post-processing': self.setup_postProcessingTab,
            'Visualisation': self.setup_visualisationTab,
            'LocalizationList': self.setup_loclistTab,
            'Run info': self.setup_logFileTab,
            'Preview visualisation': self.setup_previewTab,
            'Candidate preview': self.setup_canPreviewTab
        }
        #Run the setup of this tab
        setup_func = tab_mapping.get(tab_name)
        if setup_func:
            setup_func()

    def open_advanced_settings(self):
        #Function that opens the advanced settings window
        self.advancedSettingsWindow.show()

    def open_critical_warning(self, text):
        #Function that creates and opens the critical warning window
        self.criticalWarningWindow = CriticalWarningWindow(self, text)
        self.criticalWarningWindow.show()

    def initGlobalSettings(self):
        #Initialisation of the global settings - runs on startup to get all these values, then these can be changed later
        globalSettings = {}
        globalSettings['MinFindingBoundingBoxXY'] = {}
        globalSettings['MinFindingBoundingBoxXY']['value'] = 3
        globalSettings['MinFindingBoundingBoxXY']['input'] = float
        globalSettings['MinFindingBoundingBoxXY']['displayName'] = 'Minimum size of a bounding box in px units'
        globalSettings['MinFindingBoundingBoxT'] = {}
        globalSettings['MinFindingBoundingBoxT']['value'] = 10
        globalSettings['MinFindingBoundingBoxT']['input'] = float
        globalSettings['MinFindingBoundingBoxT']['displayName'] = 'Minimum size of a bounding box in ms units'
        globalSettings['MaxFindingBoundingBoxXY'] = {}
        globalSettings['MaxFindingBoundingBoxXY']['value'] = 20
        globalSettings['MaxFindingBoundingBoxXY']['input'] = float
        globalSettings['MaxFindingBoundingBoxXY']['displayName'] = 'Maximum size of a bounding box in px units'
        globalSettings['MaxFindingBoundingBoxT'] = {}
        globalSettings['MaxFindingBoundingBoxT']['value'] = 1000
        globalSettings['MaxFindingBoundingBoxT']['input'] = float
        globalSettings['MaxFindingBoundingBoxT']['displayName'] = 'Maximum size of a bounding box in ms units'
        globalSettings['PixelSize_nm'] = {}
        globalSettings['PixelSize_nm']['value'] = 80
        globalSettings['PixelSize_nm']['input'] = float
        globalSettings['PixelSize_nm']['displayName'] = 'Pixel size (nm)'
        globalSettings['MetaVisionPath'] = {}
        if platform.system() == 'Windows':
            globalSettings['MetaVisionPath']['value'] = "C:\Program Files\Prophesee\lib\python3\site-packages"
        elif platform.system() == 'Linux':
            globalSettings['MetaVisionPath']['value'] = "/usr/lib/python3/dist-packages"
        elif platform.system() == 'Darwin':
            globalSettings['MetaVisionPath']['value'] = "Enter Path to Metavision SDK!"
        else:
            globalSettings['MetaVisionPath']['value'] = "Enter Path to Metavision SDK!"
        globalSettings['MetaVisionPath']['input'] = str
        globalSettings['MetaVisionPath']['displayName'] = 'MetaVision SDK Path'
        globalSettings['StoreConvertedRawData'] = {}
        globalSettings['StoreConvertedRawData']['value'] = True
        globalSettings['StoreConvertedRawData']['input'] = bool
        globalSettings['StoreConvertedRawData']['displayName'] = 'Store converted raw data'
        globalSettings['StoreFileMetadata'] = {}
        globalSettings['StoreFileMetadata']['value'] = True
        globalSettings['StoreFileMetadata']['input'] = bool
        globalSettings['StoreFileMetadata']['displayName'] = 'Store metadata after running'
        globalSettings['StoreFinalOutput'] = {}
        globalSettings['StoreFinalOutput']['value'] = True
        globalSettings['StoreFinalOutput']['input'] = bool
        globalSettings['StoreFinalOutput']['displayName'] = 'Store the final output'
        globalSettings['StoreFindingOutput'] = {}
        globalSettings['StoreFindingOutput']['value'] = True
        globalSettings['StoreFindingOutput']['input'] = bool
        globalSettings['StoreFindingOutput']['displayName'] = 'Store intermediate output (after finding)'
        globalSettings['StoreFittingOutput'] = {}
        globalSettings['StoreFittingOutput']['value'] = True
        globalSettings['StoreFittingOutput']['input'] = bool
        globalSettings['StoreFittingOutput']['displayName'] = 'Store intermediate output (after fitting)'
        globalSettings['OutputDataFormat'] = {}
        globalSettings['OutputDataFormat']['value'] = 'thunderstorm'
        globalSettings['OutputDataFormat']['input'] = 'choice'
        globalSettings['OutputDataFormat']['options'] = ('thunderstorm','minimal')
        globalSettings['OutputDataFormat']['displayName'] = 'Output data format'
        user_data_folder = appdirs.user_data_dir(appname="Eve", appauthor="UniBonn")
        #Check if this folder already exists, if not, create it:
        if not os.path.exists(user_data_folder):
            os.makedirs(user_data_folder)
        globalSettings['JSONGUIstorePath'] = {}
        globalSettings['JSONGUIstorePath']['value'] = user_data_folder+os.sep+"storage.json"
        globalSettings['JSONGUIstorePath']['input'] = str
        globalSettings['GlobalOptionsStorePath'] = {}
        globalSettings['GlobalOptionsStorePath']['value'] = user_data_folder+os.sep+ "GlobSettingStorage.json"
        globalSettings['GlobalOptionsStorePath']['input'] = str
        globalSettings['LoggingFilePath'] = {}
        globalSettings['LoggingFilePath']['value'] = user_data_folder+os.sep+"logging.log"
        globalSettings['LoggingFilePath']['input'] = str

        # Multi-threading
        globalSettings['Multithread'] = {}
        globalSettings['Multithread']['value'] = True
        globalSettings['Multithread']['input'] = bool
        #Finding batching info
        globalSettings['FindingBatching'] = {}
        globalSettings['FindingBatching']['value'] = True
        globalSettings['FindingBatching']['input'] = bool
        globalSettings['FindingBatchingTimeMs'] = {}
        globalSettings['FindingBatchingTimeMs']['value'] = 50000
        globalSettings['FindingBatchingTimeMs']['input'] = float
        globalSettings['FindingBatchingTimeOverlapMs'] = {}
        globalSettings['FindingBatchingTimeOverlapMs']['value'] = 500
        globalSettings['FindingBatchingTimeOverlapMs']['input'] = float

        #Add options here that should NOT show up in the global settings window - i.e. options that should not be changed
        globalSettings['IgnoreInOptions'] = ('IgnoreInOptions','StoreFinalOutput', 'JSONGUIstorePath','GlobalOptionsStorePath')
        return globalSettings

    def datasetSearchButtonClicked(self):
        # Function that handles the dataset 'File' lookup button
        logging.debug('data lookup search button clicked')
        file_path = utils.generalFileSearchButtonAction(parent=self,text="Select File",filter="EBS files (*.raw *.npy *hdf5);;All Files (*)")
        if file_path:
            self.dataLocationInput.setText(file_path)

    def datasetFolderButtonClicked(self):
        # Function that handles the dataset 'Folder' lookup
        logging.debug('data lookup Folder button clicked')
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.dataLocationInput.setText(folder_path)

    def setup_processingTab(self):
        """
        Sets up the processing tab by creating the necessary layouts and adding the required widgets.

        Parameters:
            None

        Returns:
            None
        """
        #Create a grid layout and set it
        tab_layout = QGridLayout()
        self.tab_processing.setLayout(tab_layout)
        """
        Dataset location and searching Grid Layout
        """
        self.datasetLocation_layout = QGridLayout()
        tab_layout.addLayout(self.datasetLocation_layout, 0, 0)

        # Add a label:
        self.datasetLocation_label = QLabel("Dataset location:")
        self.datasetLocation_layout.addWidget(self.datasetLocation_label, 0, 0,2,1)
        # Create the input field
        self.dataLocationInput = QLineEdit()
        self.dataLocationInput.setObjectName("processing_dataLocationInput")
        self.datasetLocation_layout.layout().addWidget(self.dataLocationInput, 0, 1,2,1)
        # Create the search buttons
        self.datasetSearchButton = QPushButton("File...")
        self.datasetSearchButton.clicked.connect(self.datasetSearchButtonClicked)
        self.datasetLocation_layout.layout().addWidget(self.datasetSearchButton, 0, 2)
        self.datasetFolderButton = QPushButton("Folder...")
        self.datasetFolderButton.clicked.connect(self.datasetFolderButtonClicked)
        self.datasetLocation_layout.layout().addWidget(self.datasetFolderButton, 1, 2)

        """
        Data selection Grid Layout
        """
        #Add a data selection tab
        self.dataSelectionLayout = ClickableGroupBox("Data selection")
        self.dataSelectionLayout.setCheckable(True)
        self.dataSelectionLayout.setObjectName("groupboxRun")
        self.dataSelectionLayout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dataSelectionLayout.setLayout(QGridLayout())
        tab_layout.addWidget(self.dataSelectionLayout, 1, 0)

        #Add smaller GridLayouts for Polarity, Time, Position
        self.dataSelectionPolarityLayout = QGroupBox("Polarity")
        self.dataSelectionPolarityLayout.setLayout(QGridLayout())
        self.dataSelectionTimeLayout = QGroupBox("Time")
        self.dataSelectionTimeLayout.setLayout(QGridLayout())
        self.dataSelectionPositionLayout = QGroupBox("Position")
        self.dataSelectionPositionLayout.setLayout(QGridLayout())

        #Add them
        self.dataSelectionLayout.layout().addWidget(self.dataSelectionPolarityLayout,0,0)
        self.dataSelectionLayout.layout().addWidget(self.dataSelectionTimeLayout,0,1)
        self.dataSelectionLayout.layout().addWidget(self.dataSelectionPositionLayout,0,2)

        #Populate Polarity layout with a dropdown:
        self.dataSelectionPolarityLayout.layout().addWidget(QLabel("Polarity:"), 0,0)
        self.dataSelectionPolarityDropdown = QComboBox()
        self.dataSelectionPolarityDropdown.addItem(self.polarityDropdownNames[self.polarityDropdownOrder[0]])
        self.dataSelectionPolarityDropdown.addItem(self.polarityDropdownNames[self.polarityDropdownOrder[1]])
        self.dataSelectionPolarityDropdown.addItem(self.polarityDropdownNames[self.polarityDropdownOrder[2]])
        self.dataSelectionPolarityDropdown.addItem(self.polarityDropdownNames[self.polarityDropdownOrder[3]])
        self.dataSelectionPolarityLayout.layout().addWidget(self.dataSelectionPolarityDropdown, 1,0)
        #Add a lambda function to this dropdown:
        self.dataSelectionPolarityDropdown.currentIndexChanged.connect(lambda: self.polarityDropdownChanged())
        self.dataSelectionPolarityDropdown.setObjectName('selectPolarityOptions_dropdown')
        #Add one of those addStretch to push it all to the top:
        self.dataSelectionPolarityLayout.layout().addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 2,0)


        #Populate with a start time and end time inputs:
        self.dataSelectionTimeLayout.layout().addWidget(QLabel("Start time (ms):"), 0,0)
        self.dataSelectionTimeLayout.layout().addWidget(QLabel("Duration (ms):"), 2,0)
        #Also the QLineEdits that have useful names:
        self.run_startTLineEdit = QLineEdit()
        self.run_startTLineEdit.setObjectName('run_startTLineEdit')
        self.dataSelectionTimeLayout.layout().addWidget(self.run_startTLineEdit, 1,0)
        #Give this a default value:
        self.run_startTLineEdit.setText("0")
        #Same for end time:
        self.run_durationTLineEdit = QLineEdit()
        self.run_durationTLineEdit.setObjectName('run_durationTLineEdit')
        self.dataSelectionTimeLayout.layout().addWidget(self.run_durationTLineEdit, 3,0)
        self.run_durationTLineEdit.setText("Inf")

        #Also give start/end x/y values:
        self.dataSelectionPositionLayout.layout().addWidget(QLabel("Min X (px):"), 0, 0)
        self.dataSelectionPositionLayout.layout().addWidget(QLabel("Max X (px):"), 0, 1)
        self.dataSelectionPositionLayout.layout().addWidget(QLabel("Min Y (px):"), 2, 0)
        self.dataSelectionPositionLayout.layout().addWidget(QLabel("Max Y (px):"), 2, 1)
        #Also the QLineEdits that have useful names:
        self.run_minXLineEdit = QLineEdit()
        self.run_minXLineEdit.setObjectName('run_minXLineEdit')
        self.dataSelectionPositionLayout.layout().addWidget(self.run_minXLineEdit, 1, 0)
        self.run_minXLineEdit.setText("0")
        self.run_maxXLineEdit = QLineEdit()
        self.run_maxXLineEdit.setObjectName('run_maxXLineEdit')
        self.dataSelectionPositionLayout.layout().addWidget(self.run_maxXLineEdit, 1, 1)
        self.run_maxXLineEdit.setText("Inf")
        self.run_minYLineEdit = QLineEdit()
        self.run_minYLineEdit.setObjectName('run_minYLineEdit')
        self.dataSelectionPositionLayout.layout().addWidget(self.run_minYLineEdit, 3, 0)
        self.run_minYLineEdit.setText("0")
        self.run_maxYLineEdit = QLineEdit()
        self.run_maxYLineEdit.setObjectName('run_maxYLineEdit')
        self.dataSelectionPositionLayout.layout().addWidget(self.run_maxYLineEdit, 3, 1)
        self.run_maxYLineEdit.setText("Inf")

        """
        Candidate Finding Grid Layout
        """

        self.mainCandidateFindingGroupbox = QGroupBox("Candidate Finding")
        self.mainCandidateFindingGroupbox.setLayout(QGridLayout())
        tab_layout.addWidget(self.mainCandidateFindingGroupbox, 2, 0)

        names = ['Pos', 'Neg', 'Mix']
        ii = 0
        import functools
        for name in names:
            groupbox_name = f"groupboxFinding{name}"
            candidateFindingDropdown_name = f"CandidateFindingDropdown{name}"
            layout_name = f"layoutFinding{name}"
            Finding_functionNameToDisplayNameMapping_name = f"Finding_functionNameToDisplayNameMapping{name}"

            groupbox = QGroupBox(f"Candidate finding {name}")
            groupbox.setObjectName(groupbox_name)
            groupbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            groupbox.setLayout(QGridLayout())

            # Create a QComboBox and add options - this is the FINDING dropdown
            candidateFindingDropdown = QComboBox(self)
            #Increase the maximum visible items since we're always hiding 2/3rds of it (pos/neg/mix)
            candidateFindingDropdown.setMaxVisibleItems(30)
            options = utils.functionNamesFromDir('CandidateFinding')
            displaynames, Finding_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options,name.lower())
            candidateFindingDropdown.setObjectName(candidateFindingDropdown_name)
            candidateFindingDropdown.addItems(displaynames)
            #Add the candidateFindingDropdown to the layout
            groupbox.layout().addWidget(candidateFindingDropdown,1,0,1,6)

            setattr(self, Finding_functionNameToDisplayNameMapping_name, Finding_functionNameToDisplayNameMapping)

            #On startup/initiatlisation: also do changeLayout_choice
            utils.changeLayout_choice(groupbox.layout(),candidateFindingDropdown_name,getattr(self, Finding_functionNameToDisplayNameMapping_name),parent=self)

            setattr(self, groupbox_name, groupbox)
            setattr(self, candidateFindingDropdown_name, candidateFindingDropdown)
            setattr(self, layout_name, functools.partial(self.mainCandidateFindingGroupbox.layout().addWidget, groupbox, 0,ii))
            getattr(self, layout_name)()
            ii+=1

        #Can't get this part to work dynamically so eh
        #Activation for candidateFindingDropdown.activated
        self.CandidateFindingDropdownPos.activated.connect(lambda: utils.changeLayout_choice(self.groupboxFindingPos.layout(),'CandidateFindingDropdownPos', self.Finding_functionNameToDisplayNameMappingPos,parent=self))
        self.CandidateFindingDropdownNeg.activated.connect(lambda: utils.changeLayout_choice(self.groupboxFindingNeg.layout(),'CandidateFindingDropdownNeg', self.Finding_functionNameToDisplayNameMappingNeg,parent=self))
        self.CandidateFindingDropdownMix.activated.connect(lambda: utils.changeLayout_choice(self.groupboxFindingMix.layout(),'CandidateFindingDropdownMix', self.Finding_functionNameToDisplayNameMappingMix,parent=self))

        """
        Candidate Fitting Grid Layout
        """
        self.mainCandidateFittingGroupbox = QGroupBox("Candidate Fitting")
        self.mainCandidateFittingGroupbox.setLayout(QGridLayout())
        tab_layout.addWidget(self.mainCandidateFittingGroupbox, 3, 0)

        names = ['Pos', 'Neg', 'Mix']
        ii = 0
        import functools
        for name in names:
            groupbox_name = f"groupboxFitting{name}"
            candidateFittingDropdown_name = f"CandidateFittingDropdown{name}"
            layout_name = f"layoutFitting{name}"
            Fitting_functionNameToDisplayNameMapping_name = f"Fitting_functionNameToDisplayNameMapping{name}"

            groupbox = QGroupBox(f"Candidate fitting {name}")
            groupbox.setObjectName(groupbox_name)
            groupbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            groupbox.setLayout(QGridLayout())

            # Create a QComboBox and add options - this is the FINDING dropdown
            candidateFittingDropdown = QComboBox(self)
            #Increase the maximum visible items since we're always hiding 2/3rds of it (pos/neg/mix)
            candidateFittingDropdown.setMaxVisibleItems(30)
            options = utils.functionNamesFromDir('CandidateFitting')
            displaynames, Fitting_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options,name.lower())
            candidateFittingDropdown.setObjectName(candidateFittingDropdown_name)
            candidateFittingDropdown.addItems(displaynames)
            #Add the candidateFindingDropdown to the layout
            groupbox.layout().addWidget(candidateFittingDropdown,1,0,1,6)

            setattr(self, Fitting_functionNameToDisplayNameMapping_name, Fitting_functionNameToDisplayNameMapping)

            #On startup/initiatlisation: also do changeLayout_choice
            utils.changeLayout_choice(groupbox.layout(),candidateFittingDropdown_name, getattr(self, Fitting_functionNameToDisplayNameMapping_name),parent=self)

            setattr(self, groupbox_name, groupbox)
            setattr(self, layout_name, functools.partial(self.mainCandidateFittingGroupbox.layout().addWidget, groupbox, 0,ii))
            getattr(self, layout_name)()
            setattr(self, candidateFittingDropdown_name, candidateFittingDropdown)
            ii+=1

        #Can't get this part to work dynamically so eh
        #Activation for candidateFindingDropdown.activated
        self.CandidateFittingDropdownPos.activated.connect(lambda: utils.changeLayout_choice(self.groupboxFittingPos.layout(),'CandidateFittingDropdownPos', self.Fitting_functionNameToDisplayNameMappingPos,parent=self))
        self.CandidateFittingDropdownNeg.activated.connect(lambda: utils.changeLayout_choice(self.groupboxFittingNeg.layout(),'CandidateFittingDropdownNeg', self.Fitting_functionNameToDisplayNameMappingNeg,parent=self))
        self.CandidateFittingDropdownMix.activated.connect(lambda: utils.changeLayout_choice(self.groupboxFittingMix.layout(),'CandidateFittingDropdownMix', self.Fitting_functionNameToDisplayNameMappingMix,parent=self))

        """
        "Run" Group Box
        """
        #Add a run tab:
        self.runLayout = QGroupBox("Run")
        self.runLayout.setObjectName("groupboxRun")
        self.runLayout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.runLayout.setLayout(QGridLayout())

        self.buttonProcessingRun = QPushButton("Run")
        self.buttonProcessingRun.clicked.connect(lambda: self.run_processing())
        self.runLayout.layout().addWidget(self.buttonProcessingRun,2,0,1,6)

        tab_layout.addWidget(self.runLayout, 4, 0)

        """
        Spacing between things above and things below
        """
        #Add spacing so that the previewLayout is pushed to the bottom:
        tab_layout.setRowStretch(5, 1)

        """
        Preview Group Box
        """
        #Add a preview box:
        self.previewLayout = QGroupBox("Preview")
        self.previewLayout.setObjectName("groupboxPreview")
        self.previewLayout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.previewLayout.setLayout(QGridLayout())
        tab_layout.addWidget(self.previewLayout, 9, 0)

        #Populate with a start time and end time inputs:
        self.previewLayout.layout().addWidget(QLabel("Start time (ms):"), 0, 0, 1, 1)
        self.previewLayout.layout().addWidget(QLabel("Duration (ms):"), 0, 1, 1, 1)
        self.previewLayout.layout().addWidget(QLabel("Display frame time (ms):"), 0,2,1,2)
        #Also the QLineEdits that have useful names:
        self.preview_startTLineEdit = QLineEdit()
        self.preview_startTLineEdit.setObjectName('preview_startTLineEdit')
        self.previewLayout.layout().addWidget(self.preview_startTLineEdit, 1, 0, 1, 1)
        #Give this a default value:
        self.preview_startTLineEdit.setText("0")
        #Same for end time:
        self.preview_durationTLineEdit = QLineEdit()
        self.preview_durationTLineEdit.setObjectName('preview_durationTLineEdit')
        self.previewLayout.layout().addWidget(self.preview_durationTLineEdit, 1, 1, 1, 1)
        self.preview_durationTLineEdit.setText("1000")
        #And for display frame time:
        self.preview_displayFrameTime = QLineEdit()
        self.preview_displayFrameTime.setObjectName('preview_displayFrameTime')
        self.previewLayout.layout().addWidget(self.preview_displayFrameTime, 1, 2, 1, 2)
        self.preview_displayFrameTime.setText("100")

        #Also give start/end x/y values:
        self.previewLayout.layout().addWidget(QLabel("Min X (px):"), 2, 0)
        self.previewLayout.layout().addWidget(QLabel("Max X (px):"), 2, 1)
        self.previewLayout.layout().addWidget(QLabel("Min Y (px):"), 2, 2)
        self.previewLayout.layout().addWidget(QLabel("Max Y (px):"), 2, 3)
        #Also the QLineEdits that have useful names:
        self.preview_minXLineEdit = QLineEdit()
        self.preview_minXLineEdit.setObjectName('preview_minXLineEdit')
        self.previewLayout.layout().addWidget(self.preview_minXLineEdit, 3, 0)
        self.preview_minXLineEdit.setText("")
        self.preview_maxXLineEdit = QLineEdit()
        self.preview_maxXLineEdit.setObjectName('preview_maxXLineEdit')
        self.previewLayout.layout().addWidget(self.preview_maxXLineEdit, 3, 1)
        self.preview_maxXLineEdit.setText("")
        self.preview_minYLineEdit = QLineEdit()
        self.preview_minYLineEdit.setObjectName('preview_minYLineEdit')
        self.previewLayout.layout().addWidget(self.preview_minYLineEdit, 3, 2)
        self.preview_minYLineEdit.setText("")
        self.preview_maxYLineEdit = QLineEdit()
        self.preview_maxYLineEdit.setObjectName('preview_maxYLineEdit')
        self.previewLayout.layout().addWidget(self.preview_maxYLineEdit, 3, 3)
        self.preview_maxYLineEdit.setText("")

        #Add a preview button:
        self.buttonPreview = QPushButton("Preview")
        #Add a button press event:
        self.buttonPreview.clicked.connect(lambda: self.previewRun((self.preview_startTLineEdit.text(),
                self.preview_durationTLineEdit.text()),
                (self.preview_minXLineEdit.text(),self.preview_maxXLineEdit.text(),
                self.preview_minYLineEdit.text(),self.preview_maxYLineEdit.text()),
                float(self.preview_displayFrameTime.text())))
        #Add the button to the layout:
        self.previewLayout.layout().addWidget(self.buttonPreview, 4, 0)

    def polarityDropdownChanged(self):
        # """
        # Lambda function that's called when the polarity dropdown is changed
        # """
        # oldPolarity = utils.polaritySelectedFromDisplayName(self.candidateFindingDropdown.currentText())

        # #Finding changing...
        # currentlySelectedFinding = self.candidateFindingDropdown.currentText()
        # newSelectedFinding = currentlySelectedFinding.replace('('+oldPolarity+')','('+newPolarity+')')
        # self.candidateFindingDropdown.setCurrentText(newSelectedFinding)
        # #Fitting changing...
        # currentlySelectedFitting = self.candidateFittingDropdown.currentText()
        # newSelectedFitting = currentlySelectedFitting.replace('('+oldPolarity+')','('+newPolarity+')')
        # self.candidateFittingDropdown.setCurrentText(newSelectedFitting)

        # self.changeLayout_choice(self.groupboxFinding.layout(),"CandidateFinding_candidateFindingDropdown",self.Finding_functionNameToDisplayNameMapping)
        # self.changeLayout_choice(self.groupboxFitting.layout(),"CandidateFitting_candidateFittingDropdown",self.Fitting_functionNameToDisplayNameMapping)

        #We get the currently chosen polarity:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[0]:
            newPolarity = 'mix'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[1]:
            newPolarity = 'pos'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[2]:
            newPolarity = 'neg'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            newPolarity = 'posneg'

        #And we show/hide the required groupboxes:
        if newPolarity == 'pos':
            self.groupboxFindingPos.show()
            self.groupboxFindingNeg.hide()
            self.groupboxFindingMix.hide()
            self.groupboxFittingPos.show()
            self.groupboxFittingNeg.hide()
            self.groupboxFittingMix.hide()
        elif newPolarity == 'neg':
            self.groupboxFindingPos.hide()
            self.groupboxFindingNeg.show()
            self.groupboxFindingMix.hide()
            self.groupboxFittingPos.hide()
            self.groupboxFittingNeg.show()
            self.groupboxFittingMix.hide()
        elif newPolarity == 'mix':
            self.groupboxFindingPos.hide()
            self.groupboxFindingNeg.hide()
            self.groupboxFindingMix.show()
            self.groupboxFittingPos.hide()
            self.groupboxFittingNeg.hide()
            self.groupboxFittingMix.show()
        elif newPolarity == 'posneg':
            self.groupboxFindingPos.show()
            self.groupboxFindingNeg.show()
            self.groupboxFindingMix.hide()
            self.groupboxFittingPos.show()
            self.groupboxFittingNeg.show()
            self.groupboxFittingMix.hide()


        return

    def previewRun(self,timeStretch=(0,1000),xyStretch=(0,0,0,0),frameTime=100):
        """
        Generates the preview of a run analysis.

        Parameters:
            timeStretch (tuple): A tuple containing the start and end times for the preview.
            xyStretch (tuple): A tuple containing the minimum and maximum x and y coordinates for the preview.

        Returns:
            None
        """
        #Give a clear logging separation:
        logging.info("")
        logging.info("")
        logging.info("--------------- Preview Run Starting ---------------")
        logging.info("")
        logging.info("")
        
        #Switch the user to the Run info tab
        utils.changeTab(self, text='Run info')

        # Empty the event preview list
        self.previewEvents = []

        #Checking if a file is selected rather than a folder:
        if not os.path.isfile(self.dataLocationInput.text()):
            logging.error("Please choose a file rather than a folder for previews.")
            return

        #Specifically for preview runs, we don't need to load the whole data, only between preview_startT and preview_endT. This is handled differently for RAW or NPY files:
        #Thus, we check if it's raw or npy:
        if self.dataLocationInput.text().endswith('.raw'):
            sys.path.append(self.globalSettings['MetaVisionPath']['value'])
            from metavision_core.event_io.raw_reader import RawReader
            record_raw = RawReader(self.dataLocationInput.text())

            #Seek to the start time if the start time is > 0:
            if int(timeStretch[0]) > 0:
                record_raw.seek_time(int(timeStretch[0])*1000)

            #Load all events
            events=np.empty
            events = record_raw.load_delta_t(int(timeStretch[1])*1000)
            #Check if we have at least 1 event:
            if len(events) > 0:
                # correct the coordinates and time stamps
                # events['x']-=np.min(events['x'])
                # events['y']-=np.min(events['y'])
                # events['t']-=np.min(events['t'])
                #Log the nr of events found:
                logging.info(f"Preview - Found {len(events)} events in the chosen time frame.")
            else:
                logging.error("Preview - No events found in the chosen time frame.")
                return
        #Logic if a numpy file is selected:
        elif self.dataLocationInput.text().endswith('.npy'):
            #For npy, load the events in memory
            data = np.load(self.dataLocationInput.text(), mmap_mode='r')
            #Filter them on time:
            events = self.filterEvents_npy_t(data,timeStretch)
            #Check if we have at least 1 event:
            if len(events) > 0:
                # # correct the coordinates and time stamps
                # events['x']-=np.min(events['x'])
                # events['y']-=np.min(events['y'])
                # events['t']-=np.min(events['t'])
                #Log the nr of events found:
                logging.info(f"Preview - Found {len(events)} events in the chosen time frame.")
            else:
                logging.error("Preview - No events found in the chosen time frame.")
                return
        elif self.dataLocationInput.text().endswith('hdf5'):

            #Load events from HDF
            events,_ = self.timeSliceFromHDF(self.dataLocationInput.text(),requested_start_time_ms = float(timeStretch[0]),requested_end_time_ms=float(timeStretch[0])+float(timeStretch[1]),howOftenCheckHdfTime = 50000)

            #Check if we have at least 1 event:
            if len(events) > 0:
                #Log the nr of events found:
                logging.info(f"Preview - Found {len(events)} events in the chosen time frame.")
            else:
                logging.error("Preview - No events found in the chosen time frame.")
                return

            z=3
        else:
            logging.error("Please choose a .raw or .npy or .hdf5 file for previews.")
            return

        #Load the events in self memory and filter on XY
        self.previewEvents = events
        self.previewEvents = self.filterEvents_xy(self.previewEvents,xyStretch)

        self.previewEventsDict = []

        #filter on polarity:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            #Run everything twice
            #Get positive events only:
            npyevents_pos = self.filterEvents_npy_p(self.previewEvents,pValue=1)
            self.previewEventsDict.append(npyevents_pos)
            #Get negative events only:
            npyevents_neg = self.filterEvents_npy_p(self.previewEvents,pValue=0)
            self.previewEventsDict.append(npyevents_neg)
        #Only positive
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[1]:
            #Run everything once
            npyevents_pos = self.filterEvents_npy_p(self.previewEvents,pValue=1)
            self.previewEventsDict.append(npyevents_pos)
            polarityVal = 'Pos'
        #Only negative
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[2]:
            #Run everything once
            npyevents_neg = self.filterEvents_npy_p(self.previewEvents,pValue=0)
            self.previewEventsDict.append(npyevents_neg)
            polarityVal = 'Neg'
        #No discrimination
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[0]:
            #Don't do any filtering
            self.previewEventsDict.append(self.previewEvents)
            polarityVal = 'Mix'

        #Change global values so nothing is stored - we just want a preview run. This is later set back to orig values (globalSettingsOrig):
        globalSettingsOrig = copy.deepcopy(self.globalSettings)
        self.globalSettings['StoreConvertedRawData']['value'] = False
        self.globalSettings['StoreFileMetadata']['value'] = False
        # self.globalSettings['StoreFinalOutput']['StoreFinalOutput']['value'] = False
        self.globalSettings['StoreFinalOutput']['value'] = False
        self.globalSettings['StoreFindingOutput']['value'] = False
        self.globalSettings['StoreFittingOutput']['value'] = False

        events_id = 0
        partialFinding = {}
        partialFitting = {}
        for events in self.previewEventsDict:
            if self.dataSelectionPolarityDropdown.currentText() != self.polarityDropdownNames[3]:
                #Run the current finding and fitting routine only on these events:
                self.runFindingAndFitting(events,runFitting=True,storeFinding=False,polarityVal=polarityVal)
            else:
                #Run the current finding and fitting routine only on these events:
                #Should be once positive, once negative
                if all(events['p'] == 1):
                    polarityVal = 'Pos'
                elif all(events['p'] == 0):
                    polarityVal = 'Neg'
                #run the finding/fitting
                self.runFindingAndFitting(events,runFitting=True,storeFinding=False,polarityVal=polarityVal)
                #we store our finding and fitting results like this:
                partialFinding[events_id] = self.data['FindingResult']
                partialFitting[events_id] = self.data['FittingResult']

            events_id+=1

        #If the data was split in two parts... (or more)
        if events_id > 1:
            updated_partialFinding = []
            updated_partialFindingMetadatastring = ''
            updated_partialFitting = []
            updated_partialFittingMetadatastring = ''
            totNrFindingIncrease = 0
            for i in range(events_id):
                updated_partialFindingMetadatastring = updated_partialFindingMetadatastring+partialFinding[i][1]+'\n'
                updated_partialFittingMetadatastring = updated_partialFittingMetadatastring+partialFitting[i][1]+'\n'
                for eachEntry in partialFinding[i][0].items():
                    updated_partialFinding.append(eachEntry[1])
                for index,row in partialFitting[i][0].iterrows():
                    row.candidate_id+=totNrFindingIncrease
                    updated_partialFitting.append(row)


                #increase the total number of findings
                totNrFindingIncrease+=eachEntry[0]+1

            #Store them again in the self.data['FindingResult']
            self.data['FindingResult']={}
            self.data['FindingResult'][0] = dict(zip(range(len(updated_partialFinding)), updated_partialFinding))
            self.data['FindingResult'][1] = updated_partialFindingMetadatastring
            #Fitting should be changed to pd df
            res_dict_fitting = pd.DataFrame(updated_partialFitting)
            #Store them again in the self.data['FindingResult']
            self.data['FittingResult']={}
            self.data['FittingResult'][0] = res_dict_fitting
            self.data['FittingResult'][1] = updated_partialFittingMetadatastring

            #To test: np.shape(self.data['FittingResult'][0])[0]

        #Reset global settings
        self.globalSettings = globalSettingsOrig

        #Update the preview panel and localization list:
        #Note that self.previewEvents is XYT cut, but NOT p-cut
        self.updateShowPreview(previewEvents=self.previewEvents,timeStretch=timeStretch,frameTime=frameTime)
        self.updateLocList()

    def timeSliceFromHDF(self,dataLocation,requested_start_time_ms = 0,requested_end_time_ms=1000,howOftenCheckHdfTime = 100000,loggingBool=False,curr_chunk = 0):
        """Function that returns all events between start/end time in a HDF5 file. Extremely sped-up since the HDF5 file is time-sorted, and only checked every 100k (howOftenCheckHdfTime) events.

        Args:
            dataLocation (String): Storage location of the .hdf5 file
            requested_start_time_ms (int, optional): Start time in milliseconds. Defaults to 0.
            requested_end_time_ms (int, optional): End time in milliseconds. Defaults to 1000.
            howOftenCheckHdfTime (int, optional): At which N intervals the time should be checked. This means that HDF event 0,N*howOftenCheckHdfTime,(N+1)*howOftenCheckHdfTime etc will be checked and pre-loaded. After this, all events within the time bounds is loaded. Defaults to 100000.
            loggingBool (bool, optional): Whether or not logging is activated. Defaults to True.
            curr_chunk (int, optional): Starting chunk to look at. Normally should be 0. Defaults to 0.

        Returns:
            events: Events in wanted format
            latest_chunk: Last chunk that was used. Can be used to run this function more often via curr_chunk.
        """
        #Variable starting
        lookup_start_index = -1
        lookup_end_index = -1


        #Load the hdf5 file
        with h5py.File(dataLocation, mode='r') as file:
            #Events are here in this file:
            events_hdf5 = file['CD']['events']

            #Loop while either start or end index hasn't been found yet
            while lookup_start_index == -1 or lookup_end_index == -1:
                index = curr_chunk*howOftenCheckHdfTime


                if index <= events_hdf5.size:
                    #Get the time
                    foundtime = events_hdf5[index]['t']/1000 #in ms

                    if loggingBool == True:
                        logging.info('Loading HDF, currently on chunk '+str(curr_chunk)+', at time: '+str(foundtime))

                    #Check if the start time has surpassed
                    if foundtime > requested_start_time_ms:
                        if lookup_start_index == -1:
                            lookup_start_index = max(0,curr_chunk-1)*howOftenCheckHdfTime
                    #Check if the end-time is surpassed
                    if foundtime > requested_end_time_ms:
                        if lookup_end_index == -1:
                            lookup_end_index = max(1,curr_chunk+1)*howOftenCheckHdfTime
                    #Increase current chunk
                    curr_chunk+=1
                else:
                    logging.info('End of file reached while chunking HDF5')
                    if lookup_start_index == -1:
                        lookup_start_index = events_hdf5.size #Set to end of file
                    if lookup_end_index == -1:
                        lookup_end_index = events_hdf5.size #Set to end of file

            #Properly (32bit) initialise dicts
            wantedEvents_tooLarge = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})
            events_output = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})

            #Now we know the start/end index, so we cut out that area:
            wantedEvents_tooLarge = events_hdf5[lookup_start_index:lookup_end_index]
            #And we fully cut it to exact size:
            events_output = wantedEvents_tooLarge[wantedEvents_tooLarge['t']/1000>=requested_start_time_ms]
            events_output = events_output[events_output['t']/1000<=requested_end_time_ms]

        #Return the events
        return events_output,curr_chunk

    def filterEvents_npy_t(self,events,tStretch=(-np.Inf,np.Inf)):
        """
        Filter events that are in a numpy array to a certain time-stretch.
        """
        #tStretch is (start, duration)
        indices = np.where((events['t'] >= float(tStretch[0])*1000) & (events['t'] <= float(tStretch[0])*1000+float(tStretch[1])*1000))
        # Access the partial data using the indices
        events = events[indices]

        #Warning if no events are found
        if len(events) == 0:
            logging.warning("No events found in the chosen time frame.")

        return events

    def filterEvents_npy_p(self,events,pValue=0):
        """
        Filter events that are in a numpy array to a certain polarity
        """
        #tStretch is (start, duration)
        indices = np.where((events['p'] == pValue))
        # Access the partial data using the indices
        eventsFiltered = events[indices]

        #Warning if no events are found
        if len(eventsFiltered) == 0:
            logging.warning("No events found with the chosen polarity: "+str(pValue))

        return eventsFiltered

    def filterEvents_xy(self,events,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
        """
        Filter events that are in a numpy array to a certain xy stretch.
        """
        #Edit values for x,y coordinates:
        #First check if all entries in array are numbers:
        try:
            xyStretch = (float(xyStretch[0]), float(xyStretch[1]), float(xyStretch[2]), float(xyStretch[3]))
            #Check if they all are floats:
            if not all(isinstance(x, float) for x in [float(xyStretch[0]),float(xyStretch[1]),float(xyStretch[2]),float(xyStretch[3])]):
                logging.info("No XY cutting due to not all entries being floats.")
            #If they are all float values, we can proceed
            else:
                if (xyStretch[0] > 0) | (xyStretch[2] > 0) | (xyStretch[1] < np.inf) | (xyStretch[3] < np.inf):
                    logging.info("XY cutting to values: "+str(xyStretch[0])+","+str(xyStretch[1])+","+str(xyStretch[2])+","+str(xyStretch[3]))
                    #Filter on x,y coordinates:
                    events = events[(events['x'] >= float(xyStretch[0])) & (events['x'] <= float(xyStretch[1]))]
                    events = events[(events['y'] >= float(xyStretch[2])) & (events['y'] <= float(xyStretch[3]))]
        except:
            #Warning if something crashes. Note the extra dash to the end of the warning
            logging.warning("No XY cutting due to not all entries being float-.")

        return events

    def setup_postProcessingTab(self):
        """
        Dummy function that set up the post-processing tab
        """
        tab2_layout = QGridLayout()
        self.tab_postProcessing.setLayout(tab2_layout)

        self.postProcessingtab_widget = PostProcessing(self)
        self.tab_postProcessing.layout().addWidget(self.postProcessingtab_widget)

    def setup_logFileTab(self):
        """
        Function that set up the logger lab ('Run Info')
        """
        #It's simply a grid layout that only contains a text edit. This text edit is filled with the Log file ever 1 second
        tab_layout = QGridLayout()
        self.tab_logfileInfo.setLayout(tab_layout)
        self.text_edit = QTextEdit()
        tab_layout.addWidget(self.text_edit, 0, 0)

        self.last_modified = os.path.getmtime(self.log_file_path)

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_logfile_modification)
        self.timer.start(1000)  # Check every second

        self.update_log()

    def check_logfile_modification(self):
        """
        Function checking for modification of the .log file, and if it changed, the text edit is updated
        """
        current_modified = os.path.getmtime(self.log_file_path)
        if current_modified != self.last_modified:
            self.update_log()
            self.last_modified = current_modified

    def update_log(self):
        """
        Function that updates the text of the 'run info' tab if the log file has been modified
        """
        if QFile.exists(self.log_file_path):
            with open(self.log_file_path, 'r') as file:
                log_contents = file.read()

                #check every line, and if it contains 'ERROR', change it to red font:
                lines = log_contents.split('\n')
                for i, line in enumerate(lines):
                    if '[ERROR]' in line:
                        lines[i] = f'<font color="red">{line}</font>'
                    elif '[WARNING]' in line:
                        lines[i] = f'<font color="orange">{line}</font>'
                    lines[i]+='<br>'

                updated_log = '\n'.join(lines)
                self.text_edit.setHtml(updated_log)

                self.text_edit.moveCursor(QTextCursor.End)
                self.text_edit.ensureCursorVisible()

    def setup_loclistTab(self):
        """
        Function that's setting up the localization list tab
        """
        #Simply a grid layout that contains an (empty to start with) table:
        tab4_layout = QGridLayout()
        self.tab_locList.setLayout(tab4_layout)

        #Create an empty table and add this:
        self.LocListTable = QTableView()
        tab4_layout.addWidget(self.LocListTable, 0, 0)

        #Add an empty horizontal widget to load a .csv:
        self.CSVReadLayout = QHBoxLayout()

        #Add an empty line edit for a .csv loading string:
        self.CSVlocationLineEdit = QLineEdit()
        self.CSVReadLayout.addWidget(self.CSVlocationLineEdit)

        #Add a ... button that opens a file dialog:
        self.findCSVlocationButton = QPushButton("...")
        self.findCSVlocationButton.clicked.connect(self.fileDialogCSVopen)
        self.CSVReadLayout.addWidget(self.findCSVlocationButton)

        #Also add a button to read the csv
        self.buttonReadCSV = QPushButton("Read CSV")
        self.buttonReadCSV.clicked.connect(self.open_loclist_csv)
        self.CSVReadLayout.addWidget(self.buttonReadCSV)

        #Add a horizontal divider line
        horizontal_line = QFrame(self)
        horizontal_line.setFrameShape(QFrame.VLine)
        horizontal_line.setFrameShadow(QFrame.Sunken)
        self.CSVReadLayout.addWidget(horizontal_line)

        #And add a save CSV button
        self.buttonSaveCSV = QPushButton("Save CSV")
        self.buttonSaveCSV.clicked.connect(self.save_loclist_csv)
        self.CSVReadLayout.addWidget(self.buttonSaveCSV)

        tab4_layout.addLayout(self.CSVReadLayout, 1, 0)

    def fileDialogCSVopen(self):
        #Open a file dialog:
        fname = QFileDialog.getOpenFileName(self, 'Select a localization CSV file', None,"CSV file (*.csv)")
        #Set this selected file as line edit:
        self.CSVlocationLineEdit.setText(fname[0])
        #Then also read this file:
        self.open_loclist_csv()

    def save_loclist_csv(self):
        #Open a file dialog:
        fname,_ = QFileDialog.getSaveFileName(self, 'Storage location', None,"CSV file (*.csv)")
        localizations = self.data['FittingResult'][0]
        #Store the localizations
        self.storeLocalization(fname,localizations,outputType='thunderstorm')
        logging.info('Localizations stored to ' + fname)

    def open_loclist_csv(self):
        loclistcsvloc = self.CSVlocationLineEdit.text()

        if loclistcsvloc == '': #If it's an empy line, actually go to filedialog
            self.fileDialogCSVopen() #this contains a new call to open_loclist_csv
        else:#if it's actually a file, read it:
            #Read the csv:
            loclist = pd.read_csv(loclistcsvloc)

            #Rename some of the headers:
            loclist.rename(columns={'x [nm]': 'x'}, inplace=True)
            loclist.rename(columns={'y [nm]': 'y'}, inplace=True)
            loclist.rename(columns={'t [ms]': 't'}, inplace=True)

            #Set this as result:
            self.data['FittingResult'] = {}
            self.data['FittingResult'][0] = loclist
            self.updateLocList()
            logging.info('CSV loaded, loclist updated')
            #Also clear the post-processing history since it's a completely new dataset
            self.postProcessingtab_widget.PostProcessingHistoryGrid.clearHistory()

    def setup_visualisationTab(self):
        """
        Function to set up the Visualisation tab (scatter, average shifted histogram and such)
        """


        #It's a grid layout
        self.visualisationtab_layout = QGridLayout()
        self.tab_visualisation.setLayout(self.visualisationtab_layout)

        self.visualisationtab_widget = VisualisationNapari(self)
        self.visualisationtab_layout.addWidget(self.visualisationtab_widget)
        # #Add a vertical layout, not a grid layout:
        # visualisationTab_vertical_container = QVBoxLayout()
        # self.tab_visualisation.setLayout(visualisationTab_vertical_container)

        # #Add a horizontal layout to the first row of the vertical layout - this contains the buttons:
        # visualisationTab_horizontal_container = QHBoxLayout()
        # visualisationTab_vertical_container.addLayout(visualisationTab_horizontal_container)

        # #Add a button that says scatter:
        # self.buttonScatter = QPushButton("Scatter Plot")
        # visualisationTab_horizontal_container.addWidget(self.buttonScatter)
        # #Give it a function on click:
        # self.buttonScatter.clicked.connect(lambda: self.plotScatter())

        # #Add a button that says 2d interp histogram:
        # self.buttonInterpHist = QPushButton("Interp histogram")
        # visualisationTab_horizontal_container.addWidget(self.buttonInterpHist)
        # #Give it a function on click:
        # self.buttonInterpHist.clicked.connect(lambda: self.plotLinearInterpHist())

        # #Create an empty figure and store it as self.data:
        # self.data['figurePlot'], self.data['figureAx'] = plt.subplots(figsize=(5, 5))
        # self.data['figureCanvas'] = FigureCanvas(self.data['figurePlot'])
        # self.data['figurePlot'].tight_layout()

        # #Add a navigation toolbar (zoom, pan etc)
        # visualisationTab_vertical_container.addWidget(NavigationToolbar(self.data['figureCanvas'], self))
        # #Add the canvas to the tab
        # visualisationTab_vertical_container.addWidget(self.data['figureCanvas'])

    def setup_canPreviewTab(self):
        """
        Function to setup the Candidate Preview tab
        """

        #It's a grid layout
        self.canPreviewtab_layout = QGridLayout()
        self.tab_canPreview.setLayout(self.canPreviewtab_layout)

        self.canPreviewtab_widget = CandidatePreview(self)
        self.canPreviewtab_layout.addWidget(self.canPreviewtab_widget)

    def setup_previewTab(self):
        """
        Function that creates the preview tab
        Very practically, it creates 10 figures, stores those in memory, and displays them when necessary
        """
        #It's a grid layout
        self.previewtab_layout = QGridLayout()
        self.tab_previewVis.setLayout(self.previewtab_layout)

        #And I add the previewfindingfitting class (QWidget extension):
        self.previewtab_widget = PreviewFindingFitting(parent=self)
        self.previewtab_layout.addWidget(self.previewtab_widget)

    def updateShowPreview(self,previewEvents=None,timeStretch=None,frameTime=100):
        """
        Function that's called to update the preview (or show it). Requires the previewEvents, or uses the self.previewEvents.
        """

        self.previewtab_widget.displayEvents(previewEvents,frametime_ms=frameTime,findingResult = self.data['FindingResult'][0],fittingResult=self.data['FittingResult'][0],settings=self.globalSettings,timeStretch=timeStretch)


        logging.info('UpdateShowPreview ran!')

    def previewEventStartedOnThisFrame(self,row,frame):
        """
        Function that checks if a preview event *started* on this frame
        """
        if min(row['events']['t']) >= frame*self.PreviewFrameTime and min(row['events']['t']) < (frame+1)*self.PreviewFrameTime:
            return True
        else:
            return False

    def previewEventHappensOnThisFrame(self,row,frame):
        """
        Function that checks if a preview event *is happening* on this frame
        """
        if max(row['events']['t']) >= frame*self.PreviewFrameTime and min(row['events']['t']) < (frame+1)*self.PreviewFrameTime:
            return True
        else:
            return False

    def previewEventEndsOnThisFrame(self,row,frame):
        """
        Function that checks if a preview event *ended* on this frame
        """
        if max(row['events']['t']) >= frame*self.PreviewFrameTime and max(row['events']['t']) < (frame+1)*self.PreviewFrameTime:
            return True
        else:
            return False

    def createRectangle(self,fig,data,col,padding=0):
        """
        Helper function to create a rectangle with certain padding
        """
        x_min = min(data['x'])-.5-padding
        y_min = min(data['y'])-.5-padding
        width = (max(data['x']) - min(data['x']))+padding*2+1
        height = (max(data['y']) - min(data['y']))+padding*2+1
        rect = patches.Rectangle((x_min,y_min),width,height, edgecolor=col, facecolor='none',alpha=0.5)
        # Add the rectangle to the axes
        fig.add_patch(rect)

    def showText(self,fig,data,strv,col,padding=0):
        """
        Helper function to show text at a certain position
        """
        #Add the text!
        fig.text(max(data['x']),max(data['y']),str(strv),color=col)
    def find_raw_npy_files(self,directory):
        raw_files = glob.glob(os.path.join(directory, "*.raw"))
        npy_files = glob.glob(os.path.join(directory, "*.npy"))
        files = npy_files+raw_files

        unique_files = []
        file_names = set()

        for file in files:
            file_name = os.path.splitext(file)[0]
            if file_name not in file_names:
                unique_files.append(file)
                file_names.add(file_name)

        return unique_files

    def updateGUIafterNewResults(self,error=None):
        if error == None:
            self.updateLocList()
            #Also clear the post-processing history since it's a completely new dataset
            self.postProcessingtab_widget.PostProcessingHistoryGrid.clearHistory()
        else:
            self.open_critical_warning(error)

    def updateLocList(self):
        #data is stored in self.data['FittingResult'][0]
        #fill the self.LocListTable QTableWidget with the data:
        localizations = self.data['FittingResult'][0].dropna(axis=0, ignore_index=True)
        try:
            localizations = localizations.drop('fit_info', axis=1)
        except:
            pass

        # Define the number of significant digits for each column
        significant_digits = {'x': 2, 'y': 2, 'p': 0, 'id':0,'candidate_id':0,'del_x':2,'del_y':2, 't':0}

        for y in range(len(localizations.columns)):
            significant_digit = significant_digits.get(localizations.columns[y])
            if significant_digit is not None:
                localizations[localizations.columns[y]] = localizations[localizations.columns[y]].apply(lambda x: round(x, significant_digit))

        self.LocListTable.setModel(TableModel(table_data = localizations))
        return

    def checkPolarity(self,npyData):
        #Ensure that polarity is 0/1, not -1/1. If it is -1/1, convert to 0/1. Otherwise, give error
        if sum(np.unique(npyData['p']) == (0,1)) != 2:
            if sum(np.unique(npyData['p']) == (-1,1)) == 2:
                df = pd.DataFrame(npyData, columns=['x', 'y', 'p' ,'t'])
                df.loc[df['p'] == -1, 'p'] = 0
                # npyData = df.to_numpy(dtype=npyData.dtype)
                npyData = df.to_records(index=False)
            else:
                logging.critical('RAW/NPY data does not have 0/1 or -1/1 polarity! Please fix this and try again.')

        return npyData

    def loadRawData(self,dataLocation):
        """
        Load the raw (NPY or RAW)data from the specified location.

        Returns:
            - If the data location is empty, returns None.
            - If the data location does not end with ".npy" or ".raw", returns None.
            - If the data location ends with ".npy", returns the loaded numpy array.
            - If the data location ends with ".raw", returns None.
        """
        #Check if self.dataLocationInput.text() is not empty:
        if dataLocation == "":
            logging.error('No data location specified')
            return None
        #Check if it ends with npy or raw:
        if not dataLocation.endswith('.npy') and not dataLocation.endswith('.raw'):
            logging.error('Data location must end with .npy or .raw')
            return None
            #Load the data:
        if dataLocation.endswith('.npy'):
            eventsDict = []
            npyevents = np.load(dataLocation, mmap_mode='r')

            #First filter all events on xy,t:
            npyevents = self.filterEvents_npy_t(npyevents,tStretch=(float(self.run_startTLineEdit.text()),float(self.run_durationTLineEdit.text())))
            npyevents = self.filterEvents_xy(npyevents,xyStretch=(float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text()),float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())))

            #Determine whether two or one outputs needs to be returned - based on polarity option
            if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
                #Run everything twice
                #Get positive events only:
                npyevents_pos = self.filterEvents_npy_p(npyevents,pValue=1)
                eventsDict.append(npyevents_pos)
                #Get negative events only:
                npyevents_neg = self.filterEvents_npy_p(npyevents,pValue=0)
                eventsDict.append(npyevents_neg)
            elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[1]:
                #Run everything once
                npyevents_pos = self.filterEvents_npy_p(npyevents,pValue=1)
                eventsDict.append(npyevents_pos)
            elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[2]:
                #Run everything once
                npyevents_neg = self.filterEvents_npy_p(npyevents,pValue=0)
                eventsDict.append(npyevents_neg)
            elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[0]:
                #Don't do any filtering
                eventsDict.append(npyevents)

            return eventsDict
        elif dataLocation.endswith('.raw'):
            eventsDict = []
            events = self.RawToNpy(dataLocation)
            #Select xytp area specified:
            events = self.filterEvents_npy_t(events,tStretch=(float(self.run_startTLineEdit.text()),float(self.run_durationTLineEdit.text())))
            events = self.filterEvents_xy(events,xyStretch=(float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text()),float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())))

            #append events to eventsDict:
            eventsDict.append(events)

            return eventsDict

    def RawToNpy(self,filepath,buffer_size = 5e9, n_batches=5e9):
        if(os.path.exists(filepath[:-4]+'.npy')):
            events = np.load(filepath[:-4]+'.npy')
            logging.info('NPY file from RAW was already present, loading this instead of RAW!')
        else:
            logging.info('Starting to convert NPY to RAW...')
            sys.path.append(self.globalSettings['MetaVisionPath']['value'])
            from metavision_core.event_io.raw_reader import RawReader
            record_raw = RawReader(filepath,max_events=int(buffer_size))
            sums = 0
            time = 0
            events=np.empty
            while not record_raw.is_done() and record_raw.current_event_index() < buffer_size:
                #Load a batch of events
                events_temp = record_raw.load_n_events(n_batches)
                sums += events_temp.size
                #Add the events in this batch to the big array
                if sums == events_temp.size:
                    events = events_temp
                else:
                    events = np.concatenate((events,events_temp))
            record_raw.reset()
            # correct the coordinates and time stamps
            # events['x']-=np.min(events['x'])
            # events['y']-=np.min(events['y'])
            # events['t']-=np.min(events['t'])
            if self.globalSettings['StoreConvertedRawData']['value']:
                np.save(filepath[:-4]+'.npy',events)
                logging.debug('NPY file created')
            logging.info('Raw data loaded')
        return events

    def find_raw_npy_files(self,directory):
        raw_files = glob.glob(os.path.join(directory, "*.raw"))
        npy_files = glob.glob(os.path.join(directory, "*.npy"))
        files = npy_files+raw_files

        unique_files = []
        file_names = set()

        for file in files:
            file_name = os.path.splitext(file)[0]
            if file_name not in file_names:
                unique_files.append(file)
                file_names.add(file_name)

        return unique_files

    def run_processing_i(self,error=None):
        logging.info('Processing started...')
        #Get the polarity:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[0]:
            polarityVal = 'Mix'
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[1]:
            polarityVal = 'Pos'
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[2]:
            polarityVal = 'Neg'
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            #TODO: BOTH OF THEM AFTER ONE ANOTHER
            polarityVal = 'Mix'
        #Check if a folder or file is selected:
        if os.path.isdir(self.dataLocationInput.text()):
            #Find all .raw or .npy files that have unique names except for the extension (and prefer .npy):
            allFiles = self.find_raw_npy_files(self.dataLocationInput.text())
            logging.debug('Running folder analysis on files :')
            logging.debug(allFiles)
            for file in allFiles:
                try:
                    self.storeNameDateTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    logging.info('Starting to process file '+file)
                    self.processSingleFile(file,polarityVal=polarityVal)
                    logging.info('Successfully processed file '+file)
                except:
                    logging.error('Error in processing file '+file)
        #If it's a file...
        elif os.path.isfile(self.dataLocationInput.text()):
            self.storeNameDateTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            #Find polarity:
            #Check if we are loading existing fitting
            if 'ExistingFitting' in getattr(self,f"CandidateFittingDropdown{polarityVal}").currentText() or 'existing Fitting' in getattr(self,f"CandidateFittingDropdown{polarityVal}").currentText():
                logging.info('Skipping finding and fitting processing')
                # compare filenames of finding and fitting here? Following code is not yet adapted
                FittingEvalText = self.getFunctionEvalText('Fitting',"self.data['FindingResult'][0]","self.globalSettings",polarityVal)
                match = re.search(r'File_Location="([^"]+)"', FittingEvalText)
                FittingResults = match.group(1)
                with open(FittingResults, 'rb') as file:
                    FindingRef = pickle.load(file)
                if FindingRef == '':
                    logging.error('No finding results are stored for these fitting results, please check the global settings and re-run finding and fitting.')
                    error = 'No finding results are stored for fitting results trying to load.'
                elif not os.path.exists(FindingRef):
                    logging.error('Cannot find finding results, please re-run finding and fitting and save corresponding results.')
                    error = 'FileNotFound: Cannot find finding results.'
                else:
                    if 'ExistingFinding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText() or 'existing Finding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText():
                        FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings",polarityVal)
                        emptyPath = "File_Location=\"\""
                        if FindingEvalText == None:
                            FindingResults = ''
                        if FindingEvalText != None:
                            match = re.search(r'File_Location="([^"]+)"', FindingEvalText)
                            if match:
                                FindingResults = match.group(1)
                        if FindingResults != FindingRef:
                            logging.warning('Existing finding and fitting do not match, try to reset finding path to correct file.')
                            self.setFilepathExisting("Finding", polarityVal, FindingRef)
                    else:
                        logging.warning("Existing finding and fitting do not match, try to reset finding routine.")
                        print(self.getFunctionEvalText('Finding',"npyData","self.globalSettings",polarityVal))
                        self.setMethod("Finding", polarityVal, "Load an existing Finding Result")
                        # reset comboboxes
                        self.reset_single_combobox_states()
                        print(self.getFunctionEvalText('Finding',"npyData","self.globalSettings",polarityVal))
                        self.setFilepathExisting("Finding", polarityVal, FindingRef)

                    # Ensure that we don't store the finding result
                    origStoreFindingSetting=self.globalSettings['StoreFindingOutput']['value']
                    self.globalSettings['StoreFindingOutput']['value'] = False
                    # Ensure that we don't store fitting result
                    origStoreFittingSetting=self.globalSettings['StoreFittingOutput']['value']
                    self.globalSettings['StoreFittingOutput']['value'] = False
                    self.processSingleFile(self.dataLocationInput.text(),noFindingFitting=True,polarityVal=polarityVal)

                    #Reset the global setting:
                    self.globalSettings['StoreFindingOutput']['value']=origStoreFindingSetting
                    self.globalSettings['StoreFittingOutput']['value']=origStoreFittingSetting

            else:
                if 'ExistingFinding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText() or 'existing Finding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText():
                    logging.info('Skipping finding processing, going to fitting')
                    FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings",polarityVal)
                    emptyPath = "File_Location=\"\""
                    if FindingEvalText == None:
                        FindingResults = ''
                    if FindingEvalText != None:
                        match = re.search(r'File_Location="([^"]+)"', FindingEvalText)
                        if match:
                            FindingResults = match.group(1)
                    if polarityVal == 'Pos':
                        self.data['refFindingFilepos'] = FindingResults
                    if polarityVal == 'Neg':
                        self.data['refFindingFileneg'] = FindingResults
                    else:
                        self.data['refFindingFile'] = FindingResults
                    #Ensure that we don't store the finding result
                    origStoreFindingSetting=self.globalSettings['StoreFindingOutput']['value']
                    self.globalSettings['StoreFindingOutput']['value'] = False

                    self.processSingleFile(self.dataLocationInput.text(),onlyFitting=True,polarityVal=polarityVal)

                    #Reset the global setting:
                    self.globalSettings['StoreFindingOutput']['value']=origStoreFindingSetting
                else:
                    #Otherwise normally process a single file
                    self.processSingleFile(self.dataLocationInput.text(),polarityVal=polarityVal)
        #if it's neither a file nor a folder
        else:
            logging.error('Input file/folder is not correct! Please check.')
            error = 'Input file/folder is not correct! Please check.'
        self.updateGUIafterNewResults(error)
        return

    def run_processing(self):
        #Give a clear logging separation:
        logging.info("")
        logging.info("")
        logging.info("--------------- Full Run Starting ---------------")
        logging.info("")
        logging.info("")
        
        #Switch the user to the Run info tab
        utils.changeTab(self, text='Run info')
        
        self.globalSettings['StoreFinalOutput']['value'] = True
        # self.run_processing_i()
        # reset previewEvents array, every time run is pressed
        self.previewEvents = []
        thread = threading.Thread(target=self.run_processing_i)
        thread.start()

    def updateGUIafterNewResults(self,error=None):
        if error == None:
            self.updateLocList()
        # else:
        #     self.open_critical_warning(error)

    def checkPolarity(self,npyData):
        #Ensure that polarity is 0/1, not -1/1. If it is -1/1, convert to 0/1. Otherwise, give error
        if sum(np.unique(npyData['p']) == (0,1)) != 2:
            if sum(np.unique(npyData['p']) == (-1,1)) == 2:
                df = pd.DataFrame(npyData, columns=['x', 'y', 'p' ,'t'])
                df.loc[df['p'] == -1, 'p'] = 0
                # npyData = df.to_numpy(dtype=npyData.dtype)
                npyData = df.to_records(index=False)
            else:
                logging.critical('RAW/NPY data does not have 0/1 or -1/1 polarity! Please fix this and try again.')

        return npyData

    def processSingleFile(self,FileName,onlyFitting=False,polarityVal='Mix', noFindingFitting=False):

        self.data['FindingResult']={}
        self.data['FittingResult']={}
        #Runtime of finding and fitting
        self.currentFileInfo['FindingTime'] = 0
        self.currentFileInfo['FittingTime'] = 0

        if not noFindingFitting:
            if not onlyFitting:
                #Run the analysis on a single file
                self.currentFileInfo['CurrentFileLoc'] = FileName
                if self.globalSettings['FindingBatching']['value']== False:
                    npyDataCell = self.loadRawData(FileName)
                    #Note that npyDataCell has 2 entries if pos/neg are treated seperately, otherwise just one entry:
                    #Logic for if there are multiple entires in npyDataCell
                    events_id = 0
                    partialFinding = {}
                    partialFitting = {}
                    for npyData in npyDataCell:
                        if npyData is None:
                            return

                        #Sort event list on time
                        npyData = npyData[np.argsort(npyData,order='t')]

                        if self.dataSelectionPolarityDropdown.currentText() != self.polarityDropdownNames[3]:
                            #Run the current finding and fitting routine only on these events:
                            self.runFindingAndFitting(npyData,runFitting=True,storeFinding=True,polarityVal=polarityVal)
                        else:
                            #Run the current finding and fitting routine only on these events:
                            if np.all(npyData['p'] == 1):
                                polarityVal = 'Pos'
                            elif np.all(npyData['p'] == 0):
                                polarityVal = 'Neg'
                            self.runFindingAndFitting(npyData,runFitting=True,storeFinding=True,polarityVal=polarityVal)
                            #we store our finding and fitting results like this:
                            partialFinding[events_id] = self.data['FindingResult']
                            partialFitting[events_id] = self.data['FittingResult']

                        events_id+=1

                    #If the data was split in two parts... (or more)
                    if events_id > 1:
                        updated_partialFinding = []
                        updated_partialFindingMetadatastring = ''
                        updated_partialFitting = []
                        updated_partialFittingMetadatastring = ''
                        totNrFindingIncrease = 0
                        for i in range(events_id):
                            updated_partialFindingMetadatastring = updated_partialFindingMetadatastring+partialFinding[i][1]+'\n'
                            updated_partialFittingMetadatastring = updated_partialFittingMetadatastring+partialFitting[i][1]+'\n'
                            for eachEntry in partialFinding[i][0].items():
                                updated_partialFinding.append(eachEntry[1])
                            for index,row in partialFitting[i][0].iterrows():
                                row.candidate_id+=totNrFindingIncrease
                                updated_partialFitting.append(row)

                            #increase the total number of findings
                            totNrFindingIncrease+=eachEntry[0]+1

                        #Store them again in the self.data['FindingResult']
                        self.data['FindingResult']={}
                        self.data['FindingResult'][0] = dict(zip(range(len(updated_partialFinding)), updated_partialFinding))
                        self.data['FindingResult'][1] = updated_partialFindingMetadatastring
                        #Fitting should be changed to pd df
                        res_dict_fitting = pd.DataFrame(updated_partialFitting)
                        #Store them again in the self.data['FindingResult']
                        self.data['FittingResult']={}
                        self.data['FittingResult'][0] = res_dict_fitting
                        self.data['FittingResult'][1] = updated_partialFittingMetadatastring


                elif self.globalSettings['FindingBatching']['value']== True or self.globalSettings['FindingBatching']['value']== 2:
                    self.runFindingBatching()
            #If we only fit, we still run more or less the same info, butwe don't care about the npyData in the CurrentFileLoc.
            elif onlyFitting:
                self.currentFileInfo['CurrentFileLoc'] = FileName
                logging.info('Candidate finding NOT performed')
                npyData = None
                if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
                    self.runFindingAndFitting(npyData,polarityVal='Pos')
                    #Get the number of positive candidates:

                    self.runFindingAndFitting(npyData,polarityVal='Neg',findingOffset=len(self.data['FindingResult'][0]))
                else:
                    self.runFindingAndFitting(npyData,polarityVal=polarityVal)
        elif noFindingFitting:
            self.currentFileInfo['CurrentFileLoc'] = FileName
            logging.info('Candidate finding and fitting NOT performed')
            npyData = None
            if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
                self.runFindingAndFitting(npyData,polarityVal='Pos')
                self.runFindingAndFitting(npyData,polarityVal='Neg',findingOffset=len(self.data['FindingResult'][0]),fittingOffset=len(self.data['FindingResult'][0]))
            else:
                self.runFindingAndFitting(npyData,polarityVal=polarityVal)

    def FindingBatching(self,npyData,polarityVal):
        #Get polarity info and do this:
        FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings",polarityVal)
        if FindingEvalText is not None:
            self.data['FindingMethod'] = str(FindingEvalText)
            BatchFindingResult = eval(str(FindingEvalText))

            #Append the metadata
            self.data['FindingResult'][1] = np.append(self.data['FindingResult'][1],BatchFindingResult[1])

            #Append to the fullr esult
            for k in range(len(BatchFindingResult[0])):
                try:
                    if self.filter_finding_on_chunking(BatchFindingResult[0][k],self.chunckloading_currentLimits):
                        # append BatchFindingResult[0][k] to the full result:
                        #Shouldn't be appended, but should be a new entry in the dict:
                        self.data['FindingResult'][0][len(self.data['FindingResult'][0])] = BatchFindingResult[0][k]
                except:
                    print('issues with index '+str(k))

    def filter_finding_on_chunking(self,candidate,chunking_limits):
        #Return true if it should be in this chunk, false if not

        #Looking at end of chunk:
        #Special case: start is after the overlap-start of next, and end is before the overlap-end of this:
        if min(candidate['events']['t']) > chunking_limits[0][1]-(chunking_limits[1][1]-chunking_limits[0][1]) and max(candidate['events']['t']) < chunking_limits[1][1]:
            #This will mean this candidate will be found in this chunk and in the next chunk:
            #Check if the mean t is in this frame or not:
            meant = np.mean(candidate['events']['t'])
            if meant<chunking_limits[0][1]:
                return True
            else:
                return False
        #Looking at start of chunk:
        #Special case: start is after the overlap-start of previous, and end is before the overlap-end of this:
        elif min(candidate['events']['t']) > chunking_limits[1][0] and max(candidate['events']['t']) < chunking_limits[0][0]+(chunking_limits[1][1]-chunking_limits[0][1]):
            #This will mean this candidate will be found in this chunk and in the previous chunk:
            #Check if the mean t is in this frame or not:
            meant = np.mean(candidate['events']['t'])
            if meant>chunking_limits[0][0]:
                return True
            else:
                return False
        #Clear pass: start is after the true start of this, end is before the true end of this:
        elif min(candidate['events']['t']) > chunking_limits[0][0] and max(candidate['events']['t']) < chunking_limits[0][1]:
            return True
        #Looking at end of chunk:
        #Clear fail: start is after the true end of this
        elif min(candidate['events']['t']) > chunking_limits[0][1]:
            return False
        #Looking at start of chunk:
        #Clear fail: end is before the true start of this chunk
        elif max(candidate['events']['t']) < chunking_limits[0][0]:
            return False
        else:
            if max(candidate['events']['t'])-min(candidate['events']['t']) > (chunking_limits[1][1]-chunking_limits[0][1]):
                logging.warning('This candidate might be cut off due to batching! Considering increasing overlap!')
                return True
            else:
                logging.error('This candidate is never assigned! Should not happen!')
                return False

    def obtainPolarityValFromPolarityDropdown(self):
        #Get the polarity:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[0]:
            polarityVal = ['Mix']
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[1]:
            polarityVal = ['Pos']
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[2]:
            polarityVal = ['Neg']
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            #NOTE: Do both after one another
            polarityVal = ['Pos','Neg']
        return polarityVal

    def runFindingBatching(self):
        logging.info('Batching-dependant finding starting!')
        fileToRun = self.currentFileInfo['CurrentFileLoc']
        self.currentFileInfo['FindingTime'] = time.time()
        self.data['FindingResult'] = {}
        self.data['FindingResult'][0] = {}
        self.data['FindingResult'][1] = []

        #Get some logic on polarity:
        #Get the polarity:
        polarityValArray = self.obtainPolarityValFromPolarityDropdown()

        #For batching, we only load data between some timepoints.
        if fileToRun.endswith('.raw'):
            logging.info('Starting with RAW for chunking')
            #Empty dict to keep track of how many pos/neg psfs were found
            self.number_finding_found_polarity = {}
            sys.path.append(self.globalSettings['MetaVisionPath']['value'])
            from metavision_core.event_io.raw_reader import RawReader

            for polarityVal in polarityValArray:
                logging.info('Starting Batching with polarity: '+polarityVal)

                if not ('ExistingFinding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText() or 'existing Finding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText()):

                    record_raw = RawReader(fileToRun,max_events=int(5e7)) #TODO: maybe the 5e9 should be user-defined? I think this is memory-based.

                    #Seek to start time according to specifications
                    record_raw.seek_time(float(self.run_startTLineEdit.text())*1000)


                    #Read all chunks:
                    self.chunckloading_number_chuck = 0
                    events_prev = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})
                    self.chunckloading_finished_chunking = False
                    self.chunckloading_currentLimits = [[0,0],[0,0]]

                    while self.chunckloading_finished_chunking == False:
                        logging.info('New chunk analysis starting')
                        if self.chunckloading_number_chuck == 0:
                            events = record_raw.load_delta_t(float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000)
                        else:
                            events = record_raw.load_delta_t(float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000)
                        #Check if any events are still within the range of time we want to assess
                        if len(events) > 0:


                            if (min(events['t']) < (float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text()))*1000):
                                #limit to requested xy
                                events = self.filterEvents_xy(events,xyStretch=(float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text()),float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())))
                                #limit to requested t
                                # maxT = (float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text()))*1000
                                # events = events[events['t']<maxT]

                                # logging.warning('RAW2 Current event min/max time:'+str(min(events['t'])/1000)+"/"+str(max(events['t'])/1000))

                                #self.chunckloading_currentLimits is later used to check which PSFs are in the current chunk
                                self.chunckloading_currentLimits = [[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000],[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000-float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000]]
                                #Add the starting time
                                self.chunckloading_currentLimits = [[float(self.run_startTLineEdit.text())*1000 + x for x in sublist] for sublist in self.chunckloading_currentLimits]
                                #Add events_prev before these events:
                                events = np.concatenate((events_prev,events))
                                #Limit to the wanted minimum time
                                # maxT = (float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text()))*1000
                                # events = events[events['t']<maxT]



                                if len(events)>0:
                                    logging.info('Current event min/max time:'+str(min(events['t'])/1000)+"/"+str(max(events['t'])/1000))
                                    #Filter on correct polarity
                                    if polarityVal == "Pos":
                                        eventsPol = self.filterEvents_npy_p(events,1)
                                    elif polarityVal == "Neg":
                                        eventsPol = self.filterEvents_npy_p(events,0)
                                    elif polarityVal == "Mix":
                                        eventsPol = events

                                    if (len(eventsPol) > 0):
                                        self.FindingBatching(eventsPol,polarityVal)
                                        self.chunckloading_number_chuck += 1

                                        #Keep the previous 'overlap-events' for next round
                                        events_prev = events[events['t']>self.chunckloading_currentLimits[0][1]-(self.chunckloading_currentLimits[1][1]-self.chunckloading_currentLimits[0][1])]
                            else:
                                logging.info('Finished chunking!')
                                self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                                self.chunckloading_finished_chunking = True
                        else:
                            logging.info('Finished chunking!')
                            self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                            self.chunckloading_finished_chunking = True

                else: #If we want to pre-load the existing finding info:
                    findingOffset = 0
                    if polarityVal == 'Neg':
                        findingOffset = self.number_finding_found_polarity['Pos']

                    self.runFindingAndFitting('',runFitting=False,storeFinding=False,polarityVal=polarityVal,findingOffset = findingOffset,fittingOffset=findingOffset)
                    self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                    z=3

                    # customFindingEval = ["{key: value for key, value in self.data['FindingResult'][0].items() if key < "+str(self.number_finding_found_polarity['Pos'])+"}","{key-"+str(self.number_finding_found_polarity['Pos']-1)+": value for key, value in self.data['FindingResult'][0].items() if key >= "+str(self.number_finding_found_polarity['Pos'])+"}"]

        elif self.dataLocationInput.text().endswith('.npy'):
            logging.info('Loading npy file into memory for chuncking')
            #For npy, load the events in memory
            data = np.load(self.dataLocationInput.text(), mmap_mode='r')
            dataset_minx = data['x'].min()
            dataset_miny = data['y'].min()

            #Empty dict to keep track of how many pos/neg psfs were found
            self.number_finding_found_polarity = {}

            for polarityVal in polarityValArray:
                logging.info('Starting Batching with polarity: '+polarityVal)
                #Limit time correctly
                if float(self.run_startTLineEdit.text()) > data['t'].min()/1000 or float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text()) < data['t'].max()/1000:
                    #Fully limit to t according to specifications
                    data = self.filterEvents_npy_t(data,tStretch=(float(self.run_startTLineEdit.text()),float(self.run_durationTLineEdit.text())))

                #Filter on correct polarity
                if polarityVal == "Pos":
                    data_pol = self.filterEvents_npy_p(data,1)
                elif polarityVal == "Neg":
                    data_pol = self.filterEvents_npy_p(data,0)
                elif polarityVal == "Mix":
                    data_pol = data

                #set start time to zero (required for chunking):
                # data_pol['t'] -= min(data_pol['t'])

                #Read all chunks:
                self.chunckloading_number_chuck = 0
                self.chunckloading_finished_chunking = False
                self.chunckloading_currentLimits = [[0,0],[0,0]]
                #Loop over the chunks:
                while self.chunckloading_finished_chunking == False:
                    #Find the current time limits (inner, outer time)
                    self.chunckloading_currentLimits = [[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000],[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000-float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000]]
                    #Add the starting time
                    self.chunckloading_currentLimits = [[float(self.run_startTLineEdit.text())*1000 + x for x in sublist] for sublist in self.chunckloading_currentLimits]
                    #Check which are within the limits
                    indices = np.logical_and((data_pol['t'] >= self.chunckloading_currentLimits[1][0]), (data_pol['t'] <= self.chunckloading_currentLimits[1][1]))


                    #Filter to xy
                    events = self.filterEvents_xy(data_pol,xyStretch=(dataset_minx+float(self.run_minXLineEdit.text()),dataset_minx+float(self.run_maxXLineEdit.text()),dataset_miny+float(self.run_minYLineEdit.text()),dataset_miny+float(self.run_maxYLineEdit.text())))
                    #and filter to time
                    events = self.filterEvents_npy_t(events,tStretch=(self.chunckloading_currentLimits[1][0]/1000,self.chunckloading_currentLimits[1][1]/1000))

                    if len(events) > 0:
                        #Check if any events are still within the range of time we want to assess
                        if min(events['t']) < (float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text()))*1000:
                            if sum(indices) > 0:
                                #Do the actual finding
                                self.FindingBatching(events,polarityVal)
                                self.chunckloading_number_chuck+=1
                            else:
                                logging.info('Finished chunking!')
                                self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                                self.chunckloading_finished_chunking = True
                        else:
                            logging.info('Finished chunking!')
                            self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                            self.chunckloading_finished_chunking = True
                    else:
                        logging.info('Finished chunking!')
                        self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                        self.chunckloading_finished_chunking = True
        elif fileToRun.endswith('.hdf5'):
            logging.info('Starting with HDF5 for chunking')
            #Empty dict to keep track of how many pos/neg psfs were found
            self.number_finding_found_polarity = {}

            for polarityVal in polarityValArray:
                logging.info('Starting Batching with polarity: '+polarityVal)

                if not ('ExistingFinding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText() or 'existing Finding' in getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText()):



                    self.chunckloading_finished_chunking = False
                    self.chunckloading_currentLimits = [[0,0],[0,0]]


                    previous_read_hdfChunk = 0
                    events_prev = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})
                    self.chunckloading_number_chuck = 0

                    while self.chunckloading_finished_chunking == False:
                        #Get limits from GUI
                        self.chunckloading_currentLimits = [[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000],[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000-float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000]]

                        # Retrieve all entries within the specified bounding box
                        t_min = self.chunckloading_currentLimits[0][0]+float(self.run_startTLineEdit.text())*1000
                        t_max = min(self.chunckloading_currentLimits[1][1],float(self.run_durationTLineEdit.text())*1000)+float(self.run_startTLineEdit.text())*1000
                        #Function to get events from HDF
                        events,curr_chunk = self.timeSliceFromHDF(fileToRun,requested_start_time_ms = t_min/1000,requested_end_time_ms=t_max/1000,howOftenCheckHdfTime = 500000,loggingBool=False,curr_chunk = previous_read_hdfChunk)

                        #Store this current chunk id for the next chunk
                        previous_read_hdfChunk = curr_chunk-1

                        #Check if any events are still within the range of time we want to assess
                        if len(events) > 0:
                            if (min(events['t']) < (float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text()))*1000):
                                #limit to requested xy
                                events = self.filterEvents_xy(events,xyStretch=(float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text()),float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())))

                                #Add the starting time
                                self.chunckloading_currentLimits = [[float(self.run_startTLineEdit.text())*1000 + x for x in sublist] for sublist in self.chunckloading_currentLimits]
                                #Add events_prev before these events:

                                events = np.concatenate((events_prev,events))

                                if len(events)>0:
                                    logging.info('Current event min/max time:'+str(min(events['t'])/1000)+"/"+str(max(events['t'])/1000))
                                    #Filter on correct polarity
                                    if polarityVal == "Pos":
                                        eventsPol = self.filterEvents_npy_p(events,1)
                                    elif polarityVal == "Neg":
                                        eventsPol = self.filterEvents_npy_p(events,0)
                                    elif polarityVal == "Mix":
                                        eventsPol = events

                                    if (len(eventsPol) > 0):
                                        self.FindingBatching(eventsPol,polarityVal)
                                        self.chunckloading_number_chuck += 1

                                        #Keep the previous 'overlap-events' for next round
                                        events_prev = events[events['t']>self.chunckloading_currentLimits[0][1]-(self.chunckloading_currentLimits[1][1]-self.chunckloading_currentLimits[0][1])]


                            else:
                                logging.info('Finished chunking!')
                                self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                                self.chunckloading_finished_chunking = True
                        else:
                            logging.info('Finished chunking!')
                            self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])
                            self.chunckloading_finished_chunking = True

                else:
                    #We dictate a findingOffset (i.e. nr of positive events if we're running negative events afterwards)
                    findingOffset = 0
                    if polarityVal == 'Neg':
                        findingOffset = self.number_finding_found_polarity['Pos']
                    #And actually run the finding/fitting:
                    self.runFindingAndFitting('',runFitting=False,storeFinding=False,polarityVal=polarityVal,findingOffset = findingOffset,fittingOffset=findingOffset)
                    self.number_finding_found_polarity[polarityVal] = len(self.data['FindingResult'][0])

        else:
            logging.error("Please choose a .raw or .npy or .hdf5 file for previews.")
            return

        #Check if some candidates are found:
        if len(self.data['FindingResult'][0]) == 0:
            logging.error("No candidates found at all! Stopping analysis.")
            return

        #store finding results:
        if self.globalSettings['StoreFindingOutput']['value']:
            self.storeFindingOutput(polarityVal = polarityVal)
        self.currentFileInfo['FindingTime'] = time.time() - self.currentFileInfo['FindingTime']
        logging.info('Number of candidates found: '+str(len(self.data['FindingResult'][0])))
        logging.info('Candidate finding took '+str(self.currentFileInfo['FindingTime'])+' seconds.')

        #All finding is done, continue with fitting:
        if self.dataSelectionPolarityDropdown.currentText() != self.polarityDropdownNames[3]:
            self.runFitting(polarityVal = polarityVal)
        else:
            #Run twice with custom parameters:
            polArr = ['Pos','Neg']
            customFindingEval = ["{key: value for key, value in self.data['FindingResult'][0].items() if key < "+str(self.number_finding_found_polarity['Pos'])+"}","{key-"+str(self.number_finding_found_polarity['Pos']-1)+": value for key, value in self.data['FindingResult'][0].items() if key >= "+str(self.number_finding_found_polarity['Pos'])+"}"]
            self.runFitting(polarityVal = polArr,customFindingEval = customFindingEval,bothPolarities=True)

    def runFindingAndFitting(self,npyData,runFitting=True,storeFinding=True,polarityVal='Mix',findingOffset=0,fittingOffset=0):
        FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings",polarityVal)
        print(FindingEvalText)

        #I want to note that I can send this to a QThread, and it technically works, but the GUI still freezes. Implementaiton here:
        # self.thread.setFindingEvalText(self.getFunctionEvalText('Finding',"self.npyData","self.globalSettings",polarityVal))
        # self.thread.setFittingEvalText(FittingEvalText)
        # self.thread.setrunFitting(runFitting)
        # self.thread.setstoreFinding(storeFinding)
        # self.thread.setpolarityVal(polarityVal)
        # self.thread.setnpyData(npyData)
        # self.thread.setGUIinfo(self)
        # self.thread.setTypeOfThread('Finding')
        # self.thread.run()

        if FindingEvalText is not None:
            #Here we run the finding method:
            try:

                self.currentFileInfo['FindingTime'] = time.time()
                self.data['FindingMethod'] = str(FindingEvalText)
                newFindingResult = eval(str(FindingEvalText))
                if findingOffset == 0:
                    self.data['FindingResult'] = newFindingResult
                else:

                    #Update the finding result entry, their id, but the offset:
                    newFindingResult2 = {key+findingOffset: value for key, value in newFindingResult[0].items()}

                    newFindingResultTot = list(self.data['FindingResult'])
                    self.data['FindingResult'][0].update(newFindingResult2)
                    newFindingResultTot[0] = self.data['FindingResult'][0]
                    newFindingResultTot[1] += "\n"+newFindingResult[1]
                    self.data['FindingResult'] = tuple(newFindingResultTot)


                    #todo: THE string in findingresult[1] is wrong now, wont be fixed.
                    # self.data['FindingResult'][1] += "Test"+newFindingResult[1]

                self.currentFileInfo['FindingTime'] = time.time() - self.currentFileInfo['FindingTime']
                logging.info('Candidate finding completed in '+str(round(self.currentFileInfo['FindingTime'],1))+' seconds; '+str(len(self.data['FindingResult'][0])) + ' candidates found')
                logging.debug(self.data['FindingResult'])
                if storeFinding:
                    if self.globalSettings['StoreFindingOutput']['value']:
                        self.storeFindingOutput(polarityVal = polarityVal)
                #And run the fitting
                if runFitting:
                    self.runFitting(polarityVal,fittingOffset=fittingOffset)
            except Exception as e:
                error_message = f"Critical error in Finding routine! Breaking off!\nError information:\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                self.open_critical_warning(error_message)
                self.data['FindingResult'] = {}
                self.data['FindingResult'][0] = {}
                self.data['FindingResult'][1] = []
        else:
            self.open_critical_warning(f"No Finding evaluation text provided/found")

    def updateGUIafterNewFitting(self):
        logging.info('Candidate fitting completed in '+str(round(self.currentFileInfo['FittingTime'],1))+' seconds; '+str(len(self.data['FittingResult'][0].dropna(axis=0))) + ' localizations found')
        self.data['NrEvents'] = 0
        for candidateID, candidate in self.data['FindingResult'][0].items():
                self.data['NrEvents'] += len(candidate['events'])
        # Update Candidate preview with new finding/fitting results only if found candidates are unchanged
        if self.data['prevFindingMethod']=='None' and self.data['prevNrEvents']==0 and self.data['prevNrCandidates']==0:
            # Finding routine ran for the first time -> don't update preview
            self.data['prevFindingMethod'] = self.data['FindingMethod']
            self.data['prevNrCandidates'] = len(self.data['FindingResult'][0])
            self.data['prevNrEvents'] = self.data['NrEvents']
        elif self.data['prevNrEvents'] == self.data['NrEvents'] and self.data['prevNrCandidates'] == len(self.data['FindingResult'][0]) and self.data['prevFindingMethod'] == self.data['FindingMethod']:
            # same finding routine ran compared to previous run -> Candidates are unchanged
            logging.info("Candidate preview was updated!")
            self.canPreviewtab_widget.show_candidate_callback(self, reset=False)
        else:
            # found candidates have changed -> don't update preview, but update previous run data
            self.data['prevFindingMethod'] = self.data['FindingMethod']
            self.data['prevNrCandidates'] = len(self.data['FindingResult'][0])
            self.data['prevNrEvents'] = self.data['NrEvents']
            self.canPreviewtab_widget.show_candidate_callback(self, reset=True)
        logging.debug(self.data['FittingResult'])
        if len(self.data['FittingResult'][0]) == 0:
            logging.error('No localizations found after fitting!')
            return

    def runFitting(self,polarityVal,customFindingEval = None,bothPolarities=False,fittingOffset=0):
        if self.data['FindingResult'][0] is not None:
            #Run the finding function!
            try:
                if not bothPolarities:
                    if not customFindingEval:
                        FittingEvalText = self.getFunctionEvalText('Fitting',"self.data['FindingResult'][0]","self.globalSettings",polarityVal)
                    if FittingEvalText is not None:
                        self.currentFileInfo['FittingTime'] = time.time()
                        self.data['FittingMethod'] = str(FittingEvalText)
                        fittingResult = eval(str(FittingEvalText))
                        if fittingOffset == 0:
                            self.data['FittingResult'] = fittingResult
                        elif fittingOffset > 0:
                            newFittingResultTot = list(self.data['FittingResult'])
                            newFittingResultTot[0] = pd.concat([newFittingResultTot[0], fittingResult[0]])
                            newFittingResultTot[1] += "\n"+fittingResult[1]
                            self.data['FittingResult'][0].update(pd.concat([self.data['FittingResult'][0], fittingResult[0]]))
                            self.data['FittingResult'] = tuple(newFittingResultTot)
                        self.currentFileInfo['FittingTime'] = time.time() - self.currentFileInfo['FittingTime']
                        #Create and store the localization output
                        if self.globalSettings['StoreFinalOutput']['value']:
                            self.storeLocalizationOutput()
                        #Create and store the metadata
                        if self.globalSettings['StoreFileMetadata']['value']:
                            self.createAndStoreFileMetadata()
                        #Update the GUI
                        self.updateGUIafterNewFitting()
                elif bothPolarities: #If we do pos and neg separately
                    if customFindingEval == None:
                        logging.warning('RunFitting is only possible with custom finding evaluation text for both polarities')
                    else:
                        self.data['FittingMethod'] = ''
                        self.data['FittingResult'] = {}
                        self.data['FittingResult'][0] = pd.DataFrame()
                        self.data['FittingResult'][1] = ''
                        self.currentFileInfo['FittingTime'] = time.time()
                        #We're assuming that customFindingEval has 2 entries: one for pos, one for neg.
                        for i in range(0,2):
                            FittingEvalText = self.getFunctionEvalText('Fitting',customFindingEval[i],"self.globalSettings",polarityVal[i])
                            if FittingEvalText is not None:
                                currentPolFitting = eval(str(FittingEvalText))
                                #Correct the candidate id in the second go-around
                                if i > 0:
                                    currentPolFitting[0]['candidate_id'] += max(self.data['FittingResult'][0]['candidate_id'])
                                #And add to the current fitting data
                                self.data['FittingMethod'] += str(FittingEvalText) +"\n"
                                self.data['FittingResult'][0] = pd.concat([self.data['FittingResult'][0],currentPolFitting[0]], ignore_index=True)
                                self.data['FittingResult'][1] += str(currentPolFitting[1])+'\n'

                        #After both pos and neg fit is completed:
                        self.currentFileInfo['FittingTime'] = time.time() - self.currentFileInfo['FittingTime']
                        #Create and store the localization output
                        if self.globalSettings['StoreFinalOutput']['value']:
                            self.storeLocalizationOutput()
                        #Create and store the metadata
                        if self.globalSettings['StoreFileMetadata']['value']:
                            self.createAndStoreFileMetadata()
                        #Update the GUI
                        self.updateGUIafterNewFitting()
            except Exception as e:
                error_message = f"Critical error in Fitting routine! Breaking off!\nError information:\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                self.open_critical_warning(error_message)
                self.data['FittingResult'] = {}
                self.data['FittingResult'][0] = {}
                self.data['FittingResult'][1] = [{}]
        else:
            self.open_critical_warning(f"No Fitting evaluation text provided/found")

    def getStoreLocationPartial(self):
        if 'ExistingFinding' in self.data['FindingMethod']:
            FindingResultFileName = self.data['FindingMethod'][self.data['FindingMethod'].index('File_Location="')+len('File_Location="'):self.data['FindingMethod'].index('")')]
            storeLocationPartial = FindingResultFileName[:-7]
        else:
            storeLocationPartial = self.currentFileInfo['CurrentFileLoc'][:-4]
        return storeLocationPartial

    def storeLocalization(self,storeLocation,localizations,outputType='thunderstorm'):
        if outputType == 'minimal':
            localizations.to_csv(storeLocation)
        elif outputType == 'thunderstorm':
            #Add a frame column to fittingResult:
            localizations['frame'] = localizations['t'].apply(round).astype(int)
            localizations['frame'] -= min(localizations['frame'])-1
            #Create thunderstorm headers
            headers = list(localizations.columns)
            headers = ['\"x [nm]\"' if header == 'x' else '\"y [nm]\"' if header == 'y' else '\"z [nm]\"' if header == 'z' else '\"t [ms]\"' if header == 't' else header for header in headers]
            localizations.rename_axis('\"id\"').to_csv(storeLocation, header=headers, quoting=csv.QUOTE_NONE)
        else:
            #default to minimal
            localizations.to_csv(storeLocation)

    def storeLocalizationOutput(self):
        logging.debug('Attempting to store fitting results output')
        storeLocation = self.getStoreLocationPartial()+'_FitResults_'+self.storeNameDateTime+'.csv'
        #Store the localization output
        localizations = self.data['FittingResult'][0].dropna(axis=0, ignore_index=True)
        localizations = localizations.drop('fit_info', axis=1)

        #Actually store
        self.storeLocalization(storeLocation,localizations,outputType=self.globalSettings['OutputDataFormat']['value'])

        #Also store pickle information:
        #Also save pos and neg seperately if so useful:
        if self.globalSettings['StoreFittingOutput']['value']:
            if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
                try:
                    if self.number_finding_found_polarity['Pos'] > 0 and self.number_finding_found_polarity['Neg'] > 0:
                        allPosFittingResults = self.data['FittingResult'][0][0:self.number_finding_found_polarity['Pos']]
                        allNegFittingResults = self.data['FittingResult'][0][self.number_finding_found_polarity['Pos']:]

                        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FittingResults_PosOnly_'+self.storeNameDateTime+'.pickle'
                        with open(file_path, 'wb') as file:
                            pickle.dump(self.data['refFindingFilepos'], file)
                            pickle.dump(allPosFittingResults, file)


                        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FittingResults_NegOnly_'+self.storeNameDateTime+'.pickle'
                        with open(file_path, 'wb') as file:
                            pickle.dump(self.data['refFindingFileneg'], file)
                            pickle.dump(allNegFittingResults, file)

                        #And store all of them
                        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FittingResults_'+self.storeNameDateTime+'.pickle'
                        with open(file_path, 'wb') as file:
                            pickle.dump(self.data['refFindingFile'], file)
                            pickle.dump(self.data['FittingResult'][0], file)
                except:
                    logging.debug('This can be safely ignored')
            else:#Only a single pos/neg selected
                file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FittingResults_'+self.storeNameDateTime+'.pickle'
                with open(file_path, 'wb') as file:
                    pickle.dump(self.data['refFindingFile'], file)
                    pickle.dump(self.data['FittingResult'][0], file)
            logging.info('Fitting results output stored')
        else:
            pass

    def storeFindingOutput(self,polarityVal='Pos'):
        logging.debug('Attempting to store finding results output')
        #Store the Finding results output
        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_'+self.storeNameDateTime+'.pickle'
        self.data['refFindingFile'] = file_path
        with open(file_path, 'wb') as file:
            pickle.dump(self.data['FindingResult'][0], file)


        #Also save pos and neg seperately if so useful:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            try:
                if self.number_finding_found_polarity['Pos'] > 0 and self.number_finding_found_polarity['Neg'] > 0:
                    allPosFindingResults = {key: value for key, value in self.data['FindingResult'][0].items() if key < self.number_finding_found_polarity['Pos']}
                    allNegFindingResults = {key-self.number_finding_found_polarity['Pos']: value for key, value in self.data['FindingResult'][0].items() if key >= self.number_finding_found_polarity['Pos']}

                    file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_PosOnly_'+self.storeNameDateTime+'.pickle'
                    self.data['refFindingFilepos'] = file_path
                    with open(file_path, 'wb') as file:
                        pickle.dump(allPosFindingResults, file)


                    file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_NegOnly_'+self.storeNameDateTime+'.pickle'
                    self.data['refFindingFileneg'] = file_path
                    with open(file_path, 'wb') as file:
                        pickle.dump(allNegFindingResults, file)
            except:
                logging.debug('This can be safely ignored')

        logging.info('Finding results output stored')

    def createAndStoreFileMetadata(self):
        logging.debug('Attempting to create and store file metadata')
        try:
            metadatastring = dedent(f"""\
            Metadata information for file {self.currentFileInfo['CurrentFileLoc']}
            Analysis routine finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            ---- Finding metadata output: ----
            Methodology used:
            {self.data['FindingMethod']}

            Number of candidates found: {len(self.data['FindingResult'][0])}
            Candidate finding took {self.currentFileInfo['FindingTime']} seconds.

            Custom output from finding function:\n""")\
            + f"""{self.data['FindingResult'][1]}\n""" + dedent(f"""\

            ---- Fitting metadata output: ----
            Methodology used:
            {self.data['FittingMethod']}

            Number of localizations found: {len(self.data['FittingResult'][0].dropna(axis=0))}
            Candidate fitting took {self.currentFileInfo['FittingTime']} seconds.

            Custom output from fitting function:\n""")\
            + f"""{self.data['FittingResult'][1]}
            """
            #Store this metadatastring:
            with open(self.getStoreLocationPartial()+'_RunInfo_'+self.storeNameDateTime+'.txt', 'w') as f:
                f.write(metadatastring)
            logging.info('File metadata created and stored')
        except:
            logging.error('Error in creating file metadata, not stored')

    def getFunctionEvalText(self,className,p1,p2,polarity):
        #Get the dropdown info
        moduleMethodEvalTexts = []
        all_layouts = self.findChild(QWidget, "groupbox"+className+polarity).findChildren(QLayout)[0]


        methodKwargNames_method = []
        methodKwargValues_method = []
        methodName_method = ''
        # Iterate over the items in the layout
        for index in range(all_layouts.count()):
            item = all_layouts.itemAt(index)
            widget = item.widget()
            if widget is not None:#Catching layouts rather than widgets....
                if ("LineEdit" in widget.objectName()) and widget.isVisibleTo(self.tab_processing):
                    # The objectName will be along the lines of foo#bar#str
                    #Check if the objectname is part of a method or part of a scoring
                    split_list = widget.objectName().split('#')
                    methodName_method = split_list[1]
                    methodKwargNames_method.append(split_list[2])

                    #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                    methodKwargValues_method.append(widget.text().replace('\\','/'))

                # add distKwarg choice to Kwargs if given
                if ("ComboBox" in widget.objectName()) and widget.isVisibleTo(self.tab_processing) and 'dist_kwarg' in widget.objectName():
                    methodKwargNames_method.append('dist_kwarg')
                    methodKwargValues_method.append(widget.currentText())
            else:
                #If the item is a layout instead...
                if isinstance(item, QLayout):
                    for index2 in range(item.count()):
                        item_sub = item.itemAt(index2)
                        widget_sub = item_sub.widget()
                        if ("LineEdit" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.tab_processing):
                            # The objectName will be along the lines of foo#bar#str
                            #Check if the objectname is part of a method or part of a scoring
                            split_list = widget_sub.objectName().split('#')
                            methodName_method = split_list[1]
                            methodKwargNames_method.append(split_list[2])

                            #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                            methodKwargValues_method.append(widget_sub.text().replace('\\','/'))

                        # add distKwarg choice to Kwargs if given
                        if ("ComboBox" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.tab_processing) and 'dist_kwarg' in widget_sub.objectName():
                            methodKwargNames_method.append('dist_kwarg')
                            methodKwargValues_method.append(widget_sub.currentText())

        #If at this point there is no methodName_method, it means that the method has exactly 0 req or opt kwargs. Thus, we simply find the value of the QComboBox which should be the methodName:
        if methodName_method == '':
            for index in range(all_layouts.count()):
                item = all_layouts.itemAt(index)
                widget = item.widget()
                if isinstance(widget,QComboBox) and widget.isVisibleTo(self.tab_processing) and className in widget.objectName():
                    if className == 'Finding':
                        methodName_method = utils.functionNameFromDisplayName(widget.currentText(),getattr(self,f"Finding_functionNameToDisplayNameMapping{polarity}"))
                    elif className == 'Fitting':
                        methodName_method = utils.functionNameFromDisplayName(widget.currentText(),getattr(self,f"Fitting_functionNameToDisplayNameMapping{polarity}"))

        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = utils.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method,partialStringStart=str(p1)+','+str(p2))
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)

        if moduleMethodEvalTexts is not None and len(moduleMethodEvalTexts) > 0:
            return moduleMethodEvalTexts[0]
        else:
            return None

    def setFilepathExisting(self, className, polarity, newFilepath):
        #Get the dropdown info
        all_layouts = self.findChild(QWidget, "groupbox"+className+polarity).findChildren(QLayout)[0]
        # Iterate over the items in the layout
        for index in range(all_layouts.count()):
            item = all_layouts.itemAt(index)
            if isinstance(item, QLayout):
                for index2 in range(item.count()):
                    item_sub = item.itemAt(index2)
                    widget_sub = item_sub.widget()
                    if ("LineEdit" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.tab_processing):
                        widget_sub.setText(newFilepath)

    def setMethod(self, className, polarity, newMethod):
        #Get the dropdown info
        all_layouts = self.findChild(QWidget, "groupbox"+className+polarity).findChildren(QLayout)[0]
        # get dropdown item
        dropdown = all_layouts.itemAt(0)
        widget = dropdown.widget()
        widget.setCurrentText(f"{newMethod} ({polarity.lower()})")
        print(widget)

    def save_entries_to_json(self):
        self.entries = {}
        # Iterate over all editable fields and store their values in the entries dictionary
        for field_name, field_widget in self.get_editable_fields().items():
            # only store editable fields that have a name
            if not field_name=="":
                if isinstance(field_widget, QLineEdit):
                    self.entries[field_name] = field_widget.text()
                if isinstance(field_widget, QCheckBox):
                    self.entries[field_name] = field_widget.isChecked()
                elif isinstance(field_widget, QComboBox):
                    self.entries[field_name] = field_widget.currentText()

        # Specify the path and filename for the JSON file
        json_file_path = self.globalSettings['JSONGUIstorePath']['value']

        # Write the entries dictionary to the JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(self.entries, json_file)

    def load_entries_from_json(self):
        #First set all comboboxes
        self.load_entries_from_json_single(runParams=['QComboBox'])
        #Then set all line edits
        self.load_entries_from_json_single(runParams=['QLineEdit'])
        #Then set all checkboxes
        self.load_entries_from_json_single(runParams=['QCheckBox'])

    def load_entries_from_json_single(self,runParams=['QLineEdit','QComboBox','QCheckBox']):
        # Specify the path and filename for the JSON file
        json_file_path = self.globalSettings['JSONGUIstorePath']['value']

        try:
            # Load the entries from the JSON file
            with open(json_file_path, "r") as json_file:
                self.entries = json.load(json_file)

            for polVal in ["Pos","Neg","Mix"]:
                # Set the values of the editable fields from the loaded entries
                for field_name, field_widget in self.get_editable_fields().items():
                    if field_name in self.entries:
                        if isinstance(field_widget, QLineEdit):
                            if 'QLineEdit' in runParams:
                                field_widget.setText(self.entries[field_name])
                                logging.debug('Set text of field_widget '+field_name+' to '+self.entries[field_name])
                        elif isinstance(field_widget, QCheckBox):
                            if 'QCheckBox' in runParams:
                                field_widget.setChecked(self.entries[field_name])
                                logging.debug('Set text of field_widget '+field_name+' to '+str(self.entries[field_name]))
                        elif isinstance(field_widget, QComboBox):
                            if 'QComboBox' in runParams:
                                index = field_widget.findText(self.entries[field_name])
                                if index >= 0:
                                    logging.debug('Set text of field_widget '+field_name+' to '+self.entries[field_name])
                                    field_widget.setCurrentIndex(index)
                                    #Also change the lineedits and such:
                                    if 'Finding' in field_widget.objectName():
                                        polVal2 = field_widget.objectName()[-3:]
                                        utils.changeLayout_choice(getattr(self, f"groupboxFinding{polVal2}").layout(),field_widget.objectName(),getattr(self, f"Finding_functionNameToDisplayNameMapping{polVal2}"),parent=self)
                                    elif 'Fitting' in field_widget.objectName():
                                        polVal2 = field_widget.objectName()[-3:]
                                        if 'dist_kwarg' in field_widget.objectName():
                                            field_widget.setCurrentText(self.entries[field_name])
                                        else:
                                            utils.changeLayout_choice(getattr(self, f"groupboxFitting{polVal2}").layout(),field_widget.objectName(),getattr(self, f"Fitting_functionNameToDisplayNameMapping{polVal2}"),parent=self)


        except FileNotFoundError:
            # Handle the case when the JSON file doesn't exist yet
            self.save_entries_to_json()
            logging.info('No GUI settings storage found, new one created.')
            pass

    def get_editable_fields(self):
        fields = {}

        def find_editable_fields(widget):
            if isinstance(widget, QLineEdit) or isinstance(widget, QComboBox) or isinstance(widget, QCheckBox):
                fields[widget.objectName()] = widget
            elif isinstance(widget, QWidget):
                # for child_widget in widget.children():
                #     find_editable_fields(child_widget)
                for children_widget in widget.findChildren(QWidget):
                    find_editable_fields(children_widget)

        find_editable_fields(self)
        return fields


    def reset_single_combobox_states(self):
        logging.info('Ran reset_single_combobox_states')
        original_states = {}
        # try:
        def set_single_combobox_states(widget,polVal):
            if isinstance(widget, QComboBox):
                if widget.objectName()[-3:] == polVal:
                    original_states[widget] = widget.currentIndex()
                    #Update all line edits and such
                    if 'CandidateFinding' in widget.objectName():
                        #logging.info(getattr(self, f"groupboxFinding{polVal}").objectName())
                        utils.changeLayout_choice(getattr(self, f"groupboxFinding{polVal}").layout(),widget.objectName(),getattr(self, f"Finding_functionNameToDisplayNameMapping{polVal}"),parent=self)
                    elif 'CandidateFitting' in widget.objectName():
                        #logging.info(getattr(self, f"groupboxFitting{polVal}").objectName())
                        utils.changeLayout_choice(getattr(self, f"groupboxFitting{polVal}").layout(),widget.objectName(),getattr(self, f"Fitting_functionNameToDisplayNameMapping{polVal}"),parent=self)
            elif isinstance(widget, QWidget):
                for child_widget in widget.children():
                    set_single_combobox_states(child_widget,polVal)

        # Reset to orig states
        for polVal in ['Pos','Neg','Mix']:
            set_single_combobox_states(self,polVal)
            for combobox, original_state in original_states.items():
                combobox.setCurrentIndex(original_state)
                if 'Finding' in combobox.objectName():
                    logging.debug(combobox.objectName())
                    utils.changeLayout_choice(getattr(self, f"groupboxFinding{polVal}").layout(),combobox.objectName(),getattr(self, f"Finding_functionNameToDisplayNameMapping{polVal}"), parent=self)
                elif 'Fitting' in combobox.objectName():
                    logging.debug(combobox.objectName())
                    utils.changeLayout_choice(getattr(self, f"groupboxFitting{polVal}").layout(),combobox.objectName(),getattr(self, f"Fitting_functionNameToDisplayNameMapping{polVal}"),parent=self)
        # except:
        #     pass


    def set_all_combobox_states(self):
        logging.info('Ran set_all_combobox_states')
        original_states = {}
        # try:
        def set_combobox_states(widget,polVal):
            if isinstance(widget, QComboBox):
                if widget.objectName()[-3:] == polVal:
                    original_states[widget] = widget.currentIndex()
                    for i in range(widget.count()):
                        # logging.info('Set text of combobox '+widget.objectName()+' to '+widget.itemText(i))
                        widget.setCurrentIndex(i)
                        #Update all line edits and such
                        if 'CandidateFinding' in widget.objectName():
                            #logging.info(getattr(self, f"groupboxFinding{polVal}").objectName())
                            utils.changeLayout_choice(getattr(self, f"groupboxFinding{polVal}").layout(),widget.objectName(),getattr(self, f"Finding_functionNameToDisplayNameMapping{polVal}"),parent=self)
                        elif 'CandidateFitting' in widget.objectName():
                            #logging.info(getattr(self, f"groupboxFitting{polVal}").objectName())
                            utils.changeLayout_choice(getattr(self, f"groupboxFitting{polVal}").layout(),widget.objectName(),getattr(self, f"Fitting_functionNameToDisplayNameMapping{polVal}"),parent=self)
            elif isinstance(widget, QWidget):
                for child_widget in widget.children():
                    set_combobox_states(child_widget,polVal)

        # Reset to orig states
        for polVal in ['Pos','Neg','Mix']:
            set_combobox_states(self,polVal)
            for combobox, original_state in original_states.items():
                combobox.setCurrentIndex(original_state)
                if 'Finding' in combobox.objectName():
                    logging.debug(combobox.objectName())
                    utils.changeLayout_choice(getattr(self, f"groupboxFinding{polVal}").layout(),combobox.objectName(),getattr(self, f"Finding_functionNameToDisplayNameMapping{polVal}"), parent=self)
                elif 'Fitting' in combobox.objectName():
                    logging.debug(combobox.objectName())
                    utils.changeLayout_choice(getattr(self, f"groupboxFitting{polVal}").layout(),combobox.objectName(),getattr(self, f"Fitting_functionNameToDisplayNameMapping{polVal}"),parent=self)
        # except:
        #     pass

class AdvancedSettingsWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Advanced settings")
        self.resize(300, 200)

        self.parent = parent
        # Set the window icon to the parent's icon
        self.setWindowIcon(self.parent.windowIcon())

        #Create a QGridlayout to populate:
        layout = QGridLayout()
        currentRow = 0

        self.labelGlobSettings={}
        self.checkboxGlobSettings = {}
        self.input_fieldGlobSettings = {}
        self.dropdownGlobSettings = {}
        self.dropdownGlobSettingsOptions={}

        # Iterate over the settings in self.parent.globalSettings
        for setting, value in self.parent.globalSettings.items():
            # Check if the setting is not in IgnoreInOptions
            if setting not in self.parent.globalSettings.get('IgnoreInOptions'):
                # Create a label with the setting title
                if 'displayName' in value:
                    self.labelGlobSettings[setting] = QLabel(value['displayName'])
                else:
                    self.labelGlobSettings[setting] = QLabel(setting)
                #Align correctly
                self.labelGlobSettings[setting].setAlignment(Qt.AlignVCenter | Qt.AlignRight)
                layout.addWidget(self.labelGlobSettings[setting],currentRow,0)

                # Check the type of the input
                input_type = value['input']

                # Create a checkbox for boolean values
                if input_type == bool:
                    self.checkboxGlobSettings[setting] = QCheckBox()
                    self.checkboxGlobSettings[setting].setChecked(value['value'])
                    self.checkboxGlobSettings[setting].stateChanged.connect(lambda state, s=setting: self.update_global_settings(s, state))
                    layout.addWidget(self.checkboxGlobSettings[setting],currentRow,1)

                # Create an input field for string or integer values
                elif input_type in (str, float):
                    self.input_fieldGlobSettings[setting] = QLineEdit()
                    self.input_fieldGlobSettings[setting].setText(str(value['value']))
                    self.input_fieldGlobSettings[setting].textChanged.connect(lambda text, s=setting: self.update_global_settings(s, text))
                    layout.addWidget(self.input_fieldGlobSettings[setting],currentRow,1)

                # Create a dropdown for choices
                elif input_type == 'choice':
                    self.dropdownGlobSettings[setting] = QComboBox()
                    self.dropdownGlobSettingsOptions[setting] = value['options']
                    self.dropdownGlobSettings[setting].addItems(self.dropdownGlobSettingsOptions[setting])
                    current_index = self.dropdownGlobSettingsOptions[setting].index(value['value'])
                    self.dropdownGlobSettings[setting].setCurrentIndex(current_index)
                    self.dropdownGlobSettings[setting].currentIndexChanged.connect(lambda index, s=setting: self.update_global_settings(s, self.dropdownGlobSettingsOptions[s][index]))
                    layout.addWidget(self.dropdownGlobSettings[setting],currentRow,1)

                #Increment the row for the next one
                currentRow+=1

        #Load the global settings from last time:
        self.load_global_settings()

        # Create a save button
        save_button = QPushButton("Save Global Settings")
        save_button.clicked.connect(self.save_global_settings)
        layout.addWidget(save_button,currentRow,0,1,2)

        # Create a load button
        load_button = QPushButton("Load Global Settings")
        load_button.clicked.connect(self.load_global_settings)
        layout.addWidget(load_button,currentRow+1,0,1,2)

        #Add a full-reset button:
        full_reset_button = QPushButton("Fully reset GUI and global settings")
        full_reset_button.clicked.connect(self.confirm_full_reset_GUI_GlobSettings)
        layout.addWidget(full_reset_button,currentRow+2,0,1,2)

        # Create a widget and set the layout
        widget = QWidget()
        widget.setLayout(layout)

        # Set the central widget of the main window
        self.setCentralWidget(widget)

    def show(self):
        super().show()
        cursor_pos = QCursor.pos()
        # Set the position of the window
        self.move(QPoint(cursor_pos.x()-self.width()/2,cursor_pos.y()))

    def updateGlobSettingsGUIValues(self):
        # Iterate over the settings in self.parent.globalSettings
        for setting, value in self.parent.globalSettings.items():
            # Check if the setting is not in IgnoreInOptions
            if setting not in self.parent.globalSettings.get('IgnoreInOptions'):
                # Create a label with the setting title
                if 'displayName' in value:
                    self.labelGlobSettings[setting] = QLabel(value['displayName'])
                else:
                    self.labelGlobSettings[setting] = QLabel(setting)

                # Check the type of the input
                input_type = value['input']

                # Create a checkbox for boolean values
                if input_type == bool:
                    self.checkboxGlobSettings[setting].setChecked(value['value'])

                # Create an input field for string or integer values
                elif input_type in (str, float):
                    self.input_fieldGlobSettings[setting].setText(str(value['value']))

                # Create a dropdown for choices
                elif input_type == 'choice':
                    options = value['options']
                    current_index = options.index(value['value'])
                    self.dropdownGlobSettings[setting].setCurrentIndex(current_index)

    def confirm_full_reset_GUI_GlobSettings(self):
        reply = QMessageBox.question(self, 'Confirmation', "Are you sure you want to fully reset the GUI and global settings? This will close the GUI and you have to re-open.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.full_reset_GUI_GlobSettings()

    def full_reset_GUI_GlobSettings(self):
        #Remove the JSON files:
        os.remove(self.parent.globalSettings['JSONGUIstorePath']['value'])
        os.remove(self.parent.globalSettings['GlobalOptionsStorePath']['value'])
        #Restart the GUI:
        QCoreApplication.quit()

    def update_global_settings(self, setting, value):
        self.parent.globalSettings[setting]['value'] = value

    def save_global_settings(self):
        # Specify the path and filename for the JSON file
        json_file_path = self.parent.globalSettings['GlobalOptionsStorePath']['value']

        # Serialize the globalSettings dictionary, skipping over the 'input' value
        serialized_settings = {}
        for key, value in self.parent.globalSettings.items():
            if key != 'IgnoreInOptions':
                serialized_settings[key] = value.copy()
                serialized_settings[key].pop('input', None)

        # Create the parent folder(s) for the JSON file
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

        # Write the globalSettings dictionary to the JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(serialized_settings, json_file)

        #Close after saving
        logging.info('Global settings saved!')
        self.close()

    def load_global_settings(self):
        # Specify the path and filename for the JSON file
        json_file_path = self.parent.globalSettings['GlobalOptionsStorePath']['value']
        try:    # Load the globalSettings dictionary from the JSON file
            with open(json_file_path, "r") as json_file:
                loaded_settings = json.load(json_file)

            # Update the 'input' value in self.parent.globalSettings, if it exists
            for key, value in loaded_settings.items():
                if key != 'IgnoreInOptions':
                    if key in self.parent.globalSettings:
                        value['input'] = self.parent.globalSettings[key].get('input', None)

            self.parent.globalSettings.update(loaded_settings)
            #Update the values of the globalSettings GUI
            self.updateGlobSettingsGUIValues()
        except:
            self.save_global_settings()
            logging.info('No global settings storage found, new one created.')

class ImageSlider(QWidget):
    def __init__(self, figures=None, parent=None):
        super().__init__()

        self.parent = parent

        if figures is None:
            figures = []

        self.figures = figures
        self.current_figure_idx = 0
        self.canvas = FigureCanvas(self.figures[0]) if len(self.figures) > 0 else FigureCanvas()

        self.toolbar = NavigationToolbar(self.canvas, self)


        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(figures) - 1)
        self.slider.setTickInterval = 1
        self.slider.setTickPosition = QSlider.TicksBelow
        self.slider.setSingleStep = 1
        self.slider.valueChanged.connect(self.update_figure)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_figure)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_figure)


        self.llayout = QVBoxLayout()

        self.CanvasLayout = QVBoxLayout()
        self.CanvasLayout.addWidget(self.toolbar)
        self.CanvasLayout.addWidget(self.canvas)

        self.sliderlayout = QHBoxLayout()

        self.sliderlayout.addWidget(self.slider)
        self.sliderlayout.addWidget(self.prev_button)
        self.sliderlayout.addWidget(self.next_button)


        self.llayout.addLayout(self.CanvasLayout)
        self.llayout.addLayout(self.sliderlayout)


        self.setLayout(self.llayout)

    def update_toolbar(self):
        print('Update toolbar ran')
        self.toolbar.update()

    def update_figure(self, index):
        if index > -1 and index < len(self.figures) and self.figures[index] is not None:
            #Before updating, set the current canvas (i.e. the 'old' figure) back to home toolbar settinsg:
            self.toolbar._actions['home'].trigger()

            self.current_figure_idx = index
            self.canvas.figure = self.figures[index]

            if self.parent is not None:
                try:
                    # Calculate colorbar limits
                    cmin = self.parent.PreviewMinCbarVal
                    cmax = self.parent.PreviewMaxCbarVal
                    self.canvas.figure.axes[0].images[0].set_clim(cmin, cmax)
                    # Set colorbar to gray
                    self.canvas.figure.axes[0].images[0].set_cmap('gray')
                except:
                    logging.error('Error in setting cbar limits preview')
                    pass

            # Create a new canvas and toolbar for each figure
            self.canvas = FigureCanvas(self.figures[index]) if len(self.figures) > 0 else FigureCanvas()
            self.toolbar = NavigationToolbar(self.canvas, self)

            layout = self.CanvasLayout
            if layout is not None:
                # Remove the old canvas and toolbar from the layout
                for i in reversed(range(layout.count())):
                    item = layout.itemAt(i)
                    if isinstance(item.widget(), (FigureCanvas, NavigationToolbar)):
                        layout.removeWidget(item.widget())

                # Add the new canvas and toolbar to the layout
                layout.addWidget(self.toolbar)
                layout.addWidget(self.canvas)

            # self.update_toolbar()

            self.canvas.draw()

    def previous_figure(self):
        self.update_figure(max(self.current_figure_idx - 1, 0))
        self.slider.setValue(self.current_figure_idx)

    def next_figure(self):
        self.update_figure(min(self.current_figure_idx + 1, len(self.figures) - 1))
        self.slider.setValue(self.current_figure_idx)

    def update_figures(self, new_figures):
        self.figures = new_figures
        self.slider.setRange(0, len(new_figures) - 1)
        self.update_figure(0)

#Pressing clickable group boxes collapses them
class ClickableGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        # self.setCheckable(True)
        # self.setChecked(True)  # Expanded by default

        self.clicked.connect(self.toggleCollapsed)
        # Remove the checkbox indicator
        self.setStyleSheet("QGroupBox::indicator { width: 0; }")

    def toggleCollapsed(self, checked):
        layout = self.layout()
        if layout is not None:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setVisible(checked)
            layout.invalidate()
            self.adjustSize()
            self.xPaddingQLabel.setStyleSheet("color: #808080;")
            self.yPaddingQLabel.setStyleSheet("color: #808080;")
            self.tPaddingQLabel.setStyleSheet("color: #808080;")
            self.xPaddingLineEdit.setStyleSheet("color: #808080; background-color: #ECECEC;")
            self.yPaddingLineEdit.setStyleSheet("color: #808080; background-color: #ECECEC;")
            self.tPaddingLineEdit.setStyleSheet("color: #808080; background-color: #ECECEC;")
            self.xPaddingLineEdit.setReadOnly(True)
            self.yPaddingLineEdit.setReadOnly(True)
            self.tPaddingLineEdit.setReadOnly(True)

    def handlePlotOptionCheckboxChange(self, state):
        sender = self.sender()
        if sender.isChecked():
            if len(self.currentSelection) < 2:
                self.currentSelection.append(sender)
            else:
                self.currentSelection[0].setChecked(False)
                self.currentSelection.append(sender)
        else:
            if sender in self.currentSelection:
                self.currentSelection.remove(sender)

    def getSurroundingAndPaddingValues(self):
        surrounding_checked = self.showSurroundingCheckBox.isChecked()
        x_padding_value = int(self.xPaddingLineEdit.text())
        y_padding_value = int(self.yPaddingLineEdit.text())
        t_padding_value = int(self.tPaddingLineEdit.text())
        return surrounding_checked, x_padding_value, y_padding_value, t_padding_value

    def getCheckedPlotOptionClasses(self):
        checked_plot_classes = []
        for checkbox in self.plotOptionCheckboxes:
            if checkbox.isChecked():
                option_id = int(checkbox.objectName().split("_")[1])
                checked_plot_classes.append(self.plotOptions[option_id]["plotclass"])
        return checked_plot_classes

class CriticalWarningWindow(QMainWindow):
    def __init__(self, parent, text):
        super().__init__(parent)
        self.setWindowTitle("Eve - Critical warning!")
        self.resize(300, 100)

        # Add a warning text and an OK button:
        self.warning_text = QLabel(text)

        # Add the label to the window's layout:
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.warning_text)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.close)

        # Add the OK button to the layout:
        self.layout.addWidget(self.ok_button)

        self.parent = parent
        # Set the window icon to the parent's icon
        self.setWindowIcon(self.parent.windowIcon())

        # Set the main layout
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def show(self):
        super().show()
        # Set the position of the window
        cursor_pos = QCursor.pos()
        screen_geometry = QApplication.desktop().screenGeometry()
        window_geometry = self.frameGeometry()
        x = cursor_pos.x() - window_geometry.width() / 2
        y = cursor_pos.y()
        if x < screen_geometry.left():
            x = screen_geometry.left()
        elif x + window_geometry.width() > screen_geometry.right():
            x = screen_geometry.right() - window_geometry.width()
        if y + window_geometry.height() > screen_geometry.bottom():
            y = screen_geometry.bottom() - window_geometry.height()
        self.move(QPoint(x, y))

class VisualisationNapari(QWidget):
    """
    Class that visualises the data in napari in e.g. scatter/average shifted histogram
    General idea: have a single implementation similar to CandidateFinding/Fitting, where user-created visualisations can be created. They have a list of localizations as input, and output an image.
    This class handles both the showcasing/choosing of visualisations and the napari showing of the output images.
    """
    def __init__(self,parent):
        super().__init__()
        # Create a layout for the main widget
        self.mainlayout = QVBoxLayout()
        # Set the layout for the main widget
        self.setLayout(self.mainlayout)


        #------------Start of GUI dynamic layout -----------------

        #Create a groupbox for visualisation methods
        self.VisualisationGroupbox = QGroupBox("Visualisation")
        self.VisualisationGroupbox.setLayout(QGridLayout())
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        self.VisualisationGroupbox.setObjectName("VisualiseGroupboxKEEP")

        #Add a combobox-dropdown
        visualisationDropdown = QComboBox(self)
        visualisationDropdown.setMaxVisibleItems(30)
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        visualisationDropdown_name = "VisualisationDropdownKEEP"
        Visualisation_functionNameToDisplayNameMapping_name = f"Visualisation_functionNameToDisplayNameMapping"
        #Get the options dynamically from the Visualisation folder
        options = utils.functionNamesFromDir('Visualisation')
        #Also find the mapping of 'display name' to 'function name'
        displaynames, Visualisation_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options,'')
        #and add this to our self attributes
        setattr(self, Visualisation_functionNameToDisplayNameMapping_name, Visualisation_functionNameToDisplayNameMapping)

        #Set a current name and add the items
        visualisationDropdown.setObjectName(visualisationDropdown_name)
        visualisationDropdown.addItems(displaynames)

        #Add a callback to the changing of the dropdown:
        visualisationDropdown.currentTextChanged.connect(lambda text: utils.changeLayout_choice(self.VisualisationGroupbox.layout(),visualisationDropdown_name,getattr(self, Visualisation_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True))

        #Add the visualisationDropdown to the layout
        #Ensure it is full-width with the [1,6]
        self.VisualisationGroupbox.layout().addWidget(visualisationDropdown,1,0,1,6)

        #On startup/initiatlisation: also do changeLayout_choice
        utils.changeLayout_choice(self.VisualisationGroupbox.layout(),visualisationDropdown_name,getattr(self, Visualisation_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True)

        #add a 'Visualise!' button to this groupbox at the bottom:
        button = QPushButton("Visualise!", self)
        #Ensure 'KEEP' is in the objectname so it's never removed
        button.setObjectName("VisualiseRunButtonKEEP")
        #Ensure it is full-width with the [1,6]
        self.VisualisationGroupbox.layout().addWidget(button,99,0,1,6)
        #And add a callback to this:
        button.clicked.connect(lambda text, parent=parent: self.visualise_callback(parent))

        #Add the groupbox to the mainlayout
        self.mainlayout.layout().addWidget(self.VisualisationGroupbox)

        #------------End of GUI dynamic layout -----------------


        # Create a napari viewer
        self.napariviewer = Viewer(show=False)
        #Add a napariViewer to the layout
        self.viewer = QtViewer(self.napariviewer)
        self.mainlayout.addWidget(self.viewer)
        self.mainlayout.addWidget(self.viewer.controls)

        logging.info('VisualisationNapari init')

    def visualise_callback(self,parent):
        #Visuliation method from https://www.frontiersin.org/articles/10.3389/fbinf.2021.817254/full
        logging.info('Visualise button pressed')

        #Get the current function callback
        FunctionEvalText = self.getVisFunctionEvalText("parent.data['FittingResult'][0]","parent.globalSettings")

        resultImage = eval(FunctionEvalText)

        #Clear all existing layers
        for layer in reversed(self.napariviewer.layers):
            self.napariviewer.layers.remove(layer)

        #Add a new layer which is this image
        #Dynamically set contrast limits based on percentile to get proper visualisation and not be affected by outliers too much
        percentile_value_display = 0.01
        contrast_limits = np.percentile(resultImage[0], [percentile_value_display,100-percentile_value_display])
        #Ensure that contrast_limits maximum value is always higher than the minimum
        if contrast_limits[1]<=contrast_limits[0]:
            contrast_limits[1]=contrast_limits[0]+1

        #Quick check that the scale value is sensible
        if type(resultImage[1]) == float or type(resultImage[1]) == int or type(resultImage[1]) == np.float64:
            scaleValue = resultImage[1]
        else:
            scaleValue = -1

        self.napariviewer.add_image(resultImage[0], multiscale=False,contrast_limits=contrast_limits,scale = [scaleValue,scaleValue])

        if scaleValue > -1:
            self.napariviewer.scale_bar.visible=True
            self.napariviewer.scale_bar.unit = "um"
        else:
            self.napariviewer.scale_bar.visible=False

    def getVisFunctionEvalText(self,p1,p2):
        #Get the dropdown info
        moduleMethodEvalTexts = []
        all_layouts = self.VisualisationGroupbox.findChildren(QLayout)

        methodKwargNames_method = []
        methodKwargValues_method = []
        methodName_method = ''
        # Iterate over the items in the layout
        for index in range(len(all_layouts)):
            item = all_layouts[index]
            widget = item.widget()
            if isinstance(item, QLayout):
                for index2 in range(item.count()):
                    item_sub = item.itemAt(index2)
                    widget_sub = item_sub.widget()
                    try:
                        if ("LineEdit" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.VisualisationGroupbox):
                            # The objectName will be along the lines of foo#bar#str
                            #Check if the objectname is part of a method or part of a scoring
                            split_list = widget_sub.objectName().split('#')
                            methodName_method = split_list[1]
                            methodKwargNames_method.append(split_list[2])

                            #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                            methodKwargValues_method.append(widget_sub.text().replace('\\','/'))
                    except:
                        pass
        #If at this point there is no methodName_method, it means that the method has exactly 0 req or opt kwargs. Thus, we simply find the value of the QComboBox which should be the methodName:
        if methodName_method == '':
            for index in range(len(all_layouts)):
                item = all_layouts[index]
                widget = item.widget()
                if isinstance(item, QLayout):
                    for index2 in range(item.count()):
                        item_sub = item.itemAt(index2)
                        widget_sub = item_sub.widget()
                        try:
                            if "VisualisationDropdownKEEP"in widget_sub.objectName():
                                text = widget_sub.currentText()
                                for i in range(len(self.Visualisation_functionNameToDisplayNameMapping)):
                                    if self.Visualisation_functionNameToDisplayNameMapping[i][0] == text:
                                        methodName_method = self.Visualisation_functionNameToDisplayNameMapping[i][1]
                                        break
                        except:
                            pass
        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = utils.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method,partialStringStart=str(p1)+','+str(p2))
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)

        if moduleMethodEvalTexts is not None and len(moduleMethodEvalTexts) > 0:
            return moduleMethodEvalTexts[0]
        else:
            return None

class PostProcessing(QWidget):
    """
    Class that handles the post-processing
    This class handles both the GUI initialisation of the PostProcessing tab and the actual post-processing.
    """
    def __init__(self,parent):
        self.parent=parent
        super().__init__()
        # Create a layout for the main widget
        self.mainlayout = QVBoxLayout()
        # Set the layout for the main widget
        self.setLayout(self.mainlayout)


        #------------Start of GUI dynamic layout -----------------

        #Create a groupbox for PostProcessing methods
        self.PostProcessingGroupbox = QGroupBox("Post-Processing")
        self.PostProcessingGroupbox.setLayout(QGridLayout())
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        self.PostProcessingGroupbox.setObjectName("PostProcessingGroupboxKEEP")

        #Add a combobox-dropdown
        PostProcessingDropdown = QComboBox(self)
        PostProcessingDropdown.setMaxVisibleItems(30)
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        PostProcessingDropdown_name = "PostProcessingDropdownKEEP"
        PostProcessing_functionNameToDisplayNameMapping_name = f"PostProcessing_functionNameToDisplayNameMapping"
        #Get the options dynamically from the PostProcessing folder
        options = utils.functionNamesFromDir('PostProcessing')
        #Also find the mapping of 'display name' to 'function name'
        displaynames, PostProcessing_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options,'')
        #and add this to our self attributes
        setattr(self, PostProcessing_functionNameToDisplayNameMapping_name, PostProcessing_functionNameToDisplayNameMapping)

        #Set a current name and add the items
        PostProcessingDropdown.setObjectName(PostProcessingDropdown_name)
        PostProcessingDropdown.addItems(displaynames)

        #Add a callback to the changing of the dropdown:
        PostProcessingDropdown.currentTextChanged.connect(lambda text: utils.changeLayout_choice(self.PostProcessingGroupbox.layout(),PostProcessingDropdown_name,getattr(self, PostProcessing_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True))

        #Add the visualisationDropdown to the layout
        #Ensure it is full-width with the [1,6]
        self.PostProcessingGroupbox.layout().addWidget(PostProcessingDropdown,1,0,1,6)

        #On startup/initiatlisation: also do changeLayout_choice
        utils.changeLayout_choice(self.PostProcessingGroupbox.layout(),PostProcessingDropdown_name,getattr(self, PostProcessing_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True)

        #add a 'Visualise!' button to this groupbox at the bottom:
        button = QPushButton("PostProcessing!", self)
        #Ensure 'KEEP' is in the objectname so it's never removed
        button.setObjectName("PostProcessingRunButtonKEEP")
        #Ensure it is full-width with the [1,6]
        self.PostProcessingGroupbox.layout().addWidget(button,99,0,1,6)
        #And add a callback to this:
        button.clicked.connect(lambda text, parent=parent: self.PostProcessing_callback(parent))

        #Add the groupbox to the mainlayout
        self.mainlayout.layout().addWidget(self.PostProcessingGroupbox)

        #------------End of GUI dynamic layout -----------------

        #Add a vertical spacer to push everything to top and bottom:
        spacer = QSpacerItem(0, 1e5, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.mainlayout.layout().addItem(spacer)

        #Add the history grid-layout to the mainlayout:
        self.PostProcessingHistoryGrid = self.PostProcessingHistoryGrid(self)
        self.mainlayout.layout().addWidget(self.PostProcessingHistoryGrid)

        #Add a history of post-processing
        self.postProcessingHistory = {}
        logging.info('PostProcessing init')

    def PostProcessing_callback(self,parent):
        logging.info('PostProcessing button pressed')

        #Get the current function callback
        FunctionEvalText = self.getPostProcessingFunctionEvalText("parent.data['FittingResult'][0]","parent.data['FindingResult'][0]","parent.globalSettings")
        print(FunctionEvalText)

        #Store the history of postprocessing
        current_postprocessinghistoryid = len(self.postProcessingHistory)
        self.postProcessingHistory[current_postprocessinghistoryid] = {}
        self.postProcessingHistory[current_postprocessinghistoryid][0] = self.parent.data['FittingResult'][0]
        self.postProcessingHistory[current_postprocessinghistoryid][1] = FunctionEvalText
        self.postProcessingHistory[current_postprocessinghistoryid][2] = [len(self.postProcessingHistory[current_postprocessinghistoryid][0]),-1]

        postProcessingResult = eval(FunctionEvalText)
        self.parent.data['FittingResult'][0] = postProcessingResult[0]

        self.postProcessingHistory[current_postprocessinghistoryid][2][1] = len(self.parent.data['FittingResult'][0])

        #Update the history grid-widget
        self.PostProcessingHistoryGrid.addHistoryEntryToGrid(current_postprocessinghistoryid)

        self.parent.updateLocList()

    class PostProcessingHistoryGrid(QWidget):
        def __init__(self,parent):
            super().__init__()

            self.parent=parent
            # Create a vertical layout to hold the scroll area
            main_layout = QVBoxLayout()

            # Set the main layout of the widget
            self.setLayout(main_layout)

            groupbox = QGroupBox('PostProcessing history')
            main_layout.addWidget(groupbox)

            #Create a scroll area
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            #Give it a container
            container = QtWidgets.QWidget()
            scroll_area.setWidget(container)
            #Define row heights for the grid
            gridRowHeights = [30,30,30]
            #The scroll area needs to be somewhat larger than this
            scroll_area.setMinimumSize(0,sum(gridRowHeights)+25)

            #Create a qgridlayout:
            self.grid_layout = QGridLayout()
            #Set the minimum size of this grid layout
            self.grid_layout.setRowMinimumHeight(0,gridRowHeights[0])
            self.grid_layout.setRowMinimumHeight(1,gridRowHeights[1])
            self.grid_layout.setRowMinimumHeight(2,gridRowHeights[2])

            #Put it inside the container:
            container.setLayout(self.grid_layout)

            container2 = QVBoxLayout()
            container2.addWidget(scroll_area)
            groupbox.setLayout(container2)

            self.grid_layout.setSizeConstraint(QLayout.SetMinimumSize)
            main_layout.setSizeConstraint(QLayout.SetMinimumSize)
            # groupbox.setSizeConstraint(QLayout.SetMinimumSize)
            container2.setSizeConstraint(QLayout.SetMinimumSize)
            # container.setSizeConstraint(QLayout.SetMinimumSize)

            #OK, so now we have a self, which has main_layout, which contains groupbox, which contains container2, which contains scroll_area, which contains container, which contains grid_layout.

        def addHistoryEntryToGrid(self,historyId):

            #Add some entries to the gridlayout:
            #This is the title of what operation we did

            #We know from earlier that this is the text we always put in:
            #"parent.data['FittingResult'][0]","parent.data['FindingResult'][0]","parent.globalSettings"
            #Thus, we can extract the info around this
            fullText = self.parent.postProcessingHistory[historyId][1]
            historyText = fullText.split("parent.data[\'FittingResult\'][0],parent.data[\'FindingResult\'][0],parent.globalSettings")
            #We do a little cleanup:
            historyText[0] = historyText[0].replace("(","")
            historyText[1] = historyText[1].replace(")","")[1:]
            #We create a QLabel from this
            historyTextLabel = QLabel("<html><b>"+historyText[0]+'</b><br>'+historyText[1]+"</html>")
            historyTextLabel.setWordWrap(True)
            self.grid_layout.addWidget(historyTextLabel, 0,historyId)

            #Then we also show the change in nr of localizations
            self.grid_layout.addWidget(QLabel(str(self.parent.postProcessingHistory[historyId][2][0])+"-->"+str(self.parent.postProcessingHistory[historyId][2][1])+" entries"), 1,historyId)

            #This is a button to restore to before this operation
            button = QPushButton("Restore to before this", self)
            self.grid_layout.addWidget(button, 2,historyId)

            #And add a callback to this:
            button.clicked.connect(lambda text, historyId=historyId: self.historyRestore_callback(historyId))

        def removeHistoryEntryFromGrid(self,historyId):
            #Remove the entries from the grid layout at this point in history
            self.grid_layout.removeWidget(self.grid_layout.itemAtPosition(0,historyId).widget())
            self.grid_layout.removeWidget(self.grid_layout.itemAtPosition(1,historyId).widget())
            self.grid_layout.removeWidget(self.grid_layout.itemAtPosition(2,historyId).widget())

        def historyRestore_callback(self,historyId):
            #First restore the localizations
            self.parent.parent.data['FittingResult'][0] = self.parent.postProcessingHistory[historyId][0]
            #Remove all history after this point
            keys_to_remove = [k for k in self.parent.postProcessingHistory if k >= historyId]
            for k in reversed(keys_to_remove):
                del self.parent.postProcessingHistory[k]
                self.removeHistoryEntryFromGrid(k)

            #Update the grandparent loc list
            self.parent.parent.updateLocList()
            #Give some info
            logging.info('Restored postprocessing history')

        def clearHistory(self):
            for i in reversed(range(self.grid_layout.count())):
                self.grid_layout.removeWidget(self.grid_layout.itemAt(i).widget())
            self.parent.postProcessingHistory = {}
            logging.info('Cleared postprocessing history')

    def getPostProcessingFunctionEvalText(self,p1,p2,p3):
        #Get the dropdown info
        moduleMethodEvalTexts = []
        all_layouts = self.PostProcessingGroupbox.findChildren(QLayout)

        methodKwargNames_method = []
        methodKwargValues_method = []
        methodName_method = ''
        # Iterate over the items in the layout
        for index in range(len(all_layouts)):
            item = all_layouts[index]
            widget = item.widget()
            if isinstance(item, QLayout):
                for index2 in range(item.count()):
                    item_sub = item.itemAt(index2)
                    widget_sub = item_sub.widget()
                    try:
                        if ("LineEdit" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.PostProcessingGroupbox):
                            # The objectName will be along the lines of foo#bar#str
                            #Check if the objectname is part of a method or part of a scoring
                            split_list = widget_sub.objectName().split('#')
                            methodName_method = split_list[1]
                            methodKwargNames_method.append(split_list[2])

                            #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                            methodKwargValues_method.append(widget_sub.text().replace('\\','/'))
                    except:
                        pass
        #If at this point there is no methodName_method, it means that the method has exactly 0 req or opt kwargs. Thus, we simply find the value of the QComboBox which should be the methodName:
        if methodName_method == '':
            for index in range(len(all_layouts)):
                item = all_layouts[index]
                widget = item.widget()
                if isinstance(item, QLayout):
                    for index2 in range(item.count()):
                        item_sub = item.itemAt(index2)
                        widget_sub = item_sub.widget()
                        try:
                            if "PostProcessingDropdownKEEP"in widget_sub.objectName():
                                text = widget_sub.currentText()
                                for i in range(len(self.PostProcessing_functionNameToDisplayNameMapping)):
                                    if self.PostProcessing_functionNameToDisplayNameMapping[i][0] == text:
                                        methodName_method = self.PostProcessing_functionNameToDisplayNameMapping[i][1]
                                        break
                        except:
                            pass
        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = utils.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method,partialStringStart=str(p1)+','+str(p2)+','+str(p3))
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)

        if moduleMethodEvalTexts is not None and len(moduleMethodEvalTexts) > 0:
            return moduleMethodEvalTexts[0]
        else:
            return None

class PreviewFindingFitting(QWidget):
    """
    Class that runs the GUI of finding/fitting preview (i.e. showing alle vents as an image and overlays with boxes/dots)
    """
    def __init__(self,parent=None):
        """
        Initialisation of the PreviewFindingFitting class. Sets up a napari viewer and adds the viewer and viewer control widgets to the main layout.
        Also initialises some empty arrays
        """
        super().__init__()
        self.parent=parent #get parent info (main GUI)
        # Create a napari viewer
        self.napariviewer = Viewer(show=False)
        # Create a layout for the main widget
        self.mainlayout = QVBoxLayout()
        # Set the layout for the main widget
        self.setLayout(self.mainlayout)

        self.viewer = QtViewer(self.napariviewer)

        #Create a new 'this is the events under the cursor' textbox
        self.underCursorInfo = QLabel("")

        self.viewer.on_mouse_move = lambda event: self.currently_under_cursor(event)
        self.viewer.on_mouse_double_click = lambda event: self.napari_doubleClicked(event)
        #Test
        # self.viewer.window.add_plugin_dock_widget('napari-1d', 'napari-1d')



        #Add widgets to the main layout
        self.mainlayout.addWidget(self.underCursorInfo) #Text box with info under cursor
        self.mainlayout.addWidget(self.viewer) #The view itself
        self.viewer.controls.setAutoFillBackground(True)
        self.mainlayout.addWidget(self.viewer.controls) #The controls for the viewer (contrast, time-point, etc)
        # self.mainlayout.addWidget(self.viewer.layers)

        # self.mainlayout.addWidget(self.viewer.dockConsole)
        # self.mainlayout.addWidget(self.viewer.layerButtons)
        # self.mainlayout.addWidget(self.viewer.viewerButtons)
        logging.info('PreviewFindingFitting init')
        self.finding_overlays = {}
        self.fitting_overlays = {}
        self.findingFitting_overlay = self.napariviewer.add_shapes([],visible=False,name='Finding/Fitting Results')
        self.findingFitting_overlay.opacity = 0.7
        self.fittingResult = {}
        self.findingResult = {}
        self.settings = {}
        self.timeStretch=[]
        self.events = {}
        self.frametime_ms = 0
        self.maxFrames = 0

        #An array holding info about what's under the cursor
        underCursorInfo = {
            'current_pixel': [[-np.inf,-np.inf]],
            'current_time': [[-1,-1]],
            'current_candidate': [[-1]]
        }

        self.underCursorInfoDF = pd.DataFrame(underCursorInfo)

    def napari_doubleClicked(self, event:Event):
        """Callback when the mouse is double clicked. Brings someone to the candidate preview tab if a candidate is clicked

        """
        if self.underCursorInfoDF['current_candidate'][0][0] > -1:
            print(self.underCursorInfoDF['current_candidate'][0][0])

            #Set the value of the entry in the candidate preview tab
            self.parent.canPreviewtab_widget.entryCanPreview.setText(str(self.underCursorInfoDF['current_candidate'][0][0]))
            #Update the candidate preview tab:
            self.parent.canPreviewtab_widget.show_candidate_callback(self.parent)
            #And change the tab
            utils.changeTab(self.parent, text='Candidate preview')

    def currently_under_cursor(self,event: Event):
        """
        Class that determines which pixel is currently under the cursor. The main task of this function is to go from cavnas position to image position.

        Args:
            event (Event): general Vispy event, containing, amongst others, the xy position of the curosr in the canvas.
        """
        #Vispy mouse position

        #We get the canvas size/position in image pixel units
        canvas_size_in_px_units = event.source.size/self.napariviewer.camera.zoom
        camera_coords = [self.napariviewer.camera.center[2]+.5, self.napariviewer.camera.center[1]+.5]
        canvas_pos = np.vstack([camera_coords-(canvas_size_in_px_units/2),camera_coords+(canvas_size_in_px_units/2)])
        #And we can normalize the cursor position to image pixels
        cursor_unit_norm = event._pos/event.source.size
        #Thus, we can find the pixel index - at the moment we simply calculate this and not do anything with it
        highlighted_px_index=np.zeros((2,))
        highlighted_px_index[0] = cursor_unit_norm[0]*canvas_size_in_px_units[0]+canvas_pos[0][0]
        highlighted_px_index[1] = cursor_unit_norm[1]*canvas_size_in_px_units[1]+canvas_pos[0][1]

        #Here's the calculated pixel index in x,y coordinate.
        pixel_index = np.floor(highlighted_px_index).astype(int)

        self.underCursorInfoDF['current_pixel'][0] = np.floor(highlighted_px_index)
        self.updateUnderCursorInfo()

    def updateUnderCursorInfo(self):
        self.underCursorInfoDF
        fullText = ''

        if self.underCursorInfoDF['current_time'][0][0] > -np.inf:
            fullText+=f"Time: {self.underCursorInfoDF['current_time'][0][0]} - {self.underCursorInfoDF['current_time'][0][1]} ms"

        if self.underCursorInfoDF['current_pixel'][0][0] > -np.inf:
            # fullText+=f"; Current pixel: {self.underCursorInfoDF['current_pixel'][0][0]},{self.underCursorInfoDF['current_pixel'][0][1]}"

            #Determine how many pos/neg events are in this pixel and time-frame:
            events=self.events
            pos_events = len(events[(events['x']-min(events['x']) == self.underCursorInfoDF['current_pixel'][0][0]) & (events['y']-min(events['y']) == self.underCursorInfoDF['current_pixel'][0][1]) & (events['t'] >= self.underCursorInfoDF['current_time'][0][0]*1000) & (events['t'] <= self.underCursorInfoDF['current_time'][0][1]*1000) & (events['p'] == 1)])
            neg_events = len(events[(events['x']-min(events['x']) == self.underCursorInfoDF['current_pixel'][0][0]) & (events['y']-min(events['y']) == self.underCursorInfoDF['current_pixel'][0][1]) & (events['t'] >= self.underCursorInfoDF['current_time'][0][0]*1000) & (events['t'] <= self.underCursorInfoDF['current_time'][0][1]*1000) & (events['p'] == 0)])
            #Add it to the text
            fullText += f"; Pos: {pos_events}; Neg: {neg_events}"

            #Check if we are within the bounding box of a candidate:
            #Reset to now have any candidate in the dataframe
            self.underCursorInfoDF['current_candidate'][0][0] = -1
            #Loop over the frames and check
            for f in self.findingResult:
                #Check if this findingResult should be displayed:
                t_min = min(self.findingResult[f]['events']['t'])/1000
                t_max = max(self.findingResult[f]['events']['t'])/1000
                if t_min < self.underCursorInfoDF['current_time'][0][1] and t_max > self.underCursorInfoDF['current_time'][0][0]:
                    #Get the finding bbox...
                    x_min = min(self.findingResult[f]['events']['x'])-min(events['x'])
                    x_max = max(self.findingResult[f]['events']['x'])-min(events['x'])
                    y_min = min(self.findingResult[f]['events']['y'])-min(events['y'])
                    y_max = max(self.findingResult[f]['events']['y'])-min(events['y'])

                    if (x_min <= self.underCursorInfoDF['current_pixel'][0][0] <= x_max) and (y_min <= self.underCursorInfoDF['current_pixel'][0][1] <= y_max):
                        self.underCursorInfoDF['current_candidate'][0][0] = f

            if self.underCursorInfoDF['current_candidate'][0][0] > -1:
                fullText += f"; Candidate: {self.underCursorInfoDF['current_candidate'][0][0]}"

        self.underCursorInfo.setText(fullText)

        pass

    def displayEvents(self,events,frametime_ms=100,findingResult = None,fittingResult=None,settings=None,timeStretch=(0,1000)):
        """
        Function that's called with new a new preview window is generated. It requires the input of the events (as a list), others are optional

        Args:
            events (list): list of all events (x,y,pol,time)
            frametime_ms (int, optional): The displayed frametime in milliseconds. Defaults to 100.
            findingResult (dict, optional): The visualised finding result (i.e. boxes). Defaults to None.
            fittingResult (dict, optional): The visualised fitting result (i.e. crosses). Defaults to None.
            settings (dict, optional): Settings. Defaults to None.
            timeStretch (tuple, optional): The time stretch to visualise. Defaults to (0,1000).
        """
        #Add variables to self:
        self.events = events
        self.frametime_ms = frametime_ms
        self.findingResult = findingResult
        self.fittingResult = fittingResult
        self.settings = settings
        self.timeStretch = timeStretch

        #Delete all existing layers:
        for layer in reversed(self.napariviewer.layers):
            self.napariviewer.layers.remove(layer)

        #Create a new image
        preview_multiD_image = []
        #Loop over the frames:
        n_frames = int(np.ceil(float(timeStretch[1])/(frametime_ms)))
        self.maxFrames = n_frames
        self.hist_xy = eventDistributions.SumPolarity(events)
        for n in range(0,n_frames):
            #Get the events on this 'frame'
            events_this_frame = events[(events['t']>(float(timeStretch[0])*1000+n*frametime_ms*1000)) & (events['t']<(float(timeStretch[0])*1000+(n+1)*frametime_ms*1000))]
            #Create a 2d histogram out of this
            # thisHist_xy = eventDistributions.SumPolarity(events_this_frame)
            thisHist_xy = self.hist_xy(events_this_frame)
            #Add it to our image
            preview_multiD_image.append(thisHist_xy[0])

        #add this image to the napariviewer
        self.napariviewer.add_image(np.asarray(preview_multiD_image), multiscale=False)

        #Append the finding/fitting overlay on top of this:
        self.napariviewer.layers.append(self.findingFitting_overlay)

        #Select the original image-layer as selected
        self.napariviewer.layers.selection.active = self.napariviewer.layers[0]
        self.update_visibility()
        self.napariviewer.dims.events.current_step.connect(self.update_visibility)

    def update_visibility(self):
        """
        Function to update visibility based on the current layer
        """
        #Reset all data in the findingFitting overlay:
        self.findingFitting_overlay.data=[]

        #And re-draw the necessary data only:

        #Current frame is this:
        curr_frame = self.napariviewer.dims.current_step[0]

        #Get current time-bounds:
        curr_time_bounds = (float(self.timeStretch[0])+(curr_frame)*self.frametime_ms,float(self.timeStretch[0])+(curr_frame+1)*self.frametime_ms)

        #Get finding results:
        findingResults = self.findingResult
        fittingResults = self.fittingResult
        pxsize = float(self.settings['PixelSize_nm']['value'])
        candidate_polygons = []
        unfitted_candidate_polygons = []
        fitting_polygons = []
        candidates_ids = []
        unfitted_candidate_ids = []
        fitting_ids = []

        for f in findingResults:
            #Check if this findingResult should be displayed:
            t_min = min(findingResults[f]['events']['t'])/1000
            t_max = max(findingResults[f]['events']['t'])/1000
            if t_min < curr_time_bounds[1] and t_max > curr_time_bounds[0]:

                #Get the finding info...
                y_min = min(findingResults[f]['events']['x']) - self.hist_xy.xlim[0]
                y_max = max(findingResults[f]['events']['x']) - self.hist_xy.xlim[0]
                x_min = min(findingResults[f]['events']['y']) - self.hist_xy.ylim[0]
                x_max = max(findingResults[f]['events']['y']) - self.hist_xy.ylim[0]

                #Check if it succesfully fitted:
                if not np.isnan(fittingResults['x'].iloc[f]):
                    candidates_ids.append(str(f))
                    candidate_polygons.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
                else:
                    unfitted_candidate_ids.append(str(f))
                    unfitted_candidate_polygons.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))

        #Also add fitting info...
        for f in range(len(fittingResults)):
            #Check if this findingResult should be displayed:
            t = fittingResults['t'].iloc[f]
            if t > curr_time_bounds[0] and t < curr_time_bounds[1]:
                xcoord_pxcoord = fittingResults['x'].iloc[f]/float(pxsize)-self.hist_xy.xlim[0]
                ycoord_pxcoord = fittingResults['y'].iloc[f]/float(pxsize)-self.hist_xy.ylim[0]

                fitting_polygons.append(np.array([[ycoord_pxcoord-1,xcoord_pxcoord-1],[ycoord_pxcoord+1,xcoord_pxcoord+1]]))
                fitting_polygons.append(np.array([[ycoord_pxcoord-1,xcoord_pxcoord+1],[ycoord_pxcoord+1,xcoord_pxcoord-1]]))

                #Need to append empty values:
                fitting_ids.append("")
                fitting_ids.append("")

        all_ids = candidates_ids+unfitted_candidate_ids+fitting_ids
        # create features
        features = {
            'candidate_ids': all_ids,
        }
        text = {
            'string': '{candidate_ids}',
            'anchor': 'upper_left',
            'translation': [0, 0],
            'size': 8,
            'color':'blue'
        }

        #Add the finding result
        self.findingFitting_overlay.add_rectangles(candidate_polygons, edge_width=1,edge_color='coral',face_color='transparent')
        self.findingFitting_overlay.add_rectangles(unfitted_candidate_polygons, edge_width=1,edge_color='red',face_color='transparent')
        self.findingFitting_overlay.add_lines(fitting_polygons, edge_width=0.5,edge_color='red',face_color='transparent')
        self.findingFitting_overlay.features = pd.DataFrame(features)
        self.findingFitting_overlay.text = text
        self.findingFitting_overlay.visible=True

        #Update the informative text
        self.underCursorInfoDF['current_time'][0] = curr_time_bounds
        self.updateUnderCursorInfo()

class CandidatePreview(QWidget):
    """
    Class to visualise the candidates and localizations in different representations, e.g. histograms and pointclouds
    """
    def __init__(self, parent):
        super().__init__(parent)
        # Members
        self.CandidatePreviewLocs = pd.DataFrame()
        self.CandidatePreviewID = None
        # Create a layout for the main widget
        self.mainlayout = QVBoxLayout()
        # Set the layout for the main widget
        self.setLayout(self.mainlayout)

        #Add a horizontal layout to the first row of the vertical layout - this contains the entry fields and the buttons
        canPreviewtab_horizontal_container = QHBoxLayout()
        self.mainlayout.addLayout(canPreviewtab_horizontal_container)

        #Add a entry field to type the number of candidate and button to show it
        canPreviewtab_horizontal_container.addWidget(QLabel("Candidate ID: "))
        self.entryCanPreview = QLineEdit()
        onlyInt = QIntValidator()
        onlyInt.setBottom(0)
        self.entryCanPreview.setValidator(onlyInt)
        canPreviewtab_horizontal_container.addWidget(self.entryCanPreview)
        self.entryCanPreview.returnPressed.connect(lambda: self.show_candidate_callback(parent))

        self.buttonCanPreview = QPushButton("Show candidate")
        canPreviewtab_horizontal_container.addWidget(self.buttonCanPreview)

        #Give it a function on click:
        self.buttonCanPreview.clicked.connect(lambda: self.show_candidate_callback(parent))

        self.prev_buttonCan = QPushButton("Previous")
        self.prev_buttonCan.clicked.connect(lambda: self.prev_candidate_callback(parent))

        self.next_buttonCan = QPushButton("Next")
        self.next_buttonCan.clicked.connect(lambda: self.next_candidate_callback(parent))

        canPreviewtab_horizontal_container.addWidget(self.prev_buttonCan)
        canPreviewtab_horizontal_container.addWidget(self.next_buttonCan)

        self.candidate_info = QLabel('')
        self.mainlayout.addWidget(self.candidate_info)
        self.fit_info = QLabel('')
        self.fit_info.setStyleSheet("color: red;")
        self.mainlayout.addWidget(self.fit_info)

        #------------Start of first candidate plot layout -----------------
        # create a groupbox for the plot options of the first canvas
        self.firstCandidatePreviewGroupbox = QGroupBox("First Plot")
        self.firstCandidatePreviewGroupbox.setLayout(QGridLayout())
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        self.firstCandidatePreviewGroupbox.setObjectName("firstCanPrevGroupboxKEEP")

        # Add a combobox-dropdown
        firstCanPrevDropdown = QComboBox(self)
        firstCanPrevDropdown.setMaxVisibleItems(30)
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        firstCanPrevDropdown_name = f"firstCanPrevDropdownKEEP"
        firstCanPrevDropdown.setObjectName(firstCanPrevDropdown_name)
        firstCanPrev_functionNameToDisplayNameMapping_name = f"firstCanPrev_functionNameToDisplayNameMapping"
        # Get the options dynamically from the CandidatePreview folder
        options = utils.functionNamesFromDir('CandidatePreview')
        #Also find the mapping of 'display name' to 'function name'
        displaynames, firstCanPrev_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options,'')
        #and add this to our self attributes
        setattr(self, firstCanPrev_functionNameToDisplayNameMapping_name,  firstCanPrev_functionNameToDisplayNameMapping)

        #Set a current name and add the items
        firstCanPrevDropdown.addItems(displaynames)

        #Add a callback to the changing of the dropdown:
        firstCanPrevDropdown.currentTextChanged.connect(lambda text: utils.changeLayout_choice(self.firstCandidatePreviewGroupbox.layout(),firstCanPrevDropdown_name,getattr(self, firstCanPrev_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True, maxNrRows=1))
        self.firstCandidatePreviewGroupbox.layout().addWidget(firstCanPrevDropdown,1,0,1,8)

        #On startup/initiatlisation: also do changeLayout_choice
        utils.changeLayout_choice(self.firstCandidatePreviewGroupbox.layout(),firstCanPrevDropdown_name,getattr(self, firstCanPrev_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True, maxNrRows=1)

        #Add the groupbox to the mainlayout
        self.mainlayout.layout().addWidget(self.firstCandidatePreviewGroupbox)

        #Create an empty figure and store it as self.data:
        self.firstCandidateFigure = plt.figure(figsize=(6.8,4))
        self.firstCandidateCanvas = FigureCanvas(self.firstCandidateFigure)
        self.mainlayout.addWidget(self.firstCandidateCanvas)

        #Add a navigation toolbar (zoom, pan etc) and canvas to tab
        self.mainlayout.addWidget(NavigationToolbar(self.firstCandidateCanvas, self))
        #------------End of first candidate plot layout -----------------

        #------------Start of second candidate plot layout -----------------
        # create a groupbox for the plot options of the first canvas
        self.secondCandidatePreviewGroupbox = QGroupBox("Second Plot")
        self.secondCandidatePreviewGroupbox.setLayout(QGridLayout())
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        self.secondCandidatePreviewGroupbox.setObjectName("secondCanPrevGroupboxKEEP")

        # Add a combobox-dropdown
        secondCanPrevDropdown = QComboBox(self)
        secondCanPrevDropdown.setMaxVisibleItems(30)
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        secondCanPrevDropdown_name = f"secondCanPrevDropdownKEEP"
        secondCanPrevDropdown.setObjectName(secondCanPrevDropdown_name)
        secondCanPrev_functionNameToDisplayNameMapping_name = f"secondCanPrev_functionNameToDisplayNameMapping"
        # Get the options dynamically from the CandidatePreview folder
        options = utils.functionNamesFromDir('CandidatePreview')
        #Also find the mapping of 'display name' to 'function name'
        displaynames, secondCanPrev_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options,'')
        #and add this to our self attributes
        setattr(self, secondCanPrev_functionNameToDisplayNameMapping_name,  secondCanPrev_functionNameToDisplayNameMapping)

        #Set a current name and add the items
        secondCanPrevDropdown.addItems(displaynames)

        #Add a callback to the changing of the dropdown:
        secondCanPrevDropdown.currentTextChanged.connect(lambda text: utils.changeLayout_choice(self.secondCandidatePreviewGroupbox.layout(),secondCanPrevDropdown_name,getattr(self, secondCanPrev_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True, maxNrRows=1))
        self.secondCandidatePreviewGroupbox.layout().addWidget(secondCanPrevDropdown,1,0,1,8)

        #On startup/initiatlisation: also do changeLayout_choice
        utils.changeLayout_choice(self.secondCandidatePreviewGroupbox.layout(),secondCanPrevDropdown_name,getattr(self, secondCanPrev_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True, maxNrRows=1)

        #Add the groupbox to the mainlayout
        self.mainlayout.layout().addWidget(self.secondCandidatePreviewGroupbox)

        #Create an empty figure and store it as self.data:
        self.secondCandidateFigure = plt.figure(figsize=(6.8,4))
        plt.rcParams.update({'font.size': 9}) # global change of font size for all matplotlib figures
        self.secondCandidateCanvas = FigureCanvas(self.secondCandidateFigure)
        self.mainlayout.addWidget(self.secondCandidateCanvas)

        #Add a navigation toolbar (zoom, pan etc) and canvas to tab
        self.mainlayout.addWidget(NavigationToolbar(self.secondCandidateCanvas, self))
        #------------End of second candidate plot layout -----------------

        logging.info('CandidatePreview init')


    def show_candidate_callback(self, parent, reset=False):
        # First clear text and figures
        self.candidate_info.setText('')
        self.fit_info.setText('')
        self.firstCandidateFigure.clf()
        self.secondCandidateFigure.clf()
        self.firstCandidateCanvas.draw()
        self.secondCandidateCanvas.draw()

        if reset==False:
            # Check candidate entry field
            if self.entryCanPreview.text() == '':
                self.candidate_info.setText('Tried to visualise candidate, but no ID was given!')
                logging.error('Tried to visualise candidate, but no ID was given!')
            elif 'FindingMethod' in parent.data and int(self.entryCanPreview.text()) < len(parent.data['FindingResult'][0]):
                self.CandidatePreviewID = int(self.entryCanPreview.text())
                logging.debug(f"Attempting to show candidate {self.CandidatePreviewID}.")
                # Get all localizations that belong to the candidate
                self.CandidatePreviewLocs = parent.data['FittingResult'][0][parent.data['FittingResult'][0]['candidate_id'] == self.CandidatePreviewID]
                if pd.isna(self.CandidatePreviewLocs['x'].iloc[0]):
                    self.fit_info.setText(f"No localization generated due to {self.CandidatePreviewLocs['fit_info'].iloc[0]}")

                # Get some info about the candidate
                N_events = parent.data['FindingResult'][0][self.CandidatePreviewID]['N_events']
                cluster_size = parent.data['FindingResult'][0][self.CandidatePreviewID]['cluster_size']
                self.candidate_info.setText(f"This candidate cluster contains {N_events} events and has dimensions ({cluster_size[0]}, {cluster_size[1]}, {cluster_size[2]}).")
                FirstFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['FindingResult'][0][self.CandidatePreviewID]['events']", "self.CandidatePreviewLocs", "parent.previewEvents", "self.firstCandidateFigure","parent.globalSettings")
                SecondFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['FindingResult'][0][self.CandidatePreviewID]['events']", "self.CandidatePreviewLocs", "parent.previewEvents", "self.secondCandidateFigure","parent.globalSettings")
                eval(FirstFunctionEvalText)
                eval(SecondFunctionEvalText)
                # Improve figure layout and update the first canvases
                # self.firstCandidateFigure.tight_layout()
                self.firstCandidateCanvas.draw()
                # self.secondCandidateFigure.tight_layout()
                self.secondCandidateCanvas.draw()
            else:
                self.candidate_info.setText('Tried to visualise candidate but no data found!')
                logging.error('Tried to visualise candidate but no data found!')
        else:
            logging.info('Candidate preview is reset.')


    def prev_candidate_callback(self, parent):
        if not self.entryCanPreview.text()=='':
            if 'FindingMethod' in parent.data and int(self.entryCanPreview.text())-1 < len(parent.data['FindingResult'][0]):
                if int(self.entryCanPreview.text())-1 == -1:
                    max_candidate = len(parent.data['FindingResult'][0])-1
                    self.entryCanPreview.setText(str(max_candidate))
                else:
                    self.entryCanPreview.setText(str(int(self.entryCanPreview.text())-1))
                self.show_candidate_callback(parent)
            else:
                self.candidate_info.setText('Tried to visualise candidate but no data found!')
                logging.error('Tried to visualise candidate but no data found!')
        else:
            self.candidate_info.setText('Tried to visualise candidate, but no ID was given!')
            logging.error('Tried to visualise candidate, but no ID was given!')

    def next_candidate_callback(self, parent):
        if not self.entryCanPreview.text()=='':
            if 'FindingMethod' in parent.data and int(self.entryCanPreview.text())+1 <= len(parent.data['FindingResult'][0]):
                if int(self.entryCanPreview.text())+1 == len(parent.data['FindingResult'][0]):
                    self.entryCanPreview.setText('0')
                else:
                    self.entryCanPreview.setText(str(int(self.entryCanPreview.text())+1))
                self.show_candidate_callback(parent)
            else:
                self.candidate_info.setText('Tried to visualise candidate but no data found!')
                logging.error('Tried to visualise candidate but no data found!')
        else:
            self.candidate_info.setText('Tried to visualise candidate, but no ID was given!')
            logging.error('Tried to visualise candidate, but no ID was given!')

    def getCanPrevFunctionEvalText(self, p1, p2, p3, p4, p5):
        #Get the dropdown info
        moduleMethodEvalTexts = []
        figurePositon = 'first' if 'first' in p4 else 'second' if 'second' in p4 else None
        groupbox_name = f"{figurePositon}CanPrevGroupboxKEEP"
        functionNameToDisplayNameMapping = f"{figurePositon}CanPrev_functionNameToDisplayNameMapping"
        all_layouts = self.findChild(QtWidgets.QGroupBox, groupbox_name).findChildren(QtWidgets.QLayout)
        # all_layouts = getattr(self, groupbox_name).findChildren(QLayout)

        methodKwargNames_method = []
        methodKwargValues_method = []
        methodName_method = ''
        # Iterate over the items in the layout
        for index in range(len(all_layouts)):
            item = all_layouts[index]
            widget = item.widget()
            if isinstance(item, QLayout):
                for index2 in range(item.count()):
                    item_sub = item.itemAt(index2)
                    widget_sub = item_sub.widget()
                    try:
                        if ("LineEdit" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.findChild(QtWidgets.QGroupBox, groupbox_name)):
                            # The objectName will be along the lines of foo#bar#str
                            #Check if the objectname is part of a method or part of a scoring
                            split_list = widget_sub.objectName().split('#')
                            methodName_method = split_list[1]
                            methodKwargNames_method.append(split_list[2])

                            #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                            methodKwargValues_method.append(widget_sub.text().replace('\\','/'))
                    except:
                        pass
        #If at this point there is no methodName_method, it means that the method has exactly 0 req or opt kwargs. Thus, we simply find the value of the QComboBox which should be the methodName:
        if methodName_method == '':
            for index in range(len(all_layouts)):
                item = all_layouts[index]
                widget = item.widget()
                if isinstance(item, QLayout):
                    for index2 in range(item.count()):
                        item_sub = item.itemAt(index2)
                        widget_sub = item_sub.widget()
                        try:
                            if f"{figurePositon}CanPrevDropdownKEEP"in widget_sub.objectName():
                                text = widget_sub.currentText()
                                for i in range(len(getattr(self, functionNameToDisplayNameMapping))):
                                    if getattr(self, functionNameToDisplayNameMapping)[i][0] == text:
                                        methodName_method = getattr(self, functionNameToDisplayNameMapping)[i][1]
                                        break
                        except:
                            pass
        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = utils.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method,partialStringStart=str(p1)+','+str(p2)+','+str(p3)+','+str(p4)+','+str(p5))
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)

        if moduleMethodEvalTexts is not None and len(moduleMethodEvalTexts) > 0:
            return moduleMethodEvalTexts[0]
        else:
            return None

class TableModel(QAbstractTableModel):
    """TableModel that heavily speedsup the table view
    Blatantly taken from https://stackoverflow.com/questions/71076164/fastest-way-to-fill-or-read-from-a-qtablewidget-in-pyqt5
    """

    def __init__(self, table_data, parent=None):
        super().__init__(parent)
        self.table_data = table_data

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return self.table_data.shape[0]

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return self.table_data.shape[1]

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if role == Qt.DisplayRole:
            return str(self.table_data.loc[index.row()][index.column()])

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self.table_data.columns[section])

    def setColumn(self, col, array_items):
        """Set column data"""
        self.table_data[col] = array_items
        # Notify table, that data has been changed
        self.dataChanged.emit(QModelIndex(), QModelIndex())

    def getColumn(self, col):
        """Get column data"""
        return self.table_data[col]