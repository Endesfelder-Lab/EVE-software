#General imports
import sys, os, logging, json, argparse, datetime, glob, csv, ast, platform, threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import copy
import appdirs
import pickle
import time

#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile

#Custom imports
# Add the folder 2 folders up to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#Import all scripts in the custom script folders
from CandidateFitting import *
from CandidateFinding import *
#Obtain the helperfunctions
from Utils import utils, utilsHelper

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------

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
    
    def initGlobalSettings(self):
        #Initialisation of the global settings - runs on startup to get all these values, then these can be changed later
        globalSettings = {}
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File",filter="EBS files (*.raw *.npy);;All Files (*)")
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
        #Add a group box on candiddate finding
        self.groupboxFinding = QGroupBox("Candidate finding")
        self.groupboxFinding.setObjectName("groupboxFinding")
        self.groupboxFinding.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.groupboxFinding.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFinding, 2, 0)
        
        # Create a QComboBox and add options - this is the FINDING dropdown
        self.candidateFindingDropdown = QComboBox(self)
        options = utils.functionNamesFromDir('CandidateFinding')
        displaynames, self.Finding_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options)
        self.candidateFindingDropdown.setObjectName("CandidateFinding_candidateFindingDropdown")
        self.candidateFindingDropdown.addItems(displaynames)
        #Add the candidateFindingDropdown to the layout
        self.groupboxFinding.layout().addWidget(self.candidateFindingDropdown,1,0,1,2)
        #Activation for candidateFindingDropdown.activated
        self.candidateFindingDropdown.activated.connect(lambda: self.changeLayout_choice(self.groupboxFinding.layout(),"CandidateFinding_candidateFindingDropdown",self.Finding_functionNameToDisplayNameMapping))
        
        #On startup/initiatlisation: also do changeLayout_choice
        self.changeLayout_choice(self.groupboxFinding.layout(),"CandidateFinding_candidateFindingDropdown",self.Finding_functionNameToDisplayNameMapping)
        
        """
        Candidate Fitting Grid Layout
        """
        self.groupboxFitting = QGroupBox("Candidate fitting")
        self.groupboxFitting.setObjectName("groupboxFitting")
        self.groupboxFitting.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.groupboxFitting.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFitting, 3, 0)
        
        # Create a QComboBox and add options - this is the FITTING dropdown
        self.candidateFittingDropdown = QComboBox(self)
        options = utils.functionNamesFromDir('CandidateFitting')
        self.candidateFittingDropdown.setObjectName("CandidateFitting_candidateFittingDropdown")
        displaynames, self.Fitting_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options)
        self.candidateFittingDropdown.addItems(displaynames)
        #Add the candidateFindingDropdown to the layout
        self.groupboxFitting.layout().addWidget(self.candidateFittingDropdown,1,0,1,2)
        #Activation for candidateFindingDropdown.activated
        self.candidateFittingDropdown.activated.connect(lambda: self.changeLayout_choice(self.groupboxFitting.layout(),"CandidateFitting_candidateFittingDropdown",self.Fitting_functionNameToDisplayNameMapping))
        
        #On startup/initiatlisation: also do changeLayout_choice
        self.changeLayout_choice(self.groupboxFitting.layout(),"CandidateFitting_candidateFittingDropdown",self.Fitting_functionNameToDisplayNameMapping)
               
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
        self.previewLayout.layout().addWidget(QLabel("Start time (ms):"), 0, 0, 1, 2)
        self.previewLayout.layout().addWidget(QLabel("Duration (ms):"), 0, 2, 1, 2)
        #Also the QLineEdits that have useful names:
        self.preview_startTLineEdit = QLineEdit()
        self.preview_startTLineEdit.setObjectName('preview_startTLineEdit')
        self.previewLayout.layout().addWidget(self.preview_startTLineEdit, 1, 0, 1, 2)
        #Give this a default value:
        self.preview_startTLineEdit.setText("0")
        #Same for end time:
        self.preview_durationTLineEdit = QLineEdit()
        self.preview_durationTLineEdit.setObjectName('preview_durationTLineEdit')
        self.previewLayout.layout().addWidget(self.preview_durationTLineEdit, 1, 2, 1, 2)
        self.preview_durationTLineEdit.setText("1000")
        
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
                                                                    self.preview_minYLineEdit.text(),self.preview_maxYLineEdit.text())))
        #Add the button to the layout:
        self.previewLayout.layout().addWidget(self.buttonPreview, 4, 0)

    def previewRun(self,timeStretch=(0,1000),xyStretch=(0,0,0,0)):
        """
        Generates the preview of a run analysis.

        Parameters:
             timeStretch (tuple): A tuple containing the start and end times for the preview.
             xyStretch (tuple): A tuple containing the minimum and maximum x and y coordinates for the preview.

        Returns:
             None
        """
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
                events['x']-=np.min(events['x'])
                events['y']-=np.min(events['y'])
                events['t']-=np.min(events['t'])
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
                # correct the coordinates and time stamps
                events['x']-=np.min(events['x'])
                events['y']-=np.min(events['y'])
                events['t']-=np.min(events['t'])
                #Log the nr of events found:
                logging.info(f"Preview - Found {len(events)} events in the chosen time frame.")
            else:
                logging.error("Preview - No events found in the chosen time frame.")
                return
        else:
            logging.error("Please choose a .raw or .npy file for previews.")
            return
        
        #Load the events in self memory and filter on XY
        self.previewEvents = events
        self.previewEvents = self.filterEvents_xy(self.previewEvents,xyStretch)
        
        #Change global values so nothing is stored - we just want a preview run. This is later set back to orig values (globalSettingsOrig):
        globalSettingsOrig = copy.deepcopy(self.globalSettings)
        self.globalSettings['StoreConvertedRawData']['value'] = False
        self.globalSettings['StoreFileMetadata']['value'] = False
        self.globalSettings['StoreFinalOutput']['value'] = False
        self.globalSettings['StoreFindingOutput']['value'] = False
        
        #Run the current finding and fitting routine only on these events:
        self.runFindingAndFitting(events)
        
        #Reset global settings
        self.globalSettings = globalSettingsOrig
        
        #Update the preview panel and localization list:
        self.updateShowPreview(previewEvents=events)
        self.updateLocList()
    
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
    
    def filterEvents_xy(self,events,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
        """
        Filter events that are in a numpy array to a certain xy stretch.
        """
        #Edit values for x,y coordinates:
        #First check if all entries in array are numbers:
        try:
            #Check if they all are floats:
            if not all(isinstance(x, float) for x in [float(xyStretch[0]),float(xyStretch[1]),float(xyStretch[2]),float(xyStretch[3])]):
                logging.info("No XY cutting due to not all entries being floats.")
            #If they are all float values, we can proceed
            else:
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
        
        self.label2 = QLabel("Hello from Tab 2!")
        tab2_layout.addWidget(self.label2, 0, 0)

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
                self.text_edit.setPlainText(log_contents)
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
        self.LocListTable = QTableWidget()
        tab4_layout.addWidget(self.LocListTable, 0, 0)
        
    def setup_visualisationTab(self):
        """
        Function to set up the Visualisation tab (scatter, average shifted histogram and such)
        """
        #Add a vertical layout, not a grid layout:
        visualisationTab_vertical_container = QVBoxLayout()
        self.tab_visualisation.setLayout(visualisationTab_vertical_container)
        
        #Add a horizontal layout to the first row of the vertical layout - this contains the buttons:
        visualisationTab_horizontal_container = QHBoxLayout()
        visualisationTab_vertical_container.addLayout(visualisationTab_horizontal_container)
        
        #Add a button that says scatter:
        self.buttonScatter = QPushButton("Scatter Plot")
        visualisationTab_horizontal_container.addWidget(self.buttonScatter)
        #Give it a function on click:
        self.buttonScatter.clicked.connect(lambda: self.plotScatter())
        
        #Add a button that says 2d interp histogram:
        self.buttonInterpHist = QPushButton("Interp histogram")
        visualisationTab_horizontal_container.addWidget(self.buttonInterpHist)
        #Give it a function on click:
        self.buttonInterpHist.clicked.connect(lambda: self.plotLinearInterpHist())
        
        #Create an empty figure and store it as self.data:
        self.data['figurePlot'], self.data['figureAx'] = plt.subplots(figsize=(5, 5))
        self.data['figureCanvas'] = FigureCanvas(self.data['figurePlot'])
        self.data['figurePlot'].tight_layout()
        
        #Add a navigation toolbar (zoom, pan etc)
        visualisationTab_vertical_container.addWidget(NavigationToolbar(self.data['figureCanvas'], self))
        #Add the canvas to the tab
        visualisationTab_vertical_container.addWidget(self.data['figureCanvas'])
    
    def plotScatter(self):
        """
        Function that creates a scatter plot from the localizations
        """
        #check if we have results stored:
        if 'FittingMethod' in self.data:
            logging.debug('Attempting to show scatter plot')
            #Delete any existing cbar - needs to be done before anything else:
            try:
                if hasattr(self, 'cbar'):
                    self.cbar.remove()
            except:
                pass
            #Plot the contents of the loclist as scatter points, color-coded on time:
            #Clear axis
            self.data['figureAx'].clear()
            #Plot the data as scatter plot
            scatter = self.data['figureAx'].scatter(self.data['FittingResult'][0]['x'], self.data['FittingResult'][0]['y'], c=self.data['FittingResult'][0]['t'], cmap='viridis')
            self.data['figureAx'].set_xlabel('x [nm]')
            self.data['figureAx'].set_ylabel('y [nm]')
            self.data['figureAx'].set_aspect('equal', adjustable='box')  
            
            #Draw a cbar
            self.cbar = self.data['figurePlot'].colorbar(scatter)
            self.cbar.set_label('Time (ms)')
            
            #Give it a nice layout
            self.data['figurePlot'].tight_layout()
            #Update drawing of the canvas
            self.data['figureCanvas'].draw()
            logging.info('Scatter plot drawn')
        else:
            logging.error('Tried to visualise but no data found!')
    
    def plotLinearInterpHist(self,pixel_recon_dim=10):
        """
        Function that creates an average-shifted histogram (linearly-interpolated histogram)
        Currently hard-coded on both pixel_recon_dim = 10, and run with 3-x-3 shifts
        """
        #check if we have results stored:
        if 'FittingMethod' in self.data:
            #Code inspired by Frontiers Martens et al
            logging.debug('Attempting to show 2d hist plot')
            #Delete any existing cbar - needs to be done before anything else:
            try:
                if hasattr(self, 'cbar'):
                    self.cbar.remove()
            except:
                pass
            # First, an empty two-dimensional array is allocated. It will be populated with 
            # datapoints and finally will be used for the image reconstruction. 
            # Its dimensions are adjusted to the maximum X and Y coordinate values in 
            # the dataset divided by the pixel dimensions
            max_x = max(self.data['FittingResult'][0]['x'])
            max_y = max(self.data['FittingResult'][0]['y'])
            hist_2d = np.zeros((int(max_x/pixel_recon_dim)+3, int(max_y/pixel_recon_dim)+3))
            # We prepare the method which will compute the amount of intensity which will 
            # be received by the pixel based on the subpixel localization of the processed event
            def interpolation_value(x, pixel_dim=10):
                y = (-np.abs(x)/pixel_dim + 1)
                return y

            # In this for loop each datapoint is assigned to four pixels (and in very 
            # exceptional cases to a single pixel if it is positioned at the center of 
            # the reconstruction iamge pixel) in the image reconstruction.
            for index, d in self.data['FittingResult'][0].iterrows():
                if 'int' in d:
                    intensity = d['int']
                else:
                    intensity = 1
                # Based on X and Y coordinates we determine the pixel position by dividing the 
                # coordinates with the floor division (//) operation...
                coord_x = int(d['x'] // pixel_recon_dim) + 1
                coord_y = int(d['y'] // pixel_recon_dim) + 1
                # ... and also we determine the subpixel pisition of the event. We subtract 
                # the halved pixel dimension value from X and Y subpixel position in order 
                # to determine how the event is oriented with respect to the pixel center. 
                # This value will be used  for the intensity distribution and finding 
                # neighboring pixels which will receive a fraction of this intensity as well
                position_x = d['x'] % pixel_recon_dim - pixel_recon_dim/2
                position_y = d['y'] % pixel_recon_dim - pixel_recon_dim/2

                # we calculate the 'pixel-intensity' which is used for the linear interpolation
                x_int = interpolation_value(position_x,pixel_dim=pixel_recon_dim)
                y_int = interpolation_value(position_y,pixel_dim=pixel_recon_dim)

                # Finally we distribute even itnensities to pixels. 
                # The original pixel is at coord_x and coord_y values
                hist_2d[coord_x, coord_y] += x_int*y_int * intensity

                # The horizontal neighbor pixel is on the right (or left) side of the 
                # original pixel, assuming the datapoint is on the right (or left) 
                # half of the original pixel.
                if position_x > 0:
                    hist_2d[coord_x+1, coord_y] += (1-x_int)*y_int * intensity
                else:
                    hist_2d[coord_x-1, coord_y] += (1-x_int)*y_int * intensity

                # Similarly we find a vertical neighbor in up & down dimension.
                if position_y > 0:
                    hist_2d[coord_x, coord_y+1] += x_int*(1-y_int) * intensity
                else:
                    hist_2d[coord_x, coord_y-1] += x_int*(1-y_int) * intensity

                # Finally we find the diagonal neighbors by combining the code used in the 
                # horizontal and vertical neighbours
                if position_x > 0:
                    if position_y > 0:
                        hist_2d[coord_x+1, coord_y+1] += (1-x_int)*(1-y_int) * intensity
                    else:
                        hist_2d[coord_x+1, coord_y-1] += (1-x_int)*(1-y_int) * intensity
                else:
                    if position_y > 0:
                        hist_2d[coord_x-1, coord_y+1] += (1-x_int)*(1-y_int) * intensity
                    else:
                        hist_2d[coord_x-1, coord_y-1] += (1-x_int)*(1-y_int) * intensity

            #Plot hist_2d:
            #Clear axis
            self.data['figureAx'].clear()
            #Plot the data as scatter plot
            self.data['figureAx'].imshow(hist_2d)
            self.data['figureAx'].set_xlabel('x [nm]')
            self.data['figureAx'].set_ylabel('y [nm]')
            self.data['figureAx'].set_aspect('equal', adjustable='box')  
            #Give it a nice layout
            self.data['figurePlot'].tight_layout()
            #Update drawing of the canvas
            self.data['figureCanvas'].draw()
            logging.info('2d interp hist drawn')
        else:
            logging.error('Tried to visualise but no data found!')


    def setup_canPreviewTab(self):
        """
        Function to setup the Candidate Preview tab
        """

        #Add a vertical layout
        canPreviewtab_vertical_container = QVBoxLayout()
        self.tab_canPreview.setLayout(canPreviewtab_vertical_container)
        
        #Add a horizontal layout to the first row of the vertical layout - this contains the entry fields and the buttons
        canPreviewtab_horizontal_container = QHBoxLayout()
        canPreviewtab_vertical_container.addLayout(canPreviewtab_horizontal_container)
        
        #Add a entry field to type the number of candidate and button to show it
        canPreviewtab_horizontal_container.addWidget(QLabel("Candidate ID: "))
        self.entryCanPreview = QLineEdit()
        onlyInt = QIntValidator()
        self.entryCanPreview.setValidator(onlyInt)
        canPreviewtab_horizontal_container.addWidget(self.entryCanPreview)
        
        self.buttonCanPreview = QPushButton("Show candidate")
        canPreviewtab_horizontal_container.addWidget(self.buttonCanPreview)

        #Give it a function on click:
        self.buttonCanPreview.clicked.connect(lambda: self.updateCandidatePreview())

        #Add a horizontal layout to display info about the cluster
        self.canPreviewtab_horizontal_container2 = QHBoxLayout()
        canPreviewtab_vertical_container.addLayout(self.canPreviewtab_horizontal_container2)
        self.candidate_info = QLabel('')
        self.canPreviewtab_horizontal_container2.addWidget(self.candidate_info)

        #Create an empty figure and store it as self.data:
        self.data['figurePlot3D'] = plt.figure(figsize=(6.8, 4))
        self.data['figurePlot3D'].suptitle('3D pointcloud of candidate cluster')
        self.data['figureAx3D'] = self.data['figurePlot3D'].add_subplot(111, projection='3D')
        self.data['figureCanvas3D'] = FigureCanvas(self.data['figurePlot3D'])
        self.data['figurePlot3D'].tight_layout()
        self.data['figurePlot3D'].subplots_adjust(top=0.95)
        
        #Add a navigation toolbar (zoom, pan etc)
        canPreviewtab_vertical_container.addWidget(NavigationToolbar(self.data['figureCanvas3D'], self))
        #Add the canvas to the tab
        canPreviewtab_vertical_container.addWidget(self.data['figureCanvas3D'])

        #Create a second empty figure and store it as self.data:
        self.data['figurePlotProjection'] = plt.figure(figsize=(6.8, 4))
        self.data['figurePlotProjection'].suptitle('2D projections of candidate cluster')
        self.data['figureAxProjectionXY'] = self.data['figurePlotProjection'].add_subplot(121)
        self.data['figureAxProjectionXT'] = self.data['figurePlotProjection'].add_subplot(222)
        self.data['figureAxProjectionYT'] = self.data['figurePlotProjection'].add_subplot(224)
        self.data['figureCanvasProjection'] = FigureCanvas(self.data['figurePlotProjection'])
        self.data['figurePlotProjection'].tight_layout()
        
        #Add a navigation toolbar (zoom, pan etc)
        canPreviewtab_vertical_container.addWidget(NavigationToolbar(self.data['figureCanvasProjection'], self))
        #Add the canvas to the tab
        canPreviewtab_vertical_container.addWidget(self.data['figureCanvasProjection'])


    def updateCandidatePreview(self):
        """
        Function that's called when the button to show the candidate is clicked
        """

        # Clear all plots
        self.candidate_info.setText('')

        self.data['figureAx3D'].clear()
        self.data['figureCanvas3D'].draw()

        self.data['figureAxProjectionXY'].clear()
        self.data['figureAxProjectionXT'].clear()
        self.data['figureAxProjectionYT'].clear()
        self.data['figureAxProjectionXT'].set_aspect('auto')
        self.data['figureAxProjectionYT'].set_aspect('auto')
        self.data['figureCanvasProjection'].draw()

        # Check the candidate entry field
        if self.entryCanPreview.text()=='':
            self.candidate_info.setText('Tried to visualise candidate, but no ID was given!')
            logging.error('Tried to visualise candidate, but no ID was given!')

        elif 'FindingMethod' in self.data and int(self.entryCanPreview.text()) < len(self.data['FindingResult'][0]):
            logging.debug(f"Attempting to show candidate {self.data['CandidatePreviewID']}.")

            # Get some info about the candidate
            self.data['CandidatePreviewID'] = int(self.entryCanPreview.text())
            N_events = self.data['FindingResult'][0][self.data['CandidatePreviewID']]['N_events']
            cluster_size = self.data['FindingResult'][0][self.data['CandidatePreviewID']]['cluster_size']
            self.candidate_info.setText(f"This candidate cluster contains {N_events} events and has dimensions ({cluster_size[0]}, {cluster_size[1]}, {cluster_size[2]}).")

            # Do a 3d scatterplot of the event data
            self.data['figureAx3D'].scatter(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['x'], self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['y'], self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['t']*1e-3)
            self.data['figureAx3D'].set_xlabel('x [px]')
            self.data['figureAx3D'].set_ylabel('y [px]')
            self.data['figureAx3D'].set_zlabel('t [ms]')

            # Give it a nice layout
            self.data['figurePlot3D'].tight_layout()
            self.data['figurePlot3D'].subplots_adjust(top=0.95)

            # Update drawing of the canvas
            self.data['figureCanvas3D'].draw()
            logging.info(f"3D scatter plot of candidate {self.data['CandidatePreviewID']} drawn.")

            # Get xyt-limits of candidate events
            self.xlim = [np.min(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['x']), np.max(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['x'])]
            self.ylim = [np.min(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['y']), np.max(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['y'])]
            self.tlim = [np.min(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['t']*1e-3), np.max(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['t']*1e-3)]

            xy_bin_width = 1 # in px
            t_bin_width = 10. # in ms

            # calculate number of bins
            x_bins = int((self.xlim[1]-self.xlim[0])/xy_bin_width)
            y_bins = int((self.ylim[1]-self.ylim[0])/xy_bin_width)
            t_bins = int((self.tlim[1]-self.tlim[0])/t_bin_width)

            # Compute the 2D histograms
            hist_xy, x_edges, y_edges = np.histogram2d(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['x'], self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['y'], bins=(x_bins, y_bins))
            hist_tx, t_edges, x_edges = np.histogram2d(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['t']*1e-3, self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['x'], bins=(t_bins, x_bins))
            hist_ty, t_edges, y_edges = np.histogram2d(self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['t']*1e-3, self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events']['y'], bins=(t_bins, y_bins))

            # Set goodlooking aspect ratio depending on nr of xyt-bins
            aspectty = 3. * (len(t_edges)-1) / (len(y_edges)-1)
            aspecttx = 3. * (len(t_edges)-1) / (len(x_edges)-1)

            # Plot the 2D histograms
            self.data['figureAxProjectionXY'].imshow(hist_xy.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='equal', interpolation='none')
            self.data['figureAxProjectionXT'].imshow(hist_tx.T, extent=[t_edges[0], t_edges[-1], x_edges[0], x_edges[-1]], origin='lower', aspect=aspecttx, interpolation='none')
            self.data['figureAxProjectionYT'].imshow(hist_ty.T, extent=[t_edges[0], t_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect=aspectty, interpolation='none')
            
            # Add and set labels
            self.data['figureAxProjectionXY'].set_xlabel('x [px]')
            self.data['figureAxProjectionXY'].set_ylabel('y [px]')
            self.data['figureAxProjectionXT'].set_ylabel('x [px]')
            self.data['figureAxProjectionYT'].set_ylabel('y [px]')
            self.data['figureAxProjectionYT'].set_xlabel('t [ms]')
            
            # Give it a nice layout
            self.data['figurePlotProjection'].tight_layout()

            #U pdate drawing of the canvas
            self.data['figureCanvasProjection'].draw()
            logging.info(f"2D event-projections of candidate {self.data['CandidatePreviewID']} drawn.")

        else: 
            self.candidate_info.setText('Tried to visualise candidate but no data found!')
            logging.error('Tried to visualise candidate but no data found!')

    
    def setup_previewTab(self):
        """
        Function that creates the preview tab
        Very practically, it creates 10 figures, stores those in memory, and displays them when necessary
        """
        #It's a grid layout
        self.previewtab_layout = QGridLayout()
        self.tab_previewVis.setLayout(self.previewtab_layout)
        
        #It requires global variables for the color-bar limits (to allow the same limits for all images of the preview)
        self.PreviewMinCbarVal = 0
        self.PreviewMaxCbarVal = 0
        
        #It creates the ImageSlider class
        self.previewImage_slider = ImageSlider([],self)
        self.previewtab_layout.addWidget(self.previewImage_slider)
               
    def updateShowPreview(self,previewEvents=None):
        """
        Function that's called to update the preview (or show it). Requires the previewEvents, or uses the self.previewEvents.
        Hardcoded to show the events as 100-ms-time bins.
        """
        if previewEvents is None:
            previewEvents = self.previewEvents
        #Idea: create some preview image and highlight found clusters and localizations.
        self.allPreviewFigures = []
        #For now: create 5 plots:
        self.PreviewFig = {}
        self.PreviewFrameTime = 100*1000 #in us - maybe user-definable later?
        
        #Obtain the numbers of frames that are displayed
        nrFramesDisplay = int(np.ceil((max(previewEvents['t']))/self.PreviewFrameTime))
        
        #Initialise minimum, maximum colorbar values:
        self.PreviewMinCbarVal = 99
        self.PreviewMaxCbarVal = -99
        
        #Loop over the frames:
        for i in range(0,nrFramesDisplay):
            #Create an empty figure
            self.PreviewFig[i] = plt.figure()
            #Create an empty array with the sizes of self.previewEvents:
            frameBasedEvent2dArray = np.zeros((max(previewEvents['y'])+1,max(previewEvents['x'])+1))
            #Get the events belonging to this frame:
            eventsInThisFrame = previewEvents[previewEvents['t'] >= i*self.PreviewFrameTime]
            eventsInThisFrame = eventsInThisFrame[eventsInThisFrame['t'] < (i+1)*self.PreviewFrameTime]
            
            #Loop over the events and fill the corresponding pixel in frameBasedEvent2dArray:
            for j in range(0,len(eventsInThisFrame)):
                if eventsInThisFrame[j]['p'] == 0:
                    frameBasedEvent2dArray[eventsInThisFrame[j]['y'],eventsInThisFrame[j]['x']] -= 1
                else:
                    frameBasedEvent2dArray[eventsInThisFrame[j]['y'],eventsInThisFrame[j]['x']] += 1
            
            #Show this in plt as an imshow - we never call plt.show(), so it never really shows:
            fig = self.PreviewFig[i].add_subplot(111)
            fig.imshow(frameBasedEvent2dArray)
            #Set axis limits correctly:
            fig.set_xlim(min(previewEvents['x']),max(previewEvents['x']))
            fig.set_ylim(min(previewEvents['y']),max(previewEvents['y']))
            
            #Find the 'finding' results in this time-frame
            indices = [l for l in range(len(self.data['FindingResult'][0]))]
            for indexv in indices:
                try:
                    #Plot it as a rectangle with a magenta, red, or cyan color, depending on it starting, middling, or ending on this frame.
                    row = self.data['FindingResult'][0][indexv]
                    if self.previewEventStartedOnThisFrame(row,i):
                        # Create a Rectangle object
                        self.createRectangle(fig,row['events'],'m')
                        self.showText(fig,row['events'],str(indexv),'m')
                    elif self.previewEventEndsOnThisFrame(row,i):
                        self.createRectangle(fig,row['events'],'c')
                        self.showText(fig,row['events'],str(indexv),'c')
                    elif self.previewEventHappensOnThisFrame(row,i):
                        self.createRectangle(fig,row['events'],'r')
                        self.showText(fig,row['events'],str(indexv),'r')
                except:
                    pass
                
                try:
                    #Also add the corresponding fitting result
                    try:
                        #Check if the localization is on this frame
                        localization = self.data['FittingResult'][0].iloc[indexv-1]
                        if localization['t']*1000 >= i*self.PreviewFrameTime and localization['t']*1000 < (i+1)*self.PreviewFrameTime:
                            #If so, add a cross
                            fig.plot(localization['x']/self.globalSettings['PixelSize_nm']['value'],localization['y']/self.globalSettings['PixelSize_nm']['value'],'rx', alpha=0.5)
                    except:
                        breakpoint
                except:
                    pass
                           
            self.allPreviewFigures.append(self.PreviewFig[i])
            plt.close()
            
            #Check cbar values:
            #Check the 1st percentile:
            pctile = 1
            if np.percentile(frameBasedEvent2dArray,pctile) < self.PreviewMinCbarVal:
                self.PreviewMinCbarVal = np.percentile(frameBasedEvent2dArray,pctile)
            if np.percentile(frameBasedEvent2dArray,(100-pctile)) > self.PreviewMaxCbarVal:
                self.PreviewMaxCbarVal = np.percentile(frameBasedEvent2dArray,(100-pctile))

        #Create a new ImageSlider and update the figure
        self.previewImage_sliderNew = ImageSlider(parent=self,figures=self.allPreviewFigures)
        self.previewImage_slider.update_figures(self.allPreviewFigures)
        
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
                
    def changeLayout_choice(self,curr_layout,className,displayNameToFunctionNameMap):
        logging.debug('Changing layout'+curr_layout.parent().objectName())
        #This removes everything except the first entry (i.e. the drop-down menu)
        self.resetLayout(curr_layout,className)
        #Get the dropdown info
        curr_dropdown = self.getMethodDropdownInfo(curr_layout,className)
        #Get the kw-arguments from the current dropdown.
        current_selected_function = utils.functionNameFromDisplayName(curr_dropdown.currentText(),displayNameToFunctionNameMap)
        reqKwargs = utils.reqKwargsFromFunction(current_selected_function)
        #Add a widget-pair for every kw-arg
        labelposoffset = 0
        for k in range(len(reqKwargs)):
            #Value is used for scoring, and takes the output of the method
            if reqKwargs[k] != 'methodValue':
                label = QLabel(f"<b>{reqKwargs[k]}</b>")
                label.setObjectName(f"Label#{current_selected_function}#{reqKwargs[k]}")
                if self.checkAndShowWidget(curr_layout,label.objectName()) == False:
                    label.setToolTip(utils.infoFromMetadata(current_selected_function,specificKwarg=reqKwargs[k]))
                    curr_layout.addWidget(label,2+(k)+labelposoffset,0)
                line_edit = QLineEdit()
                line_edit.setObjectName(f"LineEdit#{current_selected_function}#{reqKwargs[k]}")
                defaultValue = utils.defaultValueFromKwarg(current_selected_function,reqKwargs[k])
                if self.checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                    line_edit.setToolTip(utils.infoFromMetadata(current_selected_function,specificKwarg=reqKwargs[k]))
                    if defaultValue is not None:
                        line_edit.setText(str(defaultValue))
                    curr_layout.addWidget(line_edit,2+k+labelposoffset,1)
                    #Add a on-change listener:
                    line_edit.textChanged.connect(lambda text,line_edit=line_edit: self.kwargValueInputChanged(line_edit))
            else:
                labelposoffset -= 1
            
        #Get the optional kw-arguments from the current dropdown.
        optKwargs = utils.optKwargsFromFunction(current_selected_function)
        #Add a widget-pair for every kwarg
        for k in range(len(optKwargs)):
            label = QLabel(f"<i>{optKwargs[k]}</i>")
            label.setObjectName(f"Label#{current_selected_function}#{optKwargs[k]}")
            if self.checkAndShowWidget(curr_layout,label.objectName()) == False:
                label.setToolTip(utils.infoFromMetadata(current_selected_function,specificKwarg=optKwargs[k]))
                curr_layout.addWidget(label,2+(k)+len(reqKwargs)+labelposoffset,0)
            line_edit = QLineEdit()
            line_edit.setObjectName(f"LineEdit#{current_selected_function}#{optKwargs[k]}")
            defaultValue = utils.defaultValueFromKwarg(current_selected_function,optKwargs[k])
            if self.checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                line_edit.setToolTip(utils.infoFromMetadata(current_selected_function,specificKwarg=optKwargs[k]))
                if defaultValue is not None:
                    line_edit.setText(str(defaultValue))
                curr_layout.addWidget(line_edit,2+(k)+len(reqKwargs)+labelposoffset,1)
                #Add a on-change listener:
                line_edit.textChanged.connect(lambda text,line_edit=line_edit: self.kwargValueInputChanged(line_edit))
    
    def kwargValueInputChanged(self,line_edit):
        #Get the function name
        function = line_edit.objectName().split("#")[1]
        #Get the kwarg
        kwarg = line_edit.objectName().split("#")[2]
        #Get the value
        value = line_edit.text()
        expectedType = utils.typeFromKwarg(function,kwarg)
        if expectedType is not None:
            if expectedType is not str:
                try:
                    value = eval(line_edit.text())
                    if isinstance(value,expectedType):
                        self.setLineEditStyle(line_edit,type='Normal')
                    else:
                        self.setLineEditStyle(line_edit,type='Warning')
                except:
                    #Show as warning
                    self.setLineEditStyle(line_edit,type='Warning')
            elif expectedType is str:
                try:
                    value = str(line_edit.text())
                    self.setLineEditStyle(line_edit,type='Normal')
                except:
                    #Show as warning
                    self.setLineEditStyle(line_edit,type='Warning')
        else:
            self.setLineEditStyle(line_edit,type='Normal')
        pass
    
    def setLineEditStyle(self,line_edit,type='Normal'):
        if type == 'Normal':
            line_edit.setStyleSheet("border: 1px  solid #D5D5E5;")
        elif type == 'Warning':
            line_edit.setStyleSheet("border: 1px solid red;")
    
    def checkAndShowWidget(self,layout, widgetName):
        # Iterate over the layout's items
        for index in range(layout.count()):
            item = layout.itemAt(index)
            # Check if the item is a widget
            if item.widget() is not None:
                widget = item.widget()
                # Check if the widget has the desired name
                if widget.objectName() == widgetName:
                    # Widget already exists, unhide it
                    widget.show()
                    return
        return False
                
    #Remove everythign in this layout except className_dropdown
    def resetLayout(self,curr_layout,className):
        for index in range(curr_layout.count()):
            widget_item = curr_layout.itemAt(index)
            # Check if the item is a widget (as opposed to a layout)
            if widget_item.widget() is not None:
                widget = widget_item.widget()
                #If it's the dropdown segment, label it as such
                if not ("candidateFindingDropdown" in widget.objectName()) and not ("candidateFittingDropdown" in widget.objectName()) and widget.objectName() != f"titleLabel_{className}" and not ("KEEP" in widget.objectName()):
                    logging.debug(f"Hiding {widget.objectName()}")
                    widget.hide()
    
    def getMethodDropdownInfo(self,curr_layout,className):
        curr_dropdown = []
        #Look through all widgets in the current layout
        for index in range(curr_layout.count()):
            widget_item = curr_layout.itemAt(index)
            #Check if it's fair to check
            if widget_item.widget() is not None:
                widget = widget_item.widget()
                #If it's the dropdown segment, label it as such
                if (className in widget.objectName()) and ("Dropdown" in widget.objectName()):
                    curr_dropdown = widget
        #Return the dropdown
        return curr_dropdown
    
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
            npyevents = np.load(dataLocation, mmap_mode='r')
            #Select xytp area specified:
            npyevents = self.filterEvents_npy_t(npyevents,tStretch=(float(self.run_startTLineEdit.text()),float(self.run_durationTLineEdit.text())))
            npyevents = self.filterEvents_xy(npyevents,xyStretch=(float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text()),float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())))
                
            return npyevents
        elif dataLocation.endswith('.raw'):
            events = self.RawToNpy(dataLocation)
            #Select xytp area specified:
            events = self.filterEvents_npy_t(events,tStretch=(float(self.run_startTLineEdit.text()),float(self.run_durationTLineEdit.text())))
            events = self.filterEvents_xy(events,xyStretch=(float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text()),float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())))
            
            return events
    
    def RawToNpy(self,filepath,buffer_size = 1e8,time_batches = 50e3):        
        if(os.path.exists(filepath[:-4]+'.npy')):
            events = np.load(filepath[:-4]+'.npy')
            logging.info('NPY file from RAW was already present, loading this instead of RAW!')
        else:
            logging.info('Starting to convert NPY to RAW...')
            sys.path.append(self.globalSettings['MetaVisionPath']['value']) 
            from metavision_core.event_io.raw_reader import RawReader
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
    
    def run_processing(self):
        thread = threading.Thread(target=self.run_processing_i)
        thread.start()
    
    def run_processing_i(self):
        #Check if a folder or file is selected:
        if os.path.isdir(self.dataLocationInput.text()):
            #Find all .raw or .npy files that have unique names except for the extension (and prefer .npy):
            allFiles = self.find_raw_npy_files(self.dataLocationInput.text())
            logging.debug('Running folder analysis on files :')
            logging.debug(allFiles)
            for file in allFiles:
                try:
                    logging.info('Starting to process file '+file)
                    self.processSingleFile(file)
                    logging.info('Successfully processed file '+file)
                except:
                    logging.error('Error in processing file '+file)
        #If it's a file...
        elif os.path.isfile(self.dataLocationInput.text()):
            #Check if we are loading existing finding
            if 'ExistingFinding' in self.candidateFindingDropdown.currentText():
                logging.info('Skipping finding processing, going to fitting')
                
                #Ensure that we don't store the finding result
                origStoreFindingSetting=self.globalSettings['StoreFindingOutput']['value']
                self.globalSettings['StoreFindingOutput']['value'] = False
                
                self.processSingleFile(self.dataLocationInput.text(),onlyFitting=True)
                
                #Reset the global setting:
                self.globalSettings['StoreFindingOutput']['value']=origStoreFindingSetting
            else:
                #Otherwise normally process a single file
                self.processSingleFile(self.dataLocationInput.text())
         
        self.updateGUIafterNewResults()       
        return
        
    def updateGUIafterNewResults(self):
        self.updateLocList()
    
    def updateLocList(self):
        #data is stored in self.data['FittingResult'][0]
        #fill the self.LocListTable QTableWidget with the data:
        
        #Get the shape of the data
        nrRows = np.shape(self.data['FittingResult'][0])[0]
        nrColumns = np.shape(self.data['FittingResult'][0])[1]
        
        #Give the loclisttable the correct row/column count:
        self.LocListTable.setRowCount(nrRows)
        self.LocListTable.setColumnCount(nrColumns)
        
        #Fill the loclisttable with the output:
        for r in range(nrRows):
            for c in range(nrColumns):
                nrDigits = 2
                item = QTableWidgetItem(f"{round(self.data['FittingResult'][0].iloc[r, c], nrDigits):.{nrDigits}f}")
                self.LocListTable.setItem(r, c, item)
        
        #Add headers
        self.LocListTable.setHorizontalHeaderLabels(self.data['FittingResult'][0].columns.tolist())
        
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
    
    def processSingleFile(self,FileName,onlyFitting=False):

        #Runtime of finding and fitting
        self.currentFileInfo['FindingTime'] = 0
        self.currentFileInfo['FittingTime'] = 0

        if not onlyFitting:
            #Run the analysis on a single file
            self.currentFileInfo['CurrentFileLoc'] = FileName
            if self.globalSettings['FindingBatching']['value']== False:
                npyData = self.loadRawData(FileName)
                if npyData is None:
                    return
                
                #Check polarity
                self.checkPolarity(npyData)
                
                #Sort event list on time
                npyData = npyData[np.argsort(npyData,order='t')]
                
                #Run finding/fitting
                self.runFindingAndFitting(npyData)
            elif self.globalSettings['FindingBatching']['value']== True or self.globalSettings['FindingBatching']['value']== 2:
                self.runFindingBatching()
            
        #If we only fit, we still run more or less the same info, butwe don't care about the npyData in the CurrentFileLoc.
        elif onlyFitting:
            self.currentFileInfo['CurrentFileLoc'] = FileName
            logging.info('Candidate finding NOT performed')
            npyData = None
            self.runFindingAndFitting(npyData)
        
    
    def FindingBatching(self,npyData):
        #For now, print start and final time:
        logging.info(self.chunckloading_currentLimits)
        logging.info('Start time: '+str(npyData[0]['t']))
        logging.info('End time: '+str(npyData[-1]['t']))
        
        FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings")
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
        
    
    def runFindingBatching(self):
        logging.info('Batching-dependant finding starting!')
        fileToRun = self.currentFileInfo['CurrentFileLoc']
        self.currentFileInfo['FindingTime'] = time.time()
        self.data['FindingResult'] = {}
        self.data['FindingResult'][0] = {}
        self.data['FindingResult'][1] = []
        #For batching, we only load data between some timepoints.
        if fileToRun.endswith('.raw'):
            sys.path.append(self.globalSettings['MetaVisionPath']['value']) 
            from metavision_core.event_io.raw_reader import RawReader
            record_raw = RawReader(fileToRun)
            
            
            #FSeek to start time according to specifications
            record_raw.seek_time(float(self.run_startTLineEdit.text())*1000)
            
            #Read all chunks:
            self.chunckloading_number_chuck = 0
            events_prev = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16})
            self.chunckloading_finished_chunking = False
            self.chunckloading_currentLimits = [[0,0],[0,0]]
            
            while self.chunckloading_finished_chunking == False:
                if self.chunckloading_number_chuck == 0:
                    events = record_raw.load_delta_t(float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000)
                else:
                    events = record_raw.load_delta_t(float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000)
                if len(events) > 0:
                    logging.info('New chunk analysis starting') 
                    #limit to requested xy
                    events = self.filterEvents_xy(events,xyStretch=(float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text()),float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())))
                    #limit to requested t
                    events = self.filterEvents_npy_t(events,tStretch=(-np.Inf,(float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text())))*1000)
                    
                    self.chunckloading_currentLimits = [[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000],[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000-float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000]]
                    #Add events_prev before these events:
                    events = np.concatenate((events_prev,events))
                    if len(events) > 0:
                        self.FindingBatching(events)
                        
                        events_prev = events[events['t']>((self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000-float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000)]
                    self.chunckloading_number_chuck += 1
                else:
                    logging.info('Finished chunking!')
                    self.chunckloading_finished_chunking = True
                    
        elif self.dataLocationInput.text().endswith('.npy'):
            #For npy, load the events in memory
            data = np.load(self.dataLocationInput.text(), mmap_mode='r')
            dataset_minx = data['x'].min()
            dataset_miny = data['y'].min()
            
            if float(self.run_startTLineEdit.text()) > data['t'].min()/1000 or float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text()) < data['t'].max()/1000:
                #Fully limit to t according to specifications
                data = self.filterEvents_npy_t(data,tStretch=(float(self.run_startTLineEdit.text()),float(self.run_durationTLineEdit.text())))
            
                #set start time to zero (required for chunking):
                data['t'] -= min(data['t'])
            
            #Read all chunks:
            self.chunckloading_number_chuck = 0
            self.chunckloading_finished_chunking = False
            self.chunckloading_currentLimits = [[0,0],[0,0]]
            while self.chunckloading_finished_chunking == False:
                
                self.chunckloading_currentLimits = [[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000],[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000-float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000]]
                
                indices = np.logical_and((data['t'] >= self.chunckloading_currentLimits[1][0]), (data['t'] <= self.chunckloading_currentLimits[1][1]))
                if sum(indices) > 0:
                    # Access the partial data using the indices
                    events = data[indices]
                    
                    #Filter to xy
                    events = self.filterEvents_xy(events,xyStretch=(dataset_minx+float(self.run_minXLineEdit.text()),dataset_minx+float(self.run_maxXLineEdit.text()),dataset_miny+float(self.run_minYLineEdit.text()),dataset_miny+float(self.run_maxYLineEdit.text())))
                    #Run partial on this:
                    if len(events) == 0:
                        logging.warning('Prematurely finished chunking!')
                        pass
                    self.FindingBatching(events)
                    self.chunckloading_number_chuck+=1
                else:
                    logging.info('Finished chunking!')
                    self.chunckloading_finished_chunking = True
        else:
            logging.error("Please choose a .raw or .npy file for previews.")
            return
        
        #Check if some candidates are found:
        if len(self.data['FindingResult'][0]) == 0:
            logging.error("No candidates found at all! Stopping analysis.")
            return
        
        #store finding results:
        if self.globalSettings['StoreFindingOutput']['value']:
            self.storeFindingOutput()
        self.currentFileInfo['FindingTime'] = time.time() - self.currentFileInfo['FindingTime']
        logging.info('Number of candidates found: '+str(len(self.data['FindingResult'][0])))
        logging.info('Candidate finding took '+str(self.currentFileInfo['FindingTime'])+' seconds.')
        
        #All finding is done, continue with fitting:
        self.runFitting()
        
        # self.previewEvents = events
        
        # #Edit values for x,y coordinates:
        # #First check if all entries in array are numbers:
        # try:
        #     int(xyStretch[0])
        #     int(xyStretch[1])
        #     int(xyStretch[2])
        #     int(xyStretch[3])
        #     if not all(isinstance(x, int) for x in [int(xyStretch[0]),int(xyStretch[1]),int(xyStretch[2]),int(xyStretch[3])]):
        #         logging.info("No XY cutting in preview due to not all entries being integers.")
        #     else:
        #         logging.info("XY cutting in preview to values: "+str(xyStretch[0])+","+str(xyStretch[1])+","+str(xyStretch[2])+","+str(xyStretch[3]))
        #         #Filter on x,y coordinates:
        #         events = events[(events['x'] >= int(xyStretch[0])) & (events['x'] <= int(xyStretch[1]))]
        #         events = events[(events['y'] >= int(xyStretch[2])) & (events['y'] <= int(xyStretch[3]))]
        # except:
        #      logging.info("No XY cutting in preview due to not all entries being integers-.")
             
    
    
    def runFindingAndFitting(self,npyData):
        #Run the finding function!
        FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings")
        if FindingEvalText is not None:
            self.currentFileInfo['FindingTime'] = time.time()
            self.data['FindingMethod'] = str(FindingEvalText)
            self.data['FindingResult'] = eval(str(FindingEvalText))
            self.currentFileInfo['FindingTime'] = time.time() - self.currentFileInfo['FindingTime']
            logging.info('Number of candidates found: '+str(len(self.data['FindingResult'][0])))
            logging.info('Candidate finding took '+str(self.currentFileInfo['FindingTime'])+' seconds.')
            logging.info('Candidate finding done!')
            logging.debug(self.data['FindingResult'])
            if self.globalSettings['StoreFindingOutput']['value']:
                self.storeFindingOutput()
            #And run the fitting
            self.runFitting()
        else:
            logging.error('Candidate finding NOT performed')
            
    def runFitting(self):
        if self.data['FindingResult'][0] is not None:
            #Run the finding function!
            FittingEvalText = self.getFunctionEvalText('Fitting',"self.data['FindingResult'][0]","self.globalSettings")
            if FittingEvalText is not None:
                self.currentFileInfo['FittingTime'] = time.time()
                self.data['FittingMethod'] = str(FittingEvalText)
                self.data['FittingResult'] = eval(str(FittingEvalText))
                self.currentFileInfo['FittingTime'] = time.time() - self.currentFileInfo['FittingTime']
                logging.info('Number of localizations found: '+str(len(self.data['FittingResult'][0])))
                logging.info('Candidate fitting took '+str(self.currentFileInfo['FittingTime'])+' seconds.')
                logging.info('Candidate fitting done!')
                logging.debug(self.data['FittingResult'])
                if len(self.data['FittingResult'][0]) == 0:
                    logging.error('No localizations found after fitting!')
                    return
                #Create and store the metadata
                if self.globalSettings['StoreFileMetadata']['value']:
                    self.createAndStoreFileMetadata()
                
                if self.globalSettings['StoreFinalOutput']['value']:
                    self.storeLocalizationOutput()
            else:
                logging.error('Candidate fitting NOT performed')
        else:
            logging.error('No Finding Result obtained! Fitting is not run succesfully!')
                      
    def getStoreLocationPartial(self):   
        if 'ExistingFinding' in self.data['FindingMethod']:
            FindingResultFileName = self.data['FindingMethod'][self.data['FindingMethod'].index('File_Location="')+len('File_Location="'):self.data['FindingMethod'].index('")')]
            storeLocationPartial = FindingResultFileName[:-7]
        else:
            storeLocationPartial = self.currentFileInfo['CurrentFileLoc'][:-4]
        return storeLocationPartial     
      
    def storeLocalizationOutput(self):
        logging.debug('Attempting to store fitting results output')
        storeLocation = self.getStoreLocationPartial()+'_FitResults_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
        #Store the localization output
        if self.globalSettings['OutputDataFormat']['value'] == 'minimal':
            self.data['FittingResult'][0].to_csv(storeLocation)
        elif self.globalSettings['OutputDataFormat']['value'] == 'thunderstorm':
            #Add a frame column to fittingResult:
            self.data['FittingResult'][0]['frame'] = self.data['FittingResult'][0]['t'].apply(round).astype(int)
            self.data['FittingResult'][0]['frame'] -= min(self.data['FittingResult'][0]['frame'])-1
            #Create thunderstorm headers
            headers = list(self.data['FittingResult'][0].columns)
            headers = ['\"x [nm]\"' if header == 'x' else '\"y [nm]\"' if header == 'y' else '\"z [nm]\"' if header == 'z' else '\"t [ms]\"' if header == 't' else header for header in headers]
            self.data['FittingResult'][0].rename_axis('\"id\"').to_csv(storeLocation, header=headers, quoting=csv.QUOTE_NONE)
        else:
            #default to minimal
            self.data['FittingResult'][0].to_csv(storeLocation)
        logging.info('Fitting results output stored')
        
    def storeFindingOutput(self):
        logging.debug('Attempting to store finding results output')
        #Store the Finding results output
        # np.save(self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.npy',self.data['FindingResult'][0])
        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.pickle'
        with open(file_path, 'wb') as file:
            pickle.dump(self.data['FindingResult'][0], file)
        
        logging.info('Finding results output stored')
    
    def createAndStoreFileMetadata(self):
        logging.debug('Attempting to create and store file metadata')
        try:
            metadatastring = f"""Metadata information for file {self.currentFileInfo['CurrentFileLoc']}
Analysis routine finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---- Finding metadata output: ----
Methodology used:
{self.data['FindingMethod']}

Number of candidates found: {len(self.data['FindingResult'][0])}
Candidate finding took {self.currentFileInfo['FindingTime']} seconds.

Custom output from finding function:
{self.data['FindingResult'][1]}

---- Fitting metadata output: ----
Methodology used:
{self.data['FittingMethod']}

Number of localizations found: {len(self.data['FittingResult'][0])}
Candidate fitting took {self.currentFileInfo['FittingTime']} seconds.

Custom output from fitting function:
{self.data['FittingResult'][1]}
            """
            #Store this metadatastring:
            with open(self.getStoreLocationPartial()+'_RunInfo_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.txt', 'w') as f:
                f.write(metadatastring)
            logging.info('File metadata created and stored')
        except:
            logging.error('Error in creating file metadata, not stored')
    
    def getFunctionEvalText(self,className,p1,p2):
        #Get the dropdown info
        moduleMethodEvalTexts = []
        all_layouts = self.findChild(QWidget, "groupbox"+className).findChildren(QLayout)[0]
        
        
        methodKwargNames_method = []
        methodKwargValues_method = []
        methodName_method = ''
        # Iterate over the items in the layout
        for index in range(all_layouts.count()):
            item = all_layouts.itemAt(index)
            widget = item.widget()
                        
            if ("LineEdit" in widget.objectName()) and widget.isVisibleTo(self.tab_processing):
                # The objectName will be along the lines of foo#bar#str
                #Check if the objectname is part of a method or part of a scoring
                split_list = widget.objectName().split('#')
                methodName_method = split_list[1]
                methodKwargNames_method.append(split_list[2])
                
                #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                methodKwargValues_method.append(widget.text().replace('\\','/'))

        #If at this point there is no methodName_method, it means that the method has exactly 0 req or opt kwargs. Thus, we simply find the value of the QComboBox which should be the methodName:
        if methodName_method == '':
            for index in range(all_layouts.count()):
                item = all_layouts.itemAt(index)
                widget = item.widget()
                if isinstance(widget,QComboBox) and widget.isVisibleTo(self.tab_processing) and className in widget.objectName():
                    if className == 'Finding':
                        methodName_method = utils.functionNameFromDisplayName(widget.currentText(),self.Finding_functionNameToDisplayNameMapping)
                    elif className == 'Fitting':
                        methodName_method = utils.functionNameFromDisplayName(widget.currentText(),self.Fitting_functionNameToDisplayNameMapping)
        
        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = self.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method,partialStringStart=str(p1)+','+str(p2))
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)
        if moduleMethodEvalTexts is not None and len(moduleMethodEvalTexts) > 0:
            return moduleMethodEvalTexts[0]
        else:
            return None
                
    def getEvalTextFromGUIFunction(self, methodName, methodKwargNames, methodKwargValues, partialStringStart=None, removeKwargs=None):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #methodName: the physical name of the method, i.e. StarDist.StarDistSegment
    #methodKwargNames: found kwarg NAMES from the GUI
    #methodKwargValues: found kwarg VALUES from the GUI
    #methodTypeString: type of method, i.e. 'function Type' (e.g. CellSegmentScripts, CellScoringScripts etc)'
    #Optionals: partialStringStart: gives a different start to the partial eval-string
    #Optionals: removeKwargs: removes kwargs from assessment (i.e. for scoring script, where this should always be changed by partialStringStart)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        specialcaseKwarg = [] #Kwarg where the special case is used
        specialcaseKwargPartialStringAddition = [] #text to be eval-ed in case this kwarg is found
        #We have the method name and all its kwargs, so:
        if len(methodName)>0: #if the method exists
            #Check if all req. kwargs have some value
            reqKwargs = utils.reqKwargsFromFunction(methodName)
            #Remove values from this array if wanted
            if removeKwargs is not None:
                for removeKwarg in removeKwargs:
                    if removeKwarg in reqKwargs:
                        reqKwargs.remove(removeKwarg)
                    else:
                        #nothing, but want to make a note of this (log message)
                        reqKwargs = reqKwargs
            #Stupid dummy-check whether we have the reqKwargs in the methodKwargNames, which we should (basically by definition)
            if all(elem in set(methodKwargNames) for elem in reqKwargs):
                allreqKwargsHaveValue = True
                for id in range(0,len(reqKwargs)):
                    #First find the index of the function-based reqKwargs in the GUI-based methodKwargNames. They should be the same, but you never know
                    GUIbasedIndex = methodKwargNames.index(reqKwargs[id])
                    #Get the value of the kwarg - we know the name already now due to reqKwargs.
                    kwargvalue = methodKwargValues[GUIbasedIndex]
                    if kwargvalue == '':
                        allreqKwargsHaveValue = False
                        logging.error(f'Missing required keyword argument in {methodName}: {reqKwargs[id]}, NOT CONTINUING')
                if allreqKwargsHaveValue:
                    #If we're at this point, all req kwargs have a value, so we can run!
                    #Get the string for the required kwargs
                    if partialStringStart is not None:
                        partialString = partialStringStart
                    else:
                        partialString = ''
                    for id in range(0,len(reqKwargs)):
                        #First find the index of the function-based reqKwargs in the GUI-based methodKwargNames. They should be the same, but you never know
                        GUIbasedIndex = methodKwargNames.index(reqKwargs[id])
                        #Get the value of the kwarg - we know the name already now due to reqKwargs.
                        kwargvalue = methodKwargValues[GUIbasedIndex]
                        #Add a comma if there is some info in the partialString already
                        if partialString != '':
                            partialString+=","
                        #Check for special requests of kwargs, this is normally used when pointing to the output of a different value
                        if reqKwargs[id] in specialcaseKwarg:
                            #Get the index
                            ps_index = specialcaseKwarg.index(reqKwargs[id])
                            #Change the partialString with the special case
                            partialString+=eval(specialcaseKwargPartialStringAddition[ps_index])
                        else:
                            partialString+=reqKwargs[id]+"=\""+kwargvalue+"\""
                    #Add the optional kwargs if they have a value
                    optKwargs = utils.optKwargsFromFunction(methodName)
                    for id in range(0,len(optKwargs)):
                        if methodKwargValues[id+len(reqKwargs)] != '':
                            if partialString != '':
                                partialString+=","
                            partialString+=optKwargs[id]+"=\""+methodKwargValues[id+len(reqKwargs)]+"\""
                    segmentEval = methodName+"("+partialString+")"
                    return segmentEval
                else:
                    logging.error('NOT ALL KWARGS PROVIDED!')
                    return None
            else:
                logging.error('SOMETHING VERY STUPID HAPPENED')
                return None

    def save_entries_to_json(self):
        self.entries = {}
        # Iterate over all editable fields and store their values in the entries dictionary
        for field_name, field_widget in self.get_editable_fields().items():
            if isinstance(field_widget, QLineEdit):
                self.entries[field_name] = field_widget.text()
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
    
    def load_entries_from_json_single(self,runParams=['QLineEdit','QComboBox']):
        # Specify the path and filename for the JSON file
        json_file_path = self.globalSettings['JSONGUIstorePath']['value']

        try:
            # Load the entries from the JSON file
            with open(json_file_path, "r") as json_file:
                self.entries = json.load(json_file)

            # Set the values of the editable fields from the loaded entries
            for field_name, field_widget in self.get_editable_fields().items():
                if field_name in self.entries:
                    if isinstance(field_widget, QLineEdit):
                        if 'QLineEdit' in runParams:
                            field_widget.setText(self.entries[field_name])
                            logging.debug('Set text of field_widget '+field_name+' to '+self.entries[field_name])
                    elif isinstance(field_widget, QComboBox):
                        if 'QComboBox' in runParams:
                            index = field_widget.findText(self.entries[field_name])
                            if index >= 0:
                                logging.debug('Set text of field_widget '+field_name+' to '+self.entries[field_name])
                                field_widget.setCurrentIndex(index)
                                #Also change the lineedits and such:
                                if 'Finding' in field_widget.objectName():
                                    self.changeLayout_choice(self.groupboxFinding.layout(),field_widget.objectName(),self.Finding_functionNameToDisplayNameMapping)
                                elif 'Fitting' in field_widget.objectName():
                                    self.changeLayout_choice(self.groupboxFitting.layout(),field_widget.objectName(),self.Fitting_functionNameToDisplayNameMapping)
        

        except FileNotFoundError:
            # Handle the case when the JSON file doesn't exist yet
            self.save_entries_to_json()
            logging.info('No GUI settings storage found, new one created.')
            pass
        
    def get_editable_fields(self):
        fields = {}

        def find_editable_fields(widget):
            if isinstance(widget, QLineEdit) or isinstance(widget, QComboBox):
                fields[widget.objectName()] = widget
            elif isinstance(widget, QWidget):
                # for child_widget in widget.children():
                #     find_editable_fields(child_widget)
                for children_widget in widget.findChildren(QWidget):
                    find_editable_fields(children_widget)

        find_editable_fields(self)
        return fields

    def set_all_combobox_states(self):
        original_states = {}
        # try:
        def set_combobox_states(widget):
            if isinstance(widget, QComboBox):
                original_states[widget] = widget.currentIndex()
                for i in range(widget.count()):
                    logging.debug('Set text of combobox '+widget.objectName()+' to '+widget.itemText(i))
                    widget.setCurrentIndex(i)
                    #Update all line edits and such
                    if 'Finding' in widget.objectName():
                        self.changeLayout_choice(self.groupboxFinding.layout(),widget.objectName(),self.Finding_functionNameToDisplayNameMapping)
                    elif 'Fitting' in widget.objectName():
                        self.changeLayout_choice(self.groupboxFitting.layout(),widget.objectName(),self.Fitting_functionNameToDisplayNameMapping)
            elif isinstance(widget, QWidget):
                for child_widget in widget.children():
                    set_combobox_states(child_widget)

        set_combobox_states(self)
        # Reset to orig states
        for combobox, original_state in original_states.items():
            combobox.setCurrentIndex(original_state)
            if 'Finding' in combobox.objectName():
                self.changeLayout_choice(self.groupboxFinding.layout(),combobox.objectName(),self.Finding_functionNameToDisplayNameMapping)
            elif 'Fitting' in combobox.objectName():
                self.changeLayout_choice(self.groupboxFitting.layout(),combobox.objectName(),self.Fitting_functionNameToDisplayNameMapping)
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
