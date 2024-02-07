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

#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider, QSpacerItem
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile, QThread, pyqtSignal, QObject

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
#Obtain the helperfunctions
from Utils import utils, utilsHelper

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

        #Update the candidate preview to GUI settings
        self.updateCandidatePreview(init=True)
        
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
        globalSettings['MaxFindingBoundingBoxXY'] = {}
        globalSettings['MaxFindingBoundingBoxXY']['value'] = 20
        globalSettings['MaxFindingBoundingBoxXY']['input'] = float
        globalSettings['MaxFindingBoundingBoxXY']['displayName'] = 'Maximum size of a bounding box in px units'
        globalSettings['MaxFindingBoundingBoxT'] = {}
        globalSettings['MaxFindingBoundingBoxT']['value'] = 1000000
        globalSettings['MaxFindingBoundingBoxT']['input'] = float
        globalSettings['MaxFindingBoundingBoxT']['displayName'] = 'Maximum size of a bounding box in us units'
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
        
    def previewRun(self,timeStretch=(0,1000),xyStretch=(0,0,0,0)):
        """
        Generates the preview of a run analysis.

        Parameters:
            timeStretch (tuple): A tuple containing the start and end times for the preview.
            xyStretch (tuple): A tuple containing the minimum and maximum x and y coordinates for the preview.

        Returns:
            None
        """
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
        self.updateShowPreview(previewEvents=self.previewEvents,timeStretch=timeStretch)
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
        self.LocListTable = QTableWidget()
        tab4_layout.addWidget(self.LocListTable, 0, 0)
        
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
        self.canPreviewtab_vertical_container = QVBoxLayout()
        self.tab_canPreview.setLayout(self.canPreviewtab_vertical_container)
        
        #Add a horizontal layout to the first row of the vertical layout - this contains the entry fields and the buttons
        canPreviewtab_horizontal_container = QHBoxLayout()
        self.canPreviewtab_vertical_container.addLayout(canPreviewtab_horizontal_container)
        
        #Add a entry field to type the number of candidate and button to show it
        canPreviewtab_horizontal_container.addWidget(QLabel("Candidate ID: "))
        self.entryCanPreview = QLineEdit()
        onlyInt = QIntValidator()
        onlyInt.setBottom(0)
        self.entryCanPreview.setValidator(onlyInt)
        canPreviewtab_horizontal_container.addWidget(self.entryCanPreview)
        self.entryCanPreview.returnPressed.connect(lambda: self.updateCandidatePreview())
        
        self.buttonCanPreview = QPushButton("Show candidate")
        canPreviewtab_horizontal_container.addWidget(self.buttonCanPreview)

        #Give it a function on click:
        self.buttonCanPreview.clicked.connect(lambda: self.updateCandidatePreview())

        self.prev_buttonCan = QPushButton("Previous")
        self.prev_buttonCan.clicked.connect(lambda: self.prev_candidate())

        self.next_buttonCan = QPushButton("Next")
        self.next_buttonCan.clicked.connect(lambda: self.next_candidate())

        canPreviewtab_horizontal_container.addWidget(self.prev_buttonCan)
        canPreviewtab_horizontal_container.addWidget(self.next_buttonCan)

        # Add an advanced settings button that opens a new window when clicked
        # First initialise (but not show!) advanced window:
        self.advancedOptionsWindowCanPrev = AdvancedOptionsWindowCanPrev(self)
        
        self.advanced_options_button = QPushButton("Advanced Options")
        canPreviewtab_horizontal_container.addWidget(self.advanced_options_button)
        self.advanced_options_button.clicked.connect(self.open_advanced_options_CanPreview)

        #Add a horizontal layout to display info about the cluster
        #self.canPreviewtab_horizontal_container2 = QHBoxLayout()
        #self.canPreviewtab_vertical_container.addLayout(self.canPreviewtab_horizontal_container2)
        self.candidate_info = QLabel('')
        self.canPreviewtab_vertical_container.addWidget(self.candidate_info)
        self.fit_info = QLabel('')
        self.fit_info.setStyleSheet("color: red;")
        self.canPreviewtab_vertical_container.addWidget(self.fit_info)

        #Create an empty figure and store it as self.data:
        self.data['firstCandidateFigure'] = plt.figure(figsize=(6.8,4))
        self.data['firstCandidateCanvas'] = FigureCanvas(self.data['firstCandidateFigure'])
        self.data['firstCandidatePlot'] = ThreeDPointCloud(self.data['firstCandidateFigure'])
        
        #Add a navigation toolbar (zoom, pan etc) and canvas to tab
        self.canPreviewtab_vertical_container.addWidget(NavigationToolbar(self.data['firstCandidateCanvas'], self))
        self.canPreviewtab_vertical_container.addWidget(self.data['firstCandidateCanvas'])

        #Create a second empty figure and store it as self.data:
        self.data['secondCandidateFigure'] = plt.figure(figsize=(6.8,4))
        self.data['secondCandidateCanvas'] = FigureCanvas(self.data['secondCandidateFigure'])
        self.data['secondCandidatePlot'] = TwoDProjection(self.data['secondCandidateFigure'])

        self.clear_index = [0,0]
        
        #Add a navigation toolbar (zoom, pan etc) and canvas to tab
        self.canPreviewtab_vertical_container.addWidget(NavigationToolbar(self.data['secondCandidateCanvas'], self))
        self.canPreviewtab_vertical_container.addWidget(self.data['secondCandidateCanvas'])

    def prev_candidate(self):
        if not self.entryCanPreview.text()=='':
            if 'FindingMethod' in self.data and int(self.entryCanPreview.text())-1 < len(self.data['FindingResult'][0]):
                if int(self.entryCanPreview.text())-1 == -1:
                    max_candidate = len(self.data['FindingResult'][0])-1
                    self.entryCanPreview.setText(str(max_candidate))
                else:
                    self.entryCanPreview.setText(str(int(self.entryCanPreview.text())-1))
                self.updateCandidatePreview()
            else:
                self.candidate_info.setText('Tried to visualise candidate but no data found!')
                logging.error('Tried to visualise candidate but no data found!')
        else:
            self.candidate_info.setText('Tried to visualise candidate, but no ID was given!')
            logging.error('Tried to visualise candidate, but no ID was given!')

    def next_candidate(self):
        if not self.entryCanPreview.text()=='':
            if 'FindingMethod' in self.data and int(self.entryCanPreview.text())+1 <= len(self.data['FindingResult'][0]):
                if int(self.entryCanPreview.text())+1 == len(self.data['FindingResult'][0]):
                    self.entryCanPreview.setText('0')
                else:
                    self.entryCanPreview.setText(str(int(self.entryCanPreview.text())+1))
                self.updateCandidatePreview()
            else:
                self.candidate_info.setText('Tried to visualise candidate but no data found!')
                logging.error('Tried to visualise candidate but no data found!')
        else:
            self.candidate_info.setText('Tried to visualise candidate, but no ID was given!')
            logging.error('Tried to visualise candidate, but no ID was given!')

    def open_advanced_options_CanPreview(self):
        self.advancedOptionsWindowCanPrev.show()

    def updateCandidatePreview(self, reset=False, init=False):
        """
        Function that's called when the button to show the candidate is clicked
        """
        # Clear text and plots
        self.candidate_info.setText('')
        self.fit_info.setText('')
        self.data['firstCandidatePlot'].reset()
        self.data['secondCandidatePlot'].reset()
        self.data['firstCandidateCanvas'].draw()
        self.data['secondCandidateCanvas'].draw()
        plot_prefix = ['first', 'second']
        if init==True:
            for i in range(len(self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses())):
                if self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()[i] != self.data[plot_prefix[i]+'CandidatePlot'].__class__:
                    self.data[plot_prefix[i]+'CandidateFigure'].clf()
                    self.data[plot_prefix[i]+'CandidatePlot'] = self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()[i](self.data[plot_prefix[i]+'CandidateFigure'])    
                    self.clear_index[i] = 0
            logging.info('Candidate preview is initialised.')
        elif reset==False:
            # Check advanced options
            if self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses() == []:
                self.candidate_info.setText('Tried to visualise candidate, but no plot options were selected!')
                self.data['firstCandidateFigure'].clf()
                self.data['firstCandidateCanvas'].draw()
                self.data['secondCandidateFigure'].clf()
                self.data['secondCandidateCanvas'].draw()
                self.clear_index = [1,1]
                logging.error('Tried to visualise candidate, but no plot options were selected!')
            elif len(self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()) <= 2:
                for i in range(len(self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses())):
                    if self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()[i] != self.data[plot_prefix[i]+'CandidatePlot'].__class__:
                        self.data[plot_prefix[i]+'CandidateFigure'].clf()
                        self.data[plot_prefix[i]+'CandidatePlot'] = self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()[i](self.data[plot_prefix[i]+'CandidateFigure'])    
                        self.clear_index[i] = 0
                # Check the candidate entry field
                if self.entryCanPreview.text()=='':
                    self.candidate_info.setText('Tried to visualise candidate, but no ID was given!')
                    logging.error('Tried to visualise candidate, but no ID was given!')
                elif 'FindingMethod' in self.data and int(self.entryCanPreview.text()) < len(self.data['FindingResult'][0]):
                    self.data['CandidatePreviewID'] = int(self.entryCanPreview.text())
                    logging.debug(f"Attempting to show candidate {self.data['CandidatePreviewID']}.")

                    # Get all localizations that belong to the candidate
                    self.data['CandidatePreviewLocs'] = self.data['FittingResult'][0][self.data['FittingResult'][0]['candidate_id'] == self.data['CandidatePreviewID']]
                    print(self.data['CandidatePreviewLocs']['x'].iloc[0])
                    if pd.isna(self.data['CandidatePreviewLocs']['x'].iloc[0]):
                        self.fit_info.setText(f"No localization generated due to {self.data['CandidatePreviewLocs']['fit_info'].iloc[0]}")
                        print(self.data['CandidatePreviewLocs']['fit_info'].iloc[0])
                    pixel_size = self.globalSettings['PixelSize_nm']['value']

                    # Get some info about the candidate
                    N_events = self.data['FindingResult'][0][self.data['CandidatePreviewID']]['N_events']
                    cluster_size = self.data['FindingResult'][0][self.data['CandidatePreviewID']]['cluster_size']
                    self.candidate_info.setText(f"This candidate cluster contains {N_events} events and has dimensions ({cluster_size[0]}, {cluster_size[1]}, {cluster_size[2]}).")
                    
                    # Check surrounding option
                    surrounding = pd.DataFrame()
                    surroundingOptions = self.advancedOptionsWindowCanPrev.getSurroundingAndPaddingValues()
                    if len(self.previewEvents) != 0 and surroundingOptions[0]==True:
                        surrounding = self.get_surrounding(self.previewEvents, self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events'], surroundingOptions[1], surroundingOptions[2], surroundingOptions[3])
                    
                    # Check if plot was cleared before
                    if self.clear_index[0] == 1:
                        self.data[plot_prefix[0]+'CandidatePlot'] = self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()[0](self.data[plot_prefix[0]+'CandidateFigure'])
                        self.clear_index[0] = 0
                    
                    # Plot the first candidate plot
                    self.data['firstCandidatePlot'].plot(self.data['firstCandidateFigure'], self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events'], surrounding, self.data['CandidatePreviewLocs'], pixel_size)
                    
                    # Update the first canvas
                    self.data['firstCandidateFigure'].tight_layout()
                    self.data['firstCandidateCanvas'].draw()
                    logging.info(f"3D scatter plot of candidate {self.data['CandidatePreviewID']} drawn.")
                    
                    # Clear second plot if needed
                    if len(self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()) == 1:
                        # Clear second plot
                        self.data['secondCandidateFigure'].clf()
                        self.data['secondCandidateCanvas'].draw()
                        self.clear_index[1]=1
                        logging.info('Clearing second plot.')
                    
                    # Plot the second candidate plot
                    if len(self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()) == 2:
                        if self.clear_index[1] == 1:
                            self.data[plot_prefix[1]+'CandidatePlot'] = self.advancedOptionsWindowCanPrev.getCheckedPlotOptionClasses()[1](self.data[plot_prefix[1]+'CandidateFigure'])
                            self.clear_index[1] = 0
                        self.data['secondCandidatePlot'].plot(self.data['firstCandidateFigure'], self.data['FindingResult'][0][self.data['CandidatePreviewID']]['events'], surrounding, self.data['CandidatePreviewLocs'], pixel_size)
                        # Update the second canvas
                        self.data['secondCandidateFigure'].tight_layout()
                        self.data['secondCandidateCanvas'].draw()
                        logging.info(f"2D event-projections of candidate {self.data['CandidatePreviewID']} drawn.")

                else: 
                    self.candidate_info.setText('Tried to visualise candidate but no data found!')
                    logging.error('Tried to visualise candidate but no data found!')
            else:
                self.candidate_info.setText('Tried to visualise candidate, but no plot options were selected!')
                logging.error('Tried to visualise candidate, but too many plot options were selected!')
        else:
            logging.info('Candidate preview is reset.')

    def get_surrounding(self, events, candidate_events, x_padding, y_padding, t_padding):
        xlim = [np.min(candidate_events['x'])-x_padding, np.max(candidate_events['x'])+x_padding]
        ylim = [np.min(candidate_events['y'])-y_padding, np.max(candidate_events['y'])+y_padding]
        tlim = [np.min(candidate_events['t'])-t_padding*1e3, np.max(candidate_events['t'])+t_padding*1e3]
        mask = ((events['x'] >= xlim[0]) & (events['x'] <= xlim[1]) & (events['y'] >= ylim[0]) & (events['y'] <= ylim[1]) & (events['t'] >= tlim[0]) & (events['t'] <= tlim[1]))
        all_events = pd.DataFrame(events[mask])
        surrounding = all_events.merge(candidate_events, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        return surrounding

    def setup_previewTab(self):
        """
        Function that creates the preview tab
        Very practically, it creates 10 figures, stores those in memory, and displays them when necessary
        """
        #It's a grid layout
        self.previewtab_layout = QGridLayout()
        self.tab_previewVis.setLayout(self.previewtab_layout)
        
        #And I add the previewfindingfitting class (QWidget extension):
        self.previewtab_widget = PreviewFindingFitting()
        self.previewtab_layout.addWidget(self.previewtab_widget)

    def updateShowPreview(self,previewEvents=None,timeStretch=None):
        """
        Function that's called to update the preview (or show it). Requires the previewEvents, or uses the self.previewEvents.
        """
        
        self.previewtab_widget.displayEvents(previewEvents,frametime_ms=100,findingResult = self.data['FindingResult'][0],fittingResult=self.data['FittingResult'][0],settings=self.globalSettings,timeStretch=timeStretch)
        
        
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
        # else:
        #     self.open_critical_warning(error)
    
    def updateLocList(self):
        #data is stored in self.data['FittingResult'][0]
        #fill the self.LocListTable QTableWidget with the data:
        localizations = self.data['FittingResult'][0].dropna(axis=0, ignore_index=True)
        localizations = localizations.drop('fit_info', axis=1)
        
        #Get the shape of the data
        nrRows = np.shape(localizations)[0]
        nrColumns = np.shape(localizations)[1]
        
        #Give the loclisttable the correct row/column count:
        self.LocListTable.setRowCount(nrRows)
        self.LocListTable.setColumnCount(nrColumns)
        
        #Fill the loclisttable with the output:
        for r in range(nrRows):
            for c in range(nrColumns):
                nrDigits = 2
                item = QTableWidgetItem(f"{round(localizations.iloc[r, c], nrDigits):.{nrDigits}f}")
                self.LocListTable.setItem(r, c, item)
        
        #Add headers
        self.LocListTable.setHorizontalHeaderLabels(localizations.columns.tolist())
        
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
            #Check if we are loading existing finding
            if 'ExistingFitting' in getattr(self,f"CandidateFittingDropdown{polarityVal}").currentText() or 'existing Fitting' in getattr(self,f"CandidateFittingDropdown{polarityVal}").currentText():
                logging.info('Skipping finding and fitting processing')
                # compare filenames of finding and fitting here? Following code is not yet adapted
                # finding_file = getattr(self,f"CandidateFindingDropdown{polarityVal}").currentText().replace("_FindingResults_", "_")
                # fitting_file = getattr(self,f"CandidateFittingDropdown{polarityVal}").currentText().replace("_FittingResults_", "_")

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
                    
                    
                    previous_read_hdfChunk = 1
                    events_prev = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})
                    self.chunckloading_number_chuck = 0
                    
                    while self.chunckloading_finished_chunking == False:
                
                
                        self.chunckloading_currentLimits = [[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000],[(self.chunckloading_number_chuck)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000-float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000,(self.chunckloading_number_chuck+1)*float(self.globalSettings['FindingBatchingTimeMs']['value'])*1000+float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])*1000]]        
                
                        current_read_index = 0
                        hdf5_read_chunk_size = 500000 #nr of entries that are loaded after which it's checked whether T makes sense
                        
                        # Retrieve all entries within the specified bounding box
                        t_min = self.chunckloading_currentLimits[0][0]+float(self.run_startTLineEdit.text())*1000
                        t_max = min(self.chunckloading_currentLimits[1][1],float(self.run_durationTLineEdit.text())*1000)+float(self.run_startTLineEdit.text())*1000
                        
                        #Load the hdf5 file
                        with h5py.File(fileToRun, mode='r') as file:
                            events_hdf5 = file['CD']['events']
                            
                            #We start at some chunk in time
                            n_hdfChunk = previous_read_hdfChunk
                            #We check if this chunk is fully loaded in hdf5
                            fullChunkLoaded = False
                            allhdf5chunkslices = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})
                            
                            #First figure out the start/end of this chunk:
                            startPos = -1
                            endPos = -1
                            while (startPos == -1) or (endPos == -1):
                                #Read a single entry:
                                if current_read_index*hdf5_read_chunk_size < events_hdf5.size:
                                    single_entry = events_hdf5[current_read_index*hdf5_read_chunk_size]
                                    if startPos == -1:
                                        if single_entry[3] > t_min:
                                            startPos = current_read_index*hdf5_read_chunk_size
                                    if endPos == -1:
                                        if single_entry[3] > t_max:
                                            endPos = current_read_index*hdf5_read_chunk_size
                                    current_read_index+=1
                                else:
                                    if startPos == -1:
                                        startPos = events_hdf5.size
                                    endPos = events_hdf5.size
                                    logging.info('end of file reached while streaming')
                            
                            allhdf5chunkslices = events_hdf5[max(0,startPos-hdf5_read_chunk_size):endPos]
                            
                            previous_read_hdfChunk = current_read_index
                            
                                        
                        #Better structure
                        events = allhdf5chunkslices
                        # logging.warning(min(events['t']))
                        # logging.warning(max(events['t']))
                        #Store this for the next chunk
                        previous_read_hdfChunk = n_hdfChunk-1
                        #at this point fullChunkData is a (slightly too big in time) time-slice of the hdf5 file, and previous_read_hdfChunk is set so that next run we continue where we left off.
                        
                        events = events[(events['t'] >= t_min) & (events['t'] <= t_max)]
                        
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
                        
                else: #If we want to pre-load the existing finding info:
                    findingOffset = 0
                    if polarityVal == 'Neg':
                        findingOffset = self.number_finding_found_polarity['Pos']
                    
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
                logging.info('Number of candidates found: '+str(len(self.data['FindingResult'][0])))
                logging.info('Candidate finding took '+str(self.currentFileInfo['FindingTime'])+' seconds.')
                logging.info('Candidate finding done!')
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
        else:
            self.open_critical_warning(f"No Finding evaluation text provided/found")

    def updateGUIafterNewFitting(self):
        logging.info('Number of localizations found: '+str(len(self.data['FittingResult'][0].dropna(axis=0))))
        logging.info('Candidate fitting took '+str(self.currentFileInfo['FittingTime'])+' seconds.')
        logging.info('Candidate fitting done!')
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
            self.updateCandidatePreview()
        else: 
            # found candidates have changed -> don't update preview, but update previous run data
            self.data['prevFindingMethod'] = self.data['FindingMethod']
            self.data['prevNrCandidates'] = len(self.data['FindingResult'][0])
            self.data['prevNrEvents'] = self.data['NrEvents']
            self.updateCandidatePreview(reset=True)
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
        else:
            self.open_critical_warning(f"No Fitting evaluation text provided/found")

    def getStoreLocationPartial(self):   
        if 'ExistingFinding' in self.data['FindingMethod']:
            FindingResultFileName = self.data['FindingMethod'][self.data['FindingMethod'].index('File_Location="')+len('File_Location="'):self.data['FindingMethod'].index('")')]
            storeLocationPartial = FindingResultFileName[:-7]
        else:
            storeLocationPartial = self.currentFileInfo['CurrentFileLoc'][:-4]
        return storeLocationPartial     
    
    def storeLocalizationOutput(self):
        logging.debug('Attempting to store fitting results output')
        storeLocation = self.getStoreLocationPartial()+'_FitResults_'+self.storeNameDateTime+'.csv'
        #Store the localization output
        localizations = self.data['FittingResult'][0].dropna(axis=0, ignore_index=True)
        localizations = localizations.drop('fit_info', axis=1)
        if self.globalSettings['OutputDataFormat']['value'] == 'minimal':
            localizations.to_csv(storeLocation)
        elif self.globalSettings['OutputDataFormat']['value'] == 'thunderstorm':
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
                            pickle.dump(allPosFittingResults, file)
                            
                            
                        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FittingResults_NegOnly_'+self.storeNameDateTime+'.pickle'
                        with open(file_path, 'wb') as file:
                            pickle.dump(allNegFittingResults, file)
                        
                        #And store all of them
                        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FittingResults_'+self.storeNameDateTime+'.pickle'
                        with open(file_path, 'wb') as file:
                            pickle.dump(self.data['FittingResult'][0], file)
                except:
                    logging.debug('This can be safely ignored')
            else:#Only a single pos/neg selected
                file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FittingResults_'+self.storeNameDateTime+'.pickle'
                with open(file_path, 'wb') as file:
                    pickle.dump(self.data['FittingResult'][0], file)
            logging.info('Fitting results output stored')
        else:
            pass
    
    def storeFindingOutput(self,polarityVal='Pos'):
        logging.debug('Attempting to store finding results output')
        #Store the Finding results output
        file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_'+self.storeNameDateTime+'.pickle'
        with open(file_path, 'wb') as file:
            pickle.dump(self.data['FindingResult'][0], file)
            
            
        #Also save pos and neg seperately if so useful:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            try:
                if self.number_finding_found_polarity['Pos'] > 0 and self.number_finding_found_polarity['Neg'] > 0:
                    allPosFindingResults = {key: value for key, value in self.data['FindingResult'][0].items() if key < self.number_finding_found_polarity['Pos']}
                    allNegFindingResults = {key-self.number_finding_found_polarity['Pos']: value for key, value in self.data['FindingResult'][0].items() if key >= self.number_finding_found_polarity['Pos']}
                    
                    file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_PosOnly_'+self.storeNameDateTime+'.pickle'
                    with open(file_path, 'wb') as file:
                        pickle.dump(allPosFindingResults, file)
                        
                        
                    file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_NegOnly_'+self.storeNameDateTime+'.pickle'
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

    def set_all_combobox_states(self):
        original_states = {}
        # try:
        def set_combobox_states(widget,polVal):
            if isinstance(widget, QComboBox):
                if widget.objectName()[:-3] == polVal.lower():
                    original_states[widget] = widget.currentIndex()
                    for i in range(widget.count()):
                        logging.info('Set text of combobox '+widget.objectName()+' to '+widget.itemText(i))
                        widget.setCurrentIndex(i)
                        #Update all line edits and such
                        if 'Finding' in widget.objectName():
                            logging.info('Line 2299')
                            logging.info(getattr(self, f"groupboxFinding{polVal}").objectName())
                            utils.changeLayout_choice(getattr(self, f"groupboxFinding{polVal}").layout(),widget.objectName(),getattr(self, f"Finding_functionNameToDisplayNameMapping{polVal}"),parent=self)
                        elif 'Fitting' in widget.objectName():
                            logging.info('Line 2299+2')
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
                    utils.changeLayout_choice(getattr(self, f"groupboxFinding{polVal}").layout(),combobox.objectName(),getattr(self, f"Finding_functionNameToDisplayNameMapping{polVal}"),parent=self)
                elif 'Fitting' in combobox.objectName():
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

class AdvancedOptionsWindowCanPrev(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Advanced options")
        self.resize(300, 200)

        self.parent = parent
        # Set the window icon to the parent's icon
        self.setWindowIcon(self.parent.windowIcon())

        #Create a vertical layout
        layout = QVBoxLayout()

        # Add a QGroupBox for "Plots to show"
        self.plotOptionsGroupBox = QGroupBox("Plots to show")
        plotOptionsLayout = QGridLayout()
        
        # Create and add QCheckboxes for plot options to the grid layout
        self.plotOptionCheckboxes = []
        self.currentSelection = []
        self.plotOptions = {1: {"name": "3D pointcloud", "description": "3D pointcloud of candidate cluster.", "plotclass": ThreeDPointCloud},
                            2: {"name": "3D pointcloud + first events", "description": "3D pointcloud of candidate cluster, first events per pixel are labeled.", "plotclass": ThreeDPointCloudwFirst},
                            3: {"name": "2D projections", "description": "2D projections of candidate cluster.", "plotclass": TwoDProjection},
                            4: {"name": "2D timestamps", "description": "2D timestamps (first, median, mean event) per pixel in candidate cluster.", "plotclass": TwoDTimestamps},
                            5: {"name": "2D event delays", "description": "2D event delays (min, max, mean) per pixel in candidate cluster.", "plotclass": TwoDDelays}}
        currentRow=0
        rowMax = 3
        curruntCol=0
        for option_id, option_data in self.plotOptions.items():
            checkbox = QCheckBox(option_data["name"])
            checkbox.setToolTip(option_data["description"])  # Set tooltip with description
            checkbox.setObjectName(f"PlotOption_{option_id}")  # Set object name for identification
            checkbox.stateChanged.connect(self.handlePlotOptionCheckboxChange)
            plotOptionsLayout.addWidget(checkbox, currentRow, curruntCol)
            currentRow += 1
            if currentRow == rowMax:
                currentRow = 0
                curruntCol += 1
            self.plotOptionCheckboxes.append(checkbox)

        self.plotOptionsGroupBox.setLayout(plotOptionsLayout)
        layout.addWidget(self.plotOptionsGroupBox)
        # check first checkbox
        self.plotOptionCheckboxes[3].setChecked(True)

        #Add a grid layout
        grid = QGridLayout()
        layout.addLayout(grid)
        currentRow = 0

        # show surrounding tick box
        self.showSurroundingCheckBox = QCheckBox("Show surrounding")
        self.showSurroundingCheckBox.setObjectName("CanPreview_showSurrounding")
        self.showSurroundingCheckBox.stateChanged.connect(lambda state: self.showSurrounding_checked(state))
        grid.addWidget(self.showSurroundingCheckBox, currentRow, 0)
        # show x,y,t padding line edits if show surrounding is checked make them gray if not checked and unwritable
        self.xPaddingQLabel = QLabel("x-padding:")
        self.yPaddingQLabel = QLabel("y-padding:")
        self.tPaddingQLabel = QLabel("t-padding:")
        self.xPaddingLineEdit = QLineEdit("10")
        self.yPaddingLineEdit = QLineEdit("10")
        self.tPaddingLineEdit = QLineEdit("10")
        self.xPaddingLineEdit.setObjectName("CanPreview_xPadding")
        self.yPaddingLineEdit.setObjectName("CanPreview_yPadding")
        self.tPaddingLineEdit.setObjectName("CanPreview_tPadding")
        grid.addWidget(self.xPaddingQLabel, currentRow, 1)
        grid.addWidget(self.xPaddingLineEdit, currentRow, 2)
        grid.addWidget(self.yPaddingQLabel, currentRow, 3)
        grid.addWidget(self.yPaddingLineEdit, currentRow, 4)
        grid.addWidget(self.tPaddingQLabel, currentRow, 5)
        grid.addWidget(self.tPaddingLineEdit, currentRow, 6)
        self.showSurrounding_checked(self.showSurroundingCheckBox.checkState())

        # Create a widget and set and add layouts
        widget = QWidget()
        widget.setLayout(layout)

        # Set the central widget of the main window
        self.setCentralWidget(widget)

    def showSurrounding_checked(self, state):
        if state == 2:  # 2 is Qt.Checked
            self.xPaddingQLabel.setStyleSheet("")
            self.yPaddingQLabel.setStyleSheet("")
            self.tPaddingQLabel.setStyleSheet("")
            self.xPaddingLineEdit.setStyleSheet("")
            self.yPaddingLineEdit.setStyleSheet("")
            self.tPaddingLineEdit.setStyleSheet("")
            self.xPaddingLineEdit.setReadOnly(False)
            self.yPaddingLineEdit.setReadOnly(False)
            self.tPaddingLineEdit.setReadOnly(False)
        else:
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

class ThreeDPointCloud:
    def __init__(self, figure):
        figure.suptitle("3D pointcloud of candidate cluster")
        self.ax = figure.add_subplot(111, projection='3d')
        figure.tight_layout()
        figure.subplots_adjust(top=0.95)

    def reset(self):
        self.ax.cla()

    def plot(self, figure, events, surrounding, localizations, pixel_size):
        pos_events = events[events['p'] == 1]
        neg_events = events[events['p'] == 0]
        # Do a 3d scatterplot of the event data
        if not len(pos_events)==0:
            self.ax.scatter(pos_events['x'], pos_events['y'], pos_events['t']*1e-3, label='Positive events', color='C0')
        if not len(neg_events)==0:
            self.ax.scatter(neg_events['x'], neg_events['y'], neg_events['t']*1e-3, label='Negative events', color='C1')
        if not len(surrounding)==0:
            self.ax.scatter(surrounding['x'], surrounding['y'], surrounding['t']*1e-3, label='Surrounding events', color='black')
        self.ax.set_xlabel('x [px]')
        self.ax.set_ylabel('y [px]')
        self.ax.set_zlabel('t [ms]')
        self.ax.invert_zaxis()

        # Plot the localization(s) of the candidate
        self.ax.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, localizations['t'], marker='x', c='red', label='Localization(s)')
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))

class ThreeDPointCloudwFirst:
    def __init__(self, figure):
        figure.suptitle("3D pointcloud of candidate cluster")
        self.ax = figure.add_subplot(111, projection='3d')
        figure.tight_layout()
        figure.subplots_adjust(top=0.95)

    def reset(self):
        self.ax.cla()

    def plot(self, figure, events, surrounding, localizations, pixel_size):
        first_events = utilsHelper.FirstTimestamp(events).get_smallest_t(events)
        eventsFiltered = events.merge(first_events, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        pos_events = eventsFiltered[eventsFiltered['p'] == 1]
        neg_events = eventsFiltered[eventsFiltered['p'] == 0]
        # Do a 3d scatterplot of the event data
        if not len(first_events)==0:
            self.ax.scatter(first_events['x'], first_events['y'], first_events['t']*1e-3, label='First events', color='C2')
        if not len(pos_events)==0:
            self.ax.scatter(pos_events['x'], pos_events['y'], pos_events['t']*1e-3, label='Positive events', color='C0')
        if not len(neg_events)==0:
            self.ax.scatter(neg_events['x'], neg_events['y'], neg_events['t']*1e-3, label='Negative events', color='C1')
        if not len(surrounding)==0:
            self.ax.scatter(surrounding['x'], surrounding['y'], surrounding['t']*1e-3, label='Surrounding events', color='black')
        self.ax.set_xlabel('x [px]')
        self.ax.set_ylabel('y [px]')
        self.ax.set_zlabel('t [ms]')
        self.ax.invert_zaxis()

        # Plot the localization(s) of the candidate
        self.ax.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, localizations['t'], marker='x', c='red', label='Localization(s)')
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))

class TwoDProjection:
    def __init__(self, figure):
        figure.suptitle("2D projections of candidate cluster")
        self.ax_xy = figure.add_subplot(121)
        self.ax_xt = figure.add_subplot(222)
        self.ax_yt = figure.add_subplot(224)
        figure.tight_layout()

    def reset(self):
        self.ax_xy.cla()
        self.ax_xt.cla()
        self.ax_yt.cla()
        self.ax_xt.set_aspect('auto')
        self.ax_yt.set_aspect('auto')

    def plot(self, figure, events, surrounding, localizations, pixel_size):
        hist_xy = utilsHelper.Hist2d_xy(events)
        hist_tx = utilsHelper.Hist2d_tx(events)
        hist_ty = utilsHelper.Hist2d_ty(events)

        x_edges, y_edges, t_edges = hist_xy.x_edges, hist_xy.y_edges, hist_tx.x_edges

        # Compute the 2D histograms (pos)
        hist_xy_pos = hist_xy(events[events['p'] == 1])[0]
        hist_tx_pos = hist_tx(events[events['p'] == 1])[0]
        hist_ty_pos = hist_ty(events[events['p'] == 1])[0]
        # Compute the 2D histograms (neg)
        hist_xy_neg = hist_xy(events[events['p'] == 0])[0]
        hist_tx_neg = hist_tx(events[events['p'] == 0])[0]
        hist_ty_neg = hist_ty(events[events['p'] == 0])[0]

        # Set goodlooking aspect ratio depending on nr of xyt-bins
        aspectty = 3. * (len(t_edges)-1) / (len(y_edges)-1)
        aspecttx = 3. * (len(t_edges)-1) / (len(x_edges)-1)

        # Plot the 2D histograms
        self.ax_xy.pcolormesh(x_edges, y_edges, hist_xy.dist2D)
        self.ax_xy.set_aspect('equal')
        self.ax_xy.format_coord = lambda x,y:self.format_coord_projectionxy(x,y,hist_xy_pos.T, hist_xy_neg.T, x_edges, y_edges)
        self.ax_xt.pcolormesh(t_edges, x_edges, hist_tx.dist2D)
        self.ax_xt.set_aspect(aspecttx)
        self.ax_xt.format_coord = lambda x,y:self.format_coord_projectiontx(x,y,hist_tx_pos.T, hist_tx_neg.T, t_edges, x_edges)
        self.ax_yt.pcolormesh(t_edges, y_edges, hist_ty.dist2D)
        self.ax_yt.set_aspect(aspectty)
        self.ax_yt.format_coord = lambda x,y:self.format_coord_projectionty(x,y,hist_ty_pos.T, hist_ty_neg.T, t_edges, y_edges)
        self.ax_xy.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, marker='x', c='red')
        self.ax_xt.plot(localizations['t'], localizations['x']/pixel_size, marker='x', c='red')
        self.ax_yt.plot(localizations['t'], localizations['y']/pixel_size, marker='x', c='red')

        # Add and set labels
        self.ax_xy.set_xlabel('x [px]')
        self.ax_xy.set_ylabel('y [px]')
        self.ax_xt.set_ylabel('x [px]')
        self.ax_yt.set_ylabel('y [px]')
        self.ax_yt.set_xlabel('t [ms]')

    def format_coord_projectionxy(self, x, y, pos_hist, neg_hist, x_edges, y_edges):
        """
        Function that formats the coordinates of the mouse cursor in candidate preview xy projection
        """
        x_pix = round(x)
        y_pix = round(y)
        x_bin = np.digitize(x, x_edges) - 1
        y_bin = np.digitize(y, y_edges) - 1
        pos = int(pos_hist[x_bin, y_bin])
        neg = int(neg_hist[x_bin, y_bin])

        display = f'x={x_pix}, y={y_pix}, events[pos,neg]=[{pos}, {neg}]'
        return display

    def format_coord_projectiontx(self, x, y, pos_hist, neg_hist, t_edges, x_edges):
        """
        Function that formats the coordinates of the mouse cursor in candidate preview xt projection
        """
        time = round(x)
        x_pix = round(y)
        x_bin = np.digitize(x, t_edges) - 1
        y_bin = np.digitize(y, x_edges) - 1
        pos = int(pos_hist[x_bin, y_bin])
        neg = int(neg_hist[x_bin, y_bin])

        display = f't={time} ms, x={x_pix}, events[pos,neg]=[{pos}, {neg}]'
        return display

    def format_coord_projectionty(self, x, y, pos_hist, neg_hist, t_edges, y_edges):
        """
        Function that formats the coordinates of the mouse cursor in candidate preview yt projection
        """
        time = round(x)
        y_pix = round(y)
        x_bin = np.digitize(x, t_edges) - 1
        y_bin = np.digitize(y, y_edges) - 1
        pos = int(pos_hist[x_bin, y_bin])
        neg = int(neg_hist[x_bin, y_bin])

        display = f't={time:.2f} ms, y={y_pix}, events[pos,neg]=[{pos}, {neg}]'
        return display

class TwoDTimestamps:
    def __init__(self, figure):
        figure.suptitle("2D timestamps of first, median, mean event per pixel")
        self.ax_first = figure.add_subplot(131)
        self.ax_median = figure.add_subplot(132)
        self.ax_mean = figure.add_subplot(133)
        figure.tight_layout()
        self.cb = None

    def reset(self):
        self.ax_first.cla()
        self.ax_median.cla()
        self.ax_mean.cla()
        self.ax_first.set_aspect('equal')
        self.ax_median.set_aspect('equal')
        self.ax_mean.set_aspect('equal')

    def plot(self, figure, events, surrounding, localizations, pixel_size):

        first = utilsHelper.FirstTimestamp(events).dist2D
        median = utilsHelper.MedianTimestamp(events).dist2D
        mean = utilsHelper.AverageTimestamp(events).dist2D

        x_edges, y_edges = utilsHelper.Hist2d_xy(events).x_edges, utilsHelper.Hist2d_xy(events).y_edges

        # Plot the 2D histograms
        first_mesh = self.ax_first.pcolormesh(x_edges, y_edges, first*1e-3)
        self.ax_first.set_aspect('equal')
        self.ax_first.format_coord = lambda x,y:self.format_coord_timestamp(x,y,first, x_edges, y_edges)
        self.ax_median.pcolormesh(x_edges, y_edges, median*1e-3)
        self.ax_median.set_aspect('equal')
        self.ax_mean.pcolormesh(x_edges, y_edges, mean*1e-3)
        self.ax_mean.set_aspect('equal')
        self.ax_first.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, marker='x', c='red')
        self.ax_mean.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, marker='x', c='red')
        self.ax_median.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, marker='x', c='red')

        # Add and set labels
        self.ax_first.set_xlabel('x [px]')
        self.ax_first.set_ylabel('y [px]')
        self.ax_median.set_xlabel('x [px]')
        self.ax_median.set_ylabel('y [px]')
        self.ax_mean.set_xlabel('x [px]')
        self.ax_mean.set_ylabel('y [px]')

        # Add or update colorbar
        if self.cb is None:
            divider = make_axes_locatable(self.ax_first)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cb = figure.colorbar(first_mesh, cax=cax)
            self.cb.set_label('time [ms]')
        else: 
            self.cb.update_normal(first_mesh)

    def format_coord_timestamp(self, x, y, timedist, x_edges, y_edges):
        """
        Function that formats the coordinates of the mouse cursor in candidate preview xy timeplot
        """
        x_pix = round(x)
        y_pix = round(y)
        x_bin = np.digitize(x, x_edges) - 1
        y_bin = np.digitize(y, y_edges) - 1
        time = timedist[y_bin, x_bin]*1e-3

        if time == np.nan:
            display = f'x={x_pix}, y={y_pix}'
        else:
            display = f'x={x_pix}, y={y_pix}, time={time:.2f} ms'
        return display

class TwoDDelays:
    def __init__(self, figure):
        figure.suptitle("2D event delays min, max, mean per pixel in candidate cluster")
        self.ax_min = figure.add_subplot(131)
        self.ax_max = figure.add_subplot(132)
        self.ax_mean = figure.add_subplot(133)
        figure.tight_layout()

    def reset(self):
        self.ax_min.cla()
        self.ax_max.cla()
        self.ax_mean.cla()
        self.ax_min.set_aspect('equal')
        self.ax_max.set_aspect('equal')
        self.ax_mean.set_aspect('equal')

    def plot(self, figure, events, surrounding, localizations, pixel_size):
        min = utilsHelper.MinTimeDiff(events).dist2D
        max = utilsHelper.MaxTimeDiff(events).dist2D
        mean = utilsHelper.AverageTimeDiff(events).dist2D

        x_edges, y_edges = utilsHelper.Hist2d_xy(events).x_edges, utilsHelper.Hist2d_xy(events).y_edges

        # Plot the 2D histograms
        self.ax_min.pcolormesh(x_edges, y_edges, min)
        self.ax_min.set_aspect('equal')
        self.ax_max.pcolormesh(x_edges, y_edges, max)
        self.ax_max.set_aspect('equal')
        self.ax_mean.pcolormesh(x_edges, y_edges, mean)
        self.ax_mean.set_aspect('equal')
        self.ax_min.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, marker='x', c='red')
        self.ax_mean.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, marker='x', c='red')
        self.ax_max.plot(localizations['x']/pixel_size, localizations['y']/pixel_size, marker='x', c='red')

        # Add and set labels
        self.ax_min.set_xlabel('x [px]')
        self.ax_min.set_ylabel('y [px]')
        self.ax_max.set_xlabel('x [px]')
        self.ax_max.set_ylabel('y [px]')
        self.ax_mean.set_xlabel('x [px]')
        self.ax_mean.set_ylabel('y [px]')

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
        # Create a napari viewer
        self.napariviewer = Viewer(show=False)
        # Create a layout for the main widget
        self.mainlayout = QVBoxLayout()
        # Set the layout for the main widget
        self.setLayout(self.mainlayout)
        
        #Create a groupbox for visualisation methods
        self.VisualisationGroupbox = QGroupBox("Visualisation")
        self.VisualisationGroupbox.setLayout(QGridLayout())
        
        #Add a button:
        button = QPushButton("Visualise", self)
        self.VisualisationGroupbox.layout().addWidget(button)
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        self.VisualisationGroupbox.setObjectName("VisualiseGroupboxKEEP")
        
        visualisationDropdown = QComboBox(self)
        visualisationDropdown.setMaxVisibleItems(30)
        
        #Add the visualisationDropdown to the layout
        self.VisualisationGroupbox.layout().addWidget(visualisationDropdown)
        
        #Ensure that we have 'KEEP' in the objectname so it's not deleted later
        visualisationDropdown_name = "VisualisationDropdownKEEP"
        Visualisation_functionNameToDisplayNameMapping_name = f"Visualisation_functionNameToDisplayNameMapping"
        # self.Visualisation_functionNameToDisplayNameMapping = Visualisation_functionNameToDisplayNameMapping_name
        
        options = utils.functionNamesFromDir('Visualisation')
        displaynames, Visualisation_functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options,'')
        visualisationDropdown.setObjectName(visualisationDropdown_name)
        visualisationDropdown.addItems(displaynames)
            
        setattr(self, Visualisation_functionNameToDisplayNameMapping_name, Visualisation_functionNameToDisplayNameMapping)
        groupbox_name="GroupboxVisualisation"
        layout_name = f"layoutVisualisation"
        
        # #On startup/initiatlisation: also do changeLayout_choice
        utils.changeLayout_choice(self.VisualisationGroupbox.layout(),visualisationDropdown_name,getattr(self, Visualisation_functionNameToDisplayNameMapping_name),parent=self,ignorePolarity=True)
        
        #add a 'Visualise!' button to this groupbox:
        button = QPushButton("Visualise!", self)
        button.setObjectName("VisualiseRunButtonKEEP")
        self.VisualisationGroupbox.layout().addWidget(button)
        #And add a callback to this:
        button.clicked.connect(lambda text, parent=parent: self.visualise_callback(parent))
        
        #Add the groupbox to the mainlayout
        self.mainlayout.layout().addWidget(self.VisualisationGroupbox)
        
        
        self.viewer = QtViewer(self.napariviewer)
        
        self.mainlayout.addWidget(self.viewer)
        self.mainlayout.addWidget(self.viewer.controls)
        logging.info('VisualisationNapari init')

    def visualise_callback(self,parent):
        logging.info('Visualise button pressed')
        
        #Get the current function callback
        FunctionEvalText = self.getVisFunctionEvalText("parent.data['FittingResult'][0]","parent.globalSettings")
        print(FunctionEvalText)
        resultImage = eval(FunctionEvalText)
        
        #Clear all existing layers
        for layer in reversed(self.napariviewer.layers):
            self.napariviewer.layers.remove(layer)
            
        #Add a new layer which is this image
        self.napariviewer.add_image(resultImage[0], multiscale=False)
        


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
    
class PreviewFindingFitting(QWidget):
    """
    Class that runs the GUI of finding/fitting preview (i.e. showing alle vents as an image and overlays with boxes/dots)
    """
    def __init__(self):
        """
        Initialisation of the PreviewFindingFitting class. Sets up a napari viewer and adds the viewer and viewer control widgets to the main layout.
        Also initialises some empty arrays
        """
        super().__init__()
        # Create a napari viewer
        self.napariviewer = Viewer(show=False)
        # Create a layout for the main widget
        self.mainlayout = QVBoxLayout()
        # Set the layout for the main widget
        self.setLayout(self.mainlayout)
        
        
        self.viewer = QtViewer(self.napariviewer)
        
        
        self.viewer.on_mouse_move = lambda event: self.currently_under_cursor(event)
        #Test
        # self.viewer.window.add_plugin_dock_widget('napari-1d', 'napari-1d')
        
        self.mainlayout.addWidget(self.viewer)
        self.mainlayout.addWidget(self.viewer.controls)
        # self.mainlayout.addWidget(self.viewer.layers)
        
        # self.mainlayout.addWidget(self.viewer.dockConsole)
        # self.mainlayout.addWidget(self.viewer.layerButtons)
        # self.mainlayout.addWidget(self.viewer.viewerButtons)
        logging.info('PreviewFindingFitting init')
        self.finding_overlays = {}
        self.fitting_overlays = {}
        self.maxFrames = 0
        self.napariviewer.dims.events.current_step.connect(self.update_visibility)

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
        #TODO: usefull info from mouse-over events
        # print(np.floor(highlighted_px_index))
    
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
        #Delete all existing layers:
        for layer in reversed(self.napariviewer.layers):
            self.napariviewer.layers.remove(layer)
        
        #Create a new image
        preview_multiD_image = []
        #Loop over the frames:
        n_frames = int(np.ceil(float(timeStretch[1])/(frametime_ms)))
        self.maxFrames = n_frames
        for n in range(0,n_frames):
            #Get the events on this 'frame'
            events_this_frame = events[(events['t']>(float(timeStretch[0])*1000+n*frametime_ms*1000)) & (events['t']<(float(timeStretch[0])*1000+(n+1)*frametime_ms*1000))]
            #Create a 2d histogram out of this
            self.hist_xy = utilsHelper.SumPolarity(events_this_frame)
            #Add it to our image
            preview_multiD_image.append(self.hist_xy.dist2D)
            
        #add this image to the napariviewer
        self.napariviewer.add_image(np.asarray(preview_multiD_image), multiscale=False)
        
        #Create the finding/fitting overlays or reset them to zero
        self.finding_overlays = {}
        self.fitting_overlays = []
        #Create an empty finding and fitting result overlay for each layer:
        for n in range(0,n_frames):
            logging.info('Creating overlays ' + str(n/n_frames))
            #Get the finding result in the current time bin:
            findingResults_thisbin = {}
            for findres_id in range(len(findingResult)):
                findres = findingResult[findres_id]
                if max(findres['events']['t']) >= (float(timeStretch[0])*1000+n*frametime_ms*1000) and min(findres['events']['t']) < (float(timeStretch[0])*1000+(n+1)*frametime_ms*1000):
                    findingResults_thisbin[findres_id] = findres
            #Create an overlay from this
            self.finding_overlays[n] = self.create_finding_overlay(findingResults_thisbin)
            
            #Get the fitting result in the current time bin:
            fittingResults_thisbin = fittingResult[(fittingResult['t']>=(float(timeStretch[0])+n*frametime_ms))* (fittingResult['t']<(float(timeStretch[0])+(n+1)*frametime_ms))]
            self.fitting_overlays.append(self.create_fitting_overlay(fittingResults_thisbin,pxsize=settings['PixelSize_nm']['value']))
        
        #Select the original image-layer as selected
        self.napariviewer.layers.selection.active = self.napariviewer.layers[0]
        self.update_visibility()

    def create_finding_overlay(self,findingResults):
        """
        Creation of a finding overlay

        Args:
            findingResults (dict): The finding results as given by Eve

        Returns:
            self.shapes_layer (napari layer): a napari shapes layer with the finding boxes
        """
        #Loop over the finding results, and create a polygon for each, then show these:
        polygons = []
        candidates_ids = []
        
        for f in findingResults:
            #Get the bounding box in xy:
            y_min = min(findingResults[f]['events']['x']) - self.hist_xy.xlim[0]
            y_max = max(findingResults[f]['events']['x']) - self.hist_xy.xlim[0]
            x_min = min(findingResults[f]['events']['y']) - self.hist_xy.ylim[0]
            x_max = max(findingResults[f]['events']['y']) - self.hist_xy.ylim[0]
            candidates_ids.append(f)

            polygons.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
            
        
        # create features
        features = {
            'candidate_ids': candidates_ids,
        }
        text = {
            'string': '{candidate_ids:0.0f}',
            'anchor': 'upper_left',
            'translation': [0, 0],
            'size': 8,
            'color':'coral'
        }
        # Initialize an empty shapes layer for annotations
        self.shapes_layer = self.napariviewer.add_shapes(polygons, shape_type='rectangle', edge_width=1,edge_color='coral',face_color='transparent',visible=False,name='Finding Results',features=features,text=text)
        self.shapes_layer.opacity = 0.7
        
        return self.shapes_layer
        
    def create_fitting_overlay(self,fittingResults,pxsize=80):
        """
        Creation of a fitting overlay

        Args:
            fittingResults (dict): The fitting results as given by Eve
            pxsize (float): The pixel size in nm. Defaulta to 80

        Returns:
            self.shapes_layer (napari layer): a napari shapes layer with the fitting crosses
        """
        
        polygons = []
        
        for f in range(len(fittingResults)):
            xcoord_pxcoord = fittingResults['x'].iloc[f]/pxsize-self.hist_xy.xlim[0]
            ycoord_pxcoord = fittingResults['y'].iloc[f]/pxsize-self.hist_xy.ylim[0]
            
            polygons.append(np.array([[ycoord_pxcoord-1,xcoord_pxcoord-1],[ycoord_pxcoord+1,xcoord_pxcoord+1]]))
            polygons.append(np.array([[ycoord_pxcoord-1,xcoord_pxcoord+1],[ycoord_pxcoord+1,xcoord_pxcoord-1]]))
            
        
        
        # Initialize an empty shapes layer for annotations
        self.shapes_layer = self.napariviewer.add_shapes(polygons, shape_type='line', edge_width=1,edge_color='red',face_color='transparent',visible=False,name='Fitting Results')
        self.shapes_layer.opacity = 0.7
        
        return self.shapes_layer
    
    def update_visibility(self):
        """
        Function to update visibility based on the current layer - the finding/fitting overlays are one for each time-step. All are hidden except the one that's wanted
        """
        #Disable all
        if not not self.finding_overlays: #worst syntax ever to check if a dictionary is empty or not
            for n in range(self.maxFrames):
                if self.finding_overlays[n].visible:
                    self.finding_overlays[n].visible = False
                if self.fitting_overlays[n].visible:
                    self.fitting_overlays[n].visible = False
            #Set current dim visible
            self.finding_overlays[self.napariviewer.dims.current_step[0]].visible=True
            self.fitting_overlays[self.napariviewer.dims.current_step[0]].visible=True
            
        
        # print('wopp'+str(self.napariviewer.dims.current_step[0]))