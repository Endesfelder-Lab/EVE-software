


#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator, QColor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider, QSpacerItem, QTableView, QFrame, QScrollArea, QProgressBar, QMenu, QMenuBar, QColorDialog
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile, QThread, pyqtSignal, QObject
import sys
import typing
from multiprocessing import Manager

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QWidget, QGridLayout, QPushButton

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
if platform.system() == 'Linux':
    # Add hdf5 plugin to read metavision hdf5 format encoded with ECF codec (see https://docs.prophesee.ai/stable/data/file_formats/hdf5.html)
    # default hdf5 plugin path: b'/usr/local/hdf5/lib/plugin' remains unchanged index 0 (h5py.h5pl.get(0))
    h5py.h5pl.append(b"/usr/lib/x86_64-linux-gnu/hdf5/plugins") # path for Ubuntu 20.04 (within Metavision SDK installation)
    h5py.h5pl.append(b"/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins") # path for Ubuntu 22.04 (within Metavision SDK installation)
import traceback
import re
from joblib import Parallel, delayed
from joblib import parallel_backend, cpu_count


from napari import Viewer
from napari.qt import QtViewer
from napari.layers import Image, Shapes
from napari.utils.events import Event
from vispy.color import Colormap

#Custom imports
# Add the folder 2 folders up to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    #Import all scripts in the custom script folders
    # List all files in the CandidateFitting directory
    from eve_smlm.CandidateFitting import *
    from eve_smlm.CandidateFinding import *
    from eve_smlm.Visualisation import *
    from eve_smlm.PostProcessing import *
    from eve_smlm.CandidatePreview import *
    from eve_smlm.DataAnalysis import *
    from eve_smlm.Utils import *

    #Obtain the helperfunctions
    from eve_smlm.Utils import utils, utilsHelper

    #Obtain eventdistribution functions
    from eve_smlm.EventDistributions import eventDistributions
except ImportError:
    #Import all scripts in the custom script folders
    # List all files in the CandidateFitting directory
    from CandidateFitting import *
    from CandidateFinding import *
    from Visualisation import *
    from PostProcessing import *
    from CandidatePreview import *
    from DataAnalysis import *
    from Utils import *

    #Obtain the helperfunctions
    from Utils import utils, utilsHelper

    #Obtain eventdistribution functions
    from EventDistributions import eventDistributions

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
        
        #Flag if the user wants to (gracefully) abort the analysis
        global abortFlag
        abortFlag = Manager().Value('abortFlag', False)

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
        self.data['MetaDataOutput'] = ''
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
        self.setWindowTitle("EVE")
        self.setMinimumSize(400, 400)  # Set minimum size for the GUI window

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a widget to hold the layout
        content_widget = QWidget()

        # Create a layout for the content widget
        self.layout = QGridLayout()
        content_widget.setLayout(self.layout)

        # Set the content widget to the scroll area
        scroll_area.setWidget(content_widget)

        # Create a central widget for the main window
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addWidget(scroll_area)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        self.polarityDropdownNames = ['All events treated equal','Positive only','Negative only','Pos and neg seperately']
        self.polarityDropdownOrder = [0,1,2,3]

        #Initialise (but not show!) advanced window:
        self.advancedSettingsWindow = AdvancedSettingsWindow(self)

        """
        Main tab widget (containing processing, post-processing etc tabs)
        """
        #Create a tab widget and add this to the main group box
        self.mainTabWidget = QTabWidget()
        self.mainTabWidget.setTabPosition(QTabWidget.South)
        self.layout.addWidget(self.mainTabWidget, 0, 0)

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
        self.tab_dataAnalysis = QWidget()
        self.mainTabWidget.addTab(self.tab_dataAnalysis, "Data Analysis")

        # Set up the tabs
        self.setup_tab('Processing')
        self.setup_tab('Post-processing')
        self.setup_tab('LocalizationList')
        self.setup_tab('Visualisation')
        self.setup_tab('Run info')
        self.setup_tab('Preview visualisation')
        self.setup_tab('Candidate preview')
        self.setup_tab('Data Analysis')

        #Loop through all combobox states briefly to initialise them (and hide them)
        self.set_all_combobox_states()

        #Load the GUI settings from last time:
        self.load_entries_from_json()
        
        #Resize the GUI based from GUI settings
        self.resize(float(self.globalSettings['GUIWindowWize']['value'][2]), float(self.globalSettings['GUIWindowWize']['value'][3]))
        self.move(float(self.globalSettings['GUIWindowWize']['value'][0]), float(self.globalSettings['GUIWindowWize']['value'][1]))

        #Initialise polarity value thing (only affects very first run I believe):
        self.polarityDropdownChanged()
        
        #Initialise ToolBar (File,Edit-toolbar etc)
        self.createToolBar()
        
        #Initialise empty dictionaries for storage
        self.data['FindingResult'] = {}
        self.data['FindingResult'][0] = {}
        self.data['FindingResult'][1] = []
        self.data['FittingResult'] = {}
        self.data['FittingResult'][0] = {}
        self.data['FittingResult'][1] = []

        # Initialize empty dataframes for average PSF
        self.data['AveragePSFpos'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        self.data['AveragePSFneg'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        self.data['AveragePSFmix'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        self.data['avg_candidates_mix'] = 0
        self.data['avg_candidates_pos'] = 0
        self.data['avg_candidates_neg'] = 0
        self.data['avg_cluster_size_mix'] = np.zeros(3)
        self.data['avg_cluster_size_pos'] = np.zeros(3)
        self.data['avg_cluster_size_neg'] = np.zeros(3)

        self.uselessWarningSupression()

        
        logging.info('Initialisation complete.')

    def uselessWarningSupression(self):
        #Script to supress warnings that we really don't care about
        import warnings

        # Define a filter function to suppress the specific warning
        def warning_filter(message, category, filename, lineno, file=None, line=None):
            patterns = [
                "QCoreApplication::exec: The event loop is already running"
                "QPainter::begin: Paint device returned engine == 0, type: 3"
            ]
            
            # Compile regex patterns
            compiled_patterns = [re.compile(pattern) for pattern in patterns]
            for pattern in compiled_patterns:
                if pattern.match(str(message)):
                    return False
            return True

        warnings.showwarning = warning_filter

    def createToolBar(self):
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)
        settingsMenu = menuBar.addMenu("&Settings")
        openAdvancedSettings = settingsMenu.addAction("Advanced settings")
        openAdvancedSettings.triggered.connect(self.open_advanced_settings)
        settingsMenu.addSeparator()
        saveGUIcontents = settingsMenu.addAction("Save GUI contents")
        saveGUIcontents.triggered.connect(self.save_entries_to_json)
        loadGUIcontents = settingsMenu.addAction("Load last GUI contents")
        loadGUIcontents.triggered.connect(self.load_entries_from_json)
        settingsMenu.addSeparator()
        saveGUIcontents = settingsMenu.addAction("Save-as GUI contents")
        saveGUIcontents.triggered.connect(self.saveas_entries_to_json)
        loadGUIcontents = settingsMenu.addAction("Load specific GUI contents")
        loadGUIcontents.triggered.connect(self.loadas_entries_from_json)
        settingsMenu.addSeparator()
        changeColorAction = settingsMenu.addAction("Change appearance color")
        changeColorAction.triggered.connect(self.changeAppearanceColor)
        
        #Create a new menu, and add actions corresponding to all utilsFunctions:
        utilsMenu = menuBar.addMenu("Utilities")
        utilsFunctions = utils.functionNamesFromDir('Utils')
        #Connect all found functions to this dropdown
        utilsDisplayNames = utils.displayNamesFromFunctionNames(utilsFunctions,'')
        utilActions = {}
        for i, utilsFunction in enumerate(utilsFunctions):
            logging.debug(utilsFunction)
            utilActions[i] = utilsMenu.addAction(utilsDisplayNames[0][i])
            #Run "function(self)" when triggered - passing self to the function 
            utilActions[i].triggered.connect(lambda _, s=self, func=utilsFunction: eval(func+'(s)'))
        
        
        helpMenu = menuBar.addMenu("Help")
        quickStartMenu = helpMenu.addAction("Quick start / User Manual")
        quickStartMenu.triggered.connect(lambda: self.quickStartMenu())
        devManualMenu = helpMenu.addAction("Developer manual")
        devManualMenu.triggered.connect(lambda: self.devManualMenu())
        sciBGMenu = helpMenu.addAction("Scientific information")
        sciBGMenu.triggered.connect(lambda: self.sciBGMenu())
        aboutMenu = helpMenu.addAction("About...")
        aboutMenu.triggered.connect(lambda: self.aboutMenu())

    def quickStartMenu(self):
        """
        Shows the README.md file
        """
        try:
            quickStartWindow = utils.SmallWindow(self)
            QApplication.processEvents()
            quickStartWindow.setWindowTitle('Quick start / User Manual')
            QApplication.processEvents()
            from pathlib import Path
            current_dir = Path(__file__).parent
            quickStartWindow.addMarkdown(str(current_dir)+os.sep+'README.md')
            QApplication.processEvents()
            quickStartWindow.show()
        except Exception as e:
            logging.error(f'Could not open quick start window. {e}')
    
    def sciBGMenu(self):
        """
        Shows the ScientificBackground.md file
        """
        try:
            sciBGWindow = utils.SmallWindow(self)
            QApplication.processEvents()
            sciBGWindow.setWindowTitle('Scientific background')
            QApplication.processEvents()
            from pathlib import Path
            current_dir = Path(__file__).parent
            sciBGWindow.addMarkdown(str(current_dir)+os.sep+'ScientificBackground.md')
            QApplication.processEvents()
            sciBGWindow.show()
        except:
            logging.error('Could not open quick start window.')
    
    def documentationMenu(self):
        """
        Opens the user manual .pdf externally
        """
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        url = QUrl.fromLocalFile('UserManual.pdf')
        QDesktopServices.openUrl(url)
    def devManualMenu(self):
        """
        Opens the developer manual .pdf externally
        """
        try:
            DevManWindow = utils.SmallWindow(self)
            QApplication.processEvents()
            DevManWindow.setWindowTitle('Developer Manual')
            QApplication.processEvents()
            from pathlib import Path
            current_dir = Path(__file__).parent
            DevManWindow.addMarkdown(str(current_dir)+os.sep+'DeveloperInstructions.md')
            QApplication.processEvents()
            DevManWindow.show()
        except:
            logging.error('Could not open Developer manual window.')
        
    def aboutMenu(self):
        """
        Shows a small about-window
        """
        aboutWindow = utils.SmallWindow(self)
        aboutWindow.setWindowTitle('About')
        aboutWindow.addDescription('<b>EVE is created by</b> <br>Laura Weber, Koen J.A. Martens, Ulrike Endesfelder<br>Institute of Microbiology and Biotechnology<br>University of Bonn<br><br><b>Contact</b><br>koenjamartens@gmail.com<br>endesfelder@uni-bonn.de<br><br><b>Cite</b> <br>Please cite EVE as [TBD]')
        aboutWindow.show()

    def changeAppearanceColor(self):
        #Function that changes the appearance color
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(self.get_stylesheet(accentColor = color.name()))
    
    def adjust_color_brightness(self,hex_color, percent):
        # Convert hexadecimal color to RGB
        rgb_color = tuple(int(hex_color[1+i:1+i+2], 16) for i in (0, 2, 4))
        # Convert HSL color back to RGB 
        rgb_color = tuple(int(round(i + i * percent / 100)) for i in rgb_color)
        # Bound each element between 0 and 255
        rgb_color = tuple(max(0, min(255, i)) for i in rgb_color)
        # Convert RGB color to hexadecimal
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
        
        return hex_color

    def get_stylesheet(self,accentColor= '#d5d5e5'  ):
        # External variables for colors
        background_color = '#f8f8f8'
        tab_pane_background_color = '#585858'
        accent_color = accentColor
        accent_color_darker = self.adjust_color_brightness(accent_color, -15)#'#b0b0e5'
        text_color = '#333333'
        border_radius = '0px'
        border_width = '1px'
        padding_small = '1px'#'2px'
        padding_medium = '2px'#'4px'
        padding_large = '3px'#'6px'
        margin_small = '1px'#'2px'
        margin_medium = '3px'#'5px'
        margin_between_groupbox_and_entry = '8px'  # Adjust as needed

        # Stylesheet
        stylesheet = (
            f"QWidget {{ margin: {margin_small}; }}"
            f"QGroupBox {{ background-color: {background_color}; }}"
            f"QLayout {{ background-color: {background_color}; }}"

            f"QLabel, QLineEdit, QComboBox, QCheckBox {{ color: {text_color}; }}"
            f"QPushButton {{ background-color: {accent_color}; color: {text_color}; "
            f"border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_large}; }}"
            f"QPushButton:hover {{ background-color: {accent_color_darker}; }}"
            f"QPushButton:pressed {{ background-color: {accent_color}; }}"
            f"QLineEdit {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_small}; }}"
            f"QComboBox {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_small}; }}"
            f"QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right;}}"
            f"QComboBox QAbstractItemView {{ selection-background-color: #0078D7; selection-color: {text_color};}}" 
            f"QTabWidget::pane {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; margin: 0; }}"
            f"QTabBar::tab {{ background-color: {background_color}; color: {text_color}; border: {border_width} solid {accent_color}; border-top: none; border-bottom-left-radius: {border_radius}; border-bottom-right-radius: {border_radius}; padding: {padding_medium}; margin-right: {margin_small}; }}"
            f"QTabBar::tab:selected {{ background-color: {accent_color_darker}; color: {text_color}; border-top: {border_width} solid {accent_color}; }}"
            f"QTabBar::tab:hover {{ background-color: {accent_color}; color: {text_color}; }}"
            f"QGroupBox {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; margin-top: {margin_between_groupbox_and_entry}; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: {margin_small}; padding: 0 {padding_medium}; background-color: {accent_color}; color: {text_color}; border: {border_width} solid {accent_color}; border-radius: {border_radius}; }}"
            
        )

        return stylesheet

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
            'Candidate preview': self.setup_canPreviewTab,
            'Data Analysis': self.setup_dataAnalysisTab
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
        globalSettings['HotPixelIndexes'] = {}
        globalSettings['HotPixelIndexes']['value'] = ""
        globalSettings['HotPixelIndexes']['input'] = str
        globalSettings['HotPixelIndexes']['displayName'] = 'Hot-pixel indeces'
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
        globalSettings['UseCUDA'] = {}
        globalSettings['UseCUDA']['value'] = False
        globalSettings['UseCUDA']['input'] = bool
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
        globalSettings['GUIWindowWize'] = {}
        globalSettings['GUIWindowWize']['value'] = [0,0,400,400]
        globalSettings['GUIWindowWize']['input'] = list
        

        #Add options here that should NOT show up in the global settings window - i.e. options that should not be changed
        globalSettings['IgnoreInOptions'] = ('IgnoreInOptions','StoreFinalOutput', 'JSONGUIstorePath','GlobalOptionsStorePath','GUIWindowWize','XYTOutlierRemoval','XYTOutlierRemoval_multiplier')
        return globalSettings
    
        #For prosperity
        # globalSettings['XYTOutlierRemoval'] = {}
        # globalSettings['XYTOutlierRemoval']['value'] = False
        # globalSettings['XYTOutlierRemoval']['input'] = bool
        # globalSettings['XYTOutlierRemoval']['displayName'] = 'XYTOutlierRemoval'
        # globalSettings['XYTOutlierRemoval_multiplier'] = {}
        # globalSettings['XYTOutlierRemoval_multiplier']['value'] = 2.5
        # globalSettings['XYTOutlierRemoval_multiplier']['input'] = float
        # globalSettings['XYTOutlierRemoval_multiplier']['displayName'] = 'STD multiplier for XYTOutlierRemoval'
        

    def datasetSearchButtonClicked(self):
        # Function that handles the dataset 'File' lookup button
        logging.debug('data lookup search button clicked')
        parentFolder = self.dataLocationInput.text()
        if parentFolder != "":
            parentFolder = os.path.dirname(parentFolder)
        file_path = utils.generalFileSearchButtonAction(parent=self,text="Select File",filter="EBS files (*.raw *.npy *hdf5);;All Files (*)",parentFolder=parentFolder)
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

            #Set the mapping of the fitting functions...
            setattr(self, Fitting_functionNameToDisplayNameMapping_name, Fitting_functionNameToDisplayNameMapping)

            #Set the mapping & descriptions of the dist/time functions...
            selectedFunction = utils.functionNameFromDisplayName(candidateFittingDropdown.currentText(),Fitting_functionNameToDisplayNameMapping)
            [self.timeFitValues, self.timeFit_displayNames, self.timeFit_name_to_displayName_map, self.timeFit_descriptions] = utils.classKwargValuesFromFittingFunction(selectedFunction, 'time')
            [self.distKwargValues, self.distKwarg_displayNames, self.distKwarg_name_to_displayName_map, self.distKwarg_descriptions] = utils.classKwargValuesFromFittingFunction(selectedFunction, 'dist')

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
        
        # self.buttonProcessingAbort = QPushButton("Abort")
        # self.buttonProcessingAbort.clicked.connect(lambda: self.abort_processing())
        # self.runLayout.layout().addWidget(self.buttonProcessingAbort,2,7,1,6)

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
        self.previewLayout.layout().addWidget(QLabel("Display frame time (ms):"), 0,2,1,1)
        self.previewLayout.layout().addWidget(QLabel("Display event type:"), 0,3,1,1)
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
        self.previewLayout.layout().addWidget(self.preview_displayFrameTime, 1, 2, 1, 1)
        self.preview_displayFrameTime.setText("100")
        #And for display event type:
        self.preview_eventType = QComboBox()
        self.preview_eventType.setObjectName('preview_eventType')
        self.previewLayout.layout().addWidget(self.preview_eventType, 1, 3, 1, 1)
        self.preview_eventType.addItem("All events")
        self.preview_eventType.addItem("All events treated equal")
        self.preview_eventType.addItem("Positive only")
        self.preview_eventType.addItem("Negative only")
        self.preview_eventType.setCurrentIndex(0)

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
        self.buttonPreview = QPushButton("Preview finding/fitting")
        self.buttonPreview.setToolTip("Perform the finding/fitting routines on the subset determined by the Preview groupbox, and visualise it in the Preview run tab.")
        #Add a button press event:
        self.buttonPreview.clicked.connect(lambda: self.previewRun(
            (self.preview_startTLineEdit.text(), self.preview_durationTLineEdit.text()),
            (self.preview_minXLineEdit.text(),self.preview_maxXLineEdit.text(),
            self.preview_minYLineEdit.text(),self.preview_maxYLineEdit.text()),
            float(self.preview_displayFrameTime.text()),eventType=self.preview_eventType.currentText()))
        #Add the button to the layout:
        self.previewLayout.layout().addWidget(self.buttonPreview, 4, 0, 1, 2)
        
        
        #Add a 'preview events only' button:
        self.buttonEventsPreview = QPushButton("Preview events")
        self.buttonEventsPreview.setToolTip("Only look at the events in this spatiotemporal window, don't do any fitting")
        #Add a button press event:
        self.buttonEventsPreview.clicked.connect(lambda: self.previewEventsCall((self.preview_startTLineEdit.text(),
                self.preview_durationTLineEdit.text()),
                (self.preview_minXLineEdit.text(),self.preview_maxXLineEdit.text(),
                self.preview_minYLineEdit.text(),self.preview_maxYLineEdit.text()),
                float(self.preview_displayFrameTime.text()),eventType=self.preview_eventType.currentText()))
        #Add the button to the layout:
        self.previewLayout.layout().addWidget(self.buttonEventsPreview, 4, 2, 1 ,2)

    def abort_processing(self):
        abortFlag.value = True
        self.analysis_stop_button.setText("Stopping...")
        #set button inactive as well:
        self.analysis_stop_button.setEnabled(False)

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
            time0 = time.time()
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

            time1 = time.time()
            #Properly (32bit) initialise dicts
            wantedEvents_tooLarge = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})
            events_output = np.zeros(0, dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 32})
            time2 = time.time()
            #Now we know the start/end index, so we cut out that area:
            wantedEvents_tooLarge = events_hdf5[lookup_start_index:lookup_end_index]
            
            time3 = time.time()
        #And we fully cut it to exact size:
        events_output = wantedEvents_tooLarge[(wantedEvents_tooLarge['t'] >= requested_start_time_ms*1000) & (wantedEvents_tooLarge['t'] <= requested_end_time_ms*1000)]
        time4 = time.time()
        

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

    def setup_dataAnalysisTab(self):
        """
        Function that set up the data analysis tab
        """
        tab_layout = QGridLayout()
        self.tab_dataAnalysis.setLayout(tab_layout)
        #Dataset location and searching Grid Layout
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
        self.dataAnalysistab_widget = DataAnalysisWidget(self)
        self.tab_dataAnalysis.layout().addWidget(self.dataAnalysistab_widget)

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
        
        #Add an analysis-progress-bar + stopbutton
        self.analysis_stop_button = QPushButton("Stop")
        self.analysis_stop_button.clicked.connect(lambda: self.abort_processing())
        #Set it to have a fixed width:
        self.analysis_stop_button.setFixedWidth(75)
        
        self.analysis_progressbar = QProgressBar()
        self.analysis_progressbar.setValue(100)
        
        qh = QHBoxLayout()
        qh.addWidget(self.analysis_progressbar)
        qh.addWidget(self.analysis_stop_button)
        #add qh to a widget:
        qh2 = QWidget()
        qh2.setLayout(qh)
        
        tab_layout.addWidget(qh2, 1, 0)

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
        if fname[0] != '':
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
            loclist.rename(columns={'Δx [nm]' : 'del_x'}, inplace=True)
            loclist.rename(columns={'Δy [nm]' : 'del_y'}, inplace=True)
            loclist.rename(columns={'Δt [nm]' : 'del_t'}, inplace=True)
            loclist.rename(columns={'x_dim [px]' : 'x_dim'}, inplace=True)
            loclist.rename(columns={'y_dim [px]' : 'y_dim'}, inplace=True)
            loclist.rename(columns={'t_dim [ms]' : 't_dim'}, inplace=True)

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
        if len(self.data['FittingResult'][0]) > 0:
            localizations = self.data['FittingResult'][0].dropna(axis=0, ignore_index=True)
            try:
                localizations = localizations.drop('fit_info', axis=1)
            except:
                pass

            # Define the number of significant digits for each column
            significant_digits = {'x': 2, 'y': 2, 'p': 0, 'id':0,'candidate_id':0,'del_x':2,'del_y':2, 't':2, 'del_t':2, 'N_events':0, 'x_dim':0, 'y_dim':0, 't_dim':2, 'sigma_x': 2, 'sigma_y': 2}

            for y in range(len(localizations.columns)):
                significant_digit = significant_digits.get(localizations.columns[y])
                if significant_digit is not None:
                    localizations[localizations.columns[y]] = localizations[localizations.columns[y]].apply(lambda x: round(x, significant_digit))

            #Replace common headers
            header_data = localizations.columns.tolist()
            changes = [
                ['x','x [nm]'],
                ['y','y [nm]'],
                ['del_x','Δ x [nm]'],
                ['del_y','Δ y [nm]'],
                ['t','t [ms]'],
                ['del_t','Δ t [ms]'],
                ['x_dim','x_dim [px]'],
                ['y_dim','y_dim [px]'],
                ['t_dim','t_dim [ms]'],
                ['p','polarity'],
            ]
            for change in changes:
                header_data = [change[1] if item == change[0] else item for item in header_data]


            self.LocListTable.setModel(TableModel(table_data = localizations, header_names = header_data))
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

    def RawToNpy(self,filepath,buffer_size = 1e8, n_batches=1e7):
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
            if self.globalSettings['StoreConvertedRawData']['value'] > 0:
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

    def removeStorageBasedOnPickle(self,polarityVal):
        #Remove the self.store option when pickle is loaded
        #FINDING:  if ALL polarity finding routines are loading pickle, the intermediate finding saving is disabled
        #Check if all (both pos and neg if needed) finding is loading pickles:
        if polarityVal != 'Both':
            if 'LoadExistingFinding' in self.getFunctionEvalText('Finding','','',polarityVal):
                self.globalSettings['StoreFindingOutput']['value'] = 0
                logging.info('Disabled findingoutput storage because pickle is loaded')
        elif polarityVal == 'Both':
            if 'LoadExistingFinding' in self.getFunctionEvalText('Finding','','','Neg') and 'LoadExistingFinding' in self.getFunctionEvalText('Finding','','','Pos'):
                self.globalSettings['StoreFindingOutput']['value'] = 0
                logging.info('Disabled findingoutput storage because pickle is loaded')
        
        #FITTING: if ALL polarity finding AND fitting routines are loading pickle, the final fitting saving is disabled
        if polarityVal != 'Both':
            if 'LoadExistingFinding' in self.getFunctionEvalText('Finding','','',polarityVal):
                if 'LoadExistingFitting' in self.getFunctionEvalText('Fitting','','',polarityVal):
                    self.globalSettings['StoreFittingOutput']['value'] = 0
                    self.globalSettings['StoreFinalOutput']['value'] = 0
                    self.globalSettings['StoreFileMetadata']['value'] = 0
                    logging.info('Disabled fittingoutput storage because pickle is loaded')
        elif polarityVal == 'Both':
            if 'LoadExistingFinding' in self.getFunctionEvalText('Finding','','','Neg') and 'LoadExistingFinding' in self.getFunctionEvalText('Finding','','','Pos'):
                if 'LoadExistingFitting' in self.getFunctionEvalText('Fitting','','','Neg') and 'LoadExistingFitting' in self.getFunctionEvalText('Fitting','','','Pos'):
                    self.globalSettings['StoreFittingOutput']['value'] = 0
                    self.globalSettings['StoreFinalOutput']['value'] = 0
                    self.globalSettings['StoreFileMetadata']['value'] = 0
                    logging.info('Disabled fittingoutput storage because pickle is loaded')

    def run_processing_i(self):
        self.globalSettingsBeforeRun = copy.deepcopy(self.globalSettings)
        
        #Get polarity info:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[0]:
            polarityVal = 'Mix'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[1]:
            polarityVal = 'Pos'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[2]:
            polarityVal = 'Neg'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            polarityVal = 'Both'
            
        #Change savingbehaviour based on pickle loading
        self.removeStorageBasedOnPickle(polarityVal)
        
        #Add relevant info:
        self.findingAnalysis.set_globalSettings(self.globalSettings)
        self.findingAnalysis.set_GPU(False)
        self.findingAnalysis.set_parrallellized(utilsHelper.strtobool(self.globalSettings['Multithread']['value']))
        self.findingAnalysis.set_fileLocation(self.currentFileInfo['CurrentFileLoc'])
        self.findingAnalysis.set_GUIinfo(self) #Pass the GUI info to the finding analysis
        self.findingAnalysis.set_chunkingTime([float(self.globalSettings['FindingBatchingTimeMs']['value']),float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])])

        self.findingAnalysis.set_polarityAnalysis(polarityVal)
        
        
        #'Run', not 'preview'
        self.findingAnalysis.set_timeStretchMs([float(self.run_startTLineEdit.text()),float(self.run_startTLineEdit.text())+float(self.run_durationTLineEdit.text())])
        self.findingAnalysis.set_xyStretch([[float(self.run_minXLineEdit.text()),float(self.run_maxXLineEdit.text())],[float(self.run_minYLineEdit.text()),float(self.run_maxYLineEdit.text())]])
        
        #Run the analysis
        self.findingAnalysis.analyse()
        
        #Access the results and store
        self.data['FindingResult'] = self.findingAnalysis.Results
        
        #Next, all fitting:
        #Add relevant info:
        self.fittingAnalysis.set_globalSettings(self.globalSettings)
        self.fittingAnalysis.set_GPU(False)
        self.fittingAnalysis.set_parrallellized(utilsHelper.strtobool(self.globalSettings['Multithread']['value']))
        self.fittingAnalysis.set_polarityAnalysis(polarityVal)
        self.fittingAnalysis.set_findingResult(self.findingAnalysis.Results)
        self.fittingAnalysis.set_GUIinfo(self) #Pass the GUI info to the finding analysis
        #Run the analysis
        self.fittingAnalysis.analyse()
        #Access the results and store
        self.data['FittingResult'] = self.fittingAnalysis.Results

        # reset average PSF after new results
        self.data['AveragePSFpos'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        self.data['AveragePSFneg'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        self.data['AveragePSFmix'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        
        return

    def run_processing(self,error=None):
        #Give a clear logging separation:
        logging.info("")
        logging.info("")
        logging.info("--------------- Full Run Starting ---------------")
        logging.info("")
        logging.info("")
        
        
        self.analysis_progressbar.setValue(0)
        #Reset the abort to allow for aborting
        abortFlag.value = False
        self.analysis_stop_button.setText("Stop")
        self.analysis_stop_button.setEnabled(True)
        
        self.data['MetaDataOutput'] = ''
        
        #Switch the user to the Run info tab
        utils.changeTab(self, text='Run info')
        
        #Ensure final output is stored
        self.globalSettings['StoreFinalOutput']['value'] = True

        # reset previewEvents array, every time run is pressed
        self.previewEvents = []
        
        #Check if the input is a .hdf5, .npy or .raw:
        if self.dataLocationInput.text().endswith(('.hdf5', '.npy', '.raw')):
            #Create the finding structure:
            self.FindingCompleted = False
            self.findingAnalysis = FindingAnalysis()
            #Create the finding structure:
            self.FittingCompleted = False
            self.fittingAnalysis = FittingAnalysis()
            self.signalEmitArray = None
            
            #Prepare for saving/storing results later...
            self.currentFileInfo['CurrentFileLoc'] = self.dataLocationInput.text()
            self.storeNameDateTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
            #Run the processing on a different thread for GUI proper working
            thread = threading.Thread(target=self.run_processing_i)
            thread.start()
            # self.run_processing_i()
            
            #Visually show we're started by updating the progress bar to 5%
            self.updateProgressBar(overwriteValue = 5)
            
            self.analysisStartTime = time.time()
            #Await completion of finding  - i need to do this, since calling/emitting requires a QObject, which cannot be pickled for threading.
            while self.FindingCompleted == False and abortFlag.value == False:
                self.updateProgressBar(findorfit='find')
                QApplication.processEvents() #continue as normal
            if abortFlag.value == True:
                logging.warning("Stopping request requested... EVE is stopping soon.")
                return
            self.findingAnalysisComplete() #Run this function once finding is completed
            self.updateProgressBar(findorfit='find')
            
            while self.FittingCompleted == False and abortFlag.value == False:
                self.updateProgressBar(findorfit='fit')
                QApplication.processEvents() #continue as normal
            if abortFlag.value == True:
                logging.warning("Stopping request requested... EVE is stopping soon.")
                return
            self.fittingAnalysisComplete() #Run this function once fitting is completed
            self.updateProgressBar(findorfit='fit')
        elif os.path.isdir(self.dataLocationInput.text()):
            #Find all files in the directory which end in .hdf5, .npy or .raw:
            fileList = [f for f in os.listdir(self.dataLocationInput.text()) if f.endswith(('.hdf5', '.npy', '.raw'))]
            for index, file in enumerate(fileList):
                fullPath = os.path.join(self.dataLocationInput.text(), file)
                try:
                    
                    logging.info("Starting batch run of file: " + fullPath)
                    currentVisProgress = index / len(fileList) * 90+5
                    
                    #Create the finding structure:
                    self.FindingCompleted = False
                    self.findingAnalysis = FindingAnalysis()
                    #Create the finding structure:
                    self.FittingCompleted = False
                    self.fittingAnalysis = FittingAnalysis()
                    self.signalEmitArray = None
                    
                    #Prepare for saving/storing results later...
                    self.currentFileInfo['CurrentFileLoc'] = fullPath
                    self.storeNameDateTime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                    #Run the processing on a different thread for GUI proper working
                    thread = threading.Thread(target=self.run_processing_i)
                    thread.start()
                    # self.run_processing_i()
                    
                    #Visually show we're started by updating the progress bar to 5%
                    self.updateProgressBar(overwriteValue = currentVisProgress)
                    
                    currentVisProgress = (index+.25) / len(fileList) * 90+5
                    self.analysisStartTime = time.time()
                    #Await completion of finding  - i need to do this, since calling/emitting requires a QObject, which cannot be pickled for threading.
                    while self.FindingCompleted == False and abortFlag.value == False:
                        self.updateProgressBar(overwriteValue = currentVisProgress)
                        QApplication.processEvents() #continue as normal
                    if abortFlag.value == True:
                        logging.warning("Stopping request requested... EVE is stopping soon.")
                        return
                    self.findingAnalysisComplete() #Run this function once finding is completed
                    self.updateProgressBar(overwriteValue = currentVisProgress)
                    
                    currentVisProgress = (index+.75) / len(fileList) * 90+5
                    while self.FittingCompleted == False and abortFlag.value == False:
                        self.updateProgressBar(overwriteValue = currentVisProgress)
                        QApplication.processEvents() #continue as normal
                    if abortFlag.value == True:
                        logging.warning("Stopping request requested... EVE is stopping soon.")
                        return
                    self.fittingAnalysisComplete() #Run this function once fitting is completed
                    self.updateProgressBar(overwriteValue = currentVisProgress)
                    
                    logging.info("Finished batch run of file: " + fullPath)
                    logging.info("-----------------------------")
                except:
                    logging.error("Error encountered in file "+fullpath+", continuing with other files")
            self.updateProgressBar(overwriteValue = 100)
            logging.info("Fully completed all files in folder " + self.dataLocationInput.text())
            self.updateGUIafterNewResults()
    
    def run_preview_i(self,error=None):
        #Get polarity info:
        if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[0]:
            polarityVal = 'Mix'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[1]:
            polarityVal = 'Pos'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[2]:
            polarityVal = 'Neg'
        elif self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
            polarityVal = 'Both'
        
        #Add relevant info:
        self.findingAnalysis.set_globalSettings(self.globalSettings)
        self.findingAnalysis.set_GPU(False)
        self.findingAnalysis.set_parrallellized(utilsHelper.strtobool(self.globalSettings['Multithread']['value']))
        self.findingAnalysis.set_fileLocation(self.dataLocationInput.text())
        self.findingAnalysis.set_GUIinfo(self) #Pass the GUI info to the finding analysis
        self.findingAnalysis.set_chunkingTime([float(self.globalSettings['FindingBatchingTimeMs']['value']),float(self.globalSettings['FindingBatchingTimeOverlapMs']['value'])])
        
        self.findingAnalysis.set_polarityAnalysis(polarityVal)
        
        #'preview', not 'run'
        self.findingAnalysis.set_timeStretchMs([float(self.previewTimeStretch[0]),float(self.previewTimeStretch[0])+float(self.previewTimeStretch[1])])
        #Check if any of previewXYstretch are non-floats:
        xystretchisallfloats = True
        for item in self.previewXYStretch:
            try:
                float(item)
            except:
                xystretchisallfloats = False
        if not xystretchisallfloats:
            self.previewXYStretch = [-np.inf,np.inf,-np.inf,np.inf]
            logging.warning('Set preview XY stretch to -Inf,Inf,-Inf,Inf')
        self.findingAnalysis.set_xyStretch([[float(self.previewXYStretch[0]),float(self.previewXYStretch[1])],[float(self.previewXYStretch[2]),float(self.previewXYStretch[3])]])
        
        #Run the analysis
        self.findingAnalysis.analyse()
        
        #Access the results and store
        self.data['FindingResult'] = self.findingAnalysis.Results
        
        #Next, all fitting:
        #Add relevant info:
        self.fittingAnalysis.set_globalSettings(self.globalSettings)
        self.fittingAnalysis.set_GPU(False)
        self.fittingAnalysis.set_parrallellized(utilsHelper.strtobool(self.globalSettings['Multithread']['value']))
        self.fittingAnalysis.set_polarityAnalysis(polarityVal)
        self.fittingAnalysis.set_findingResult(self.findingAnalysis.Results)
        self.fittingAnalysis.set_GUIinfo(self) #Pass the GUI info to the finding analysis
        #Run the analysis
        self.fittingAnalysis.analyse()
        #Access the results and store
        self.data['FittingResult'] = self.fittingAnalysis.Results
        # reset average PSF after new results
        self.data['AveragePSFpos'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        self.data['AveragePSFneg'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
        self.data['AveragePSFmix'] = pd.DataFrame(columns=['x', 'y', 't', 'p'])
    
    def previewEventsCall(self,timeStretch=(0,1000),xyStretch=(0,0,0,0),frameTime=100,eventType="All events"):
        """
        Generates the preview the events in the time/xy stretch.

        Parameters:
            timeStretch (tuple): A tuple containing the start and end times for the preview.
            xyStretch (tuple): A tuple containing the minimum and maximum x and y coordinates for the preview.
            frametime (int): The frame time in ms.
            eventType (str): "All events", "All events treated equal", "Positive only", "Negative only"

        Returns:
            None
        """
        #Switch the user to the Run info tab
        utils.changeTab(self, text='Run info')
        logging.info("Previewing events")
        self.updateProgressBar(overwriteValue = 5)
        #we need to find the events to display in the preview:
        if self.dataLocationInput.text().endswith('.hdf5'):
            previewEvents,_ = self.timeSliceFromHDF(self.dataLocationInput.text(),requested_start_time_ms = float(timeStretch[0]),requested_end_time_ms=float(timeStretch[0])+float(timeStretch[1]),howOftenCheckHdfTime = 50000)
        elif self.dataLocationInput.text().endswith('.raw'):
            previewEvents = utils.readRawTimeStretch(self.dataLocationInput.text(),self.globalSettings['MetaVisionPath']['value'],buffer_size = 5e7, n_batches=5e7, timeStretchMs=[float(timeStretch[0])*1000,float(timeStretch[1])*1000])
        elif self.dataLocationInput.text().endswith('.npy'):
            #Load the data:
            previewEvents = np.load(self.dataLocationInput.text())
            #constrict to correct time:
            previewEvents = self.filterEvents_npy_t(previewEvents,timeStretch)
        
        #Remove hotpixels
        hotpixelarray = eval("["+self.globalSettings['HotPixelIndexes']['value']+"]")
        previewEvents = utils.removeHotPixelEvents(previewEvents,hotpixelarray)
        
        #Check if we have at least 1 event:
        if len(previewEvents) <= 0:
        #     #Log the nr of events found:
        #     logging.info(f"Preview - Found {len(previewEvents)} events in the chosen time frame.")
        # else:
            logging.error("Preview - No events found in the chosen time frame.")
            return
        
        #Load the events in self memory and filter on XY
        self.previewEvents = previewEvents
        self.previewEvents = self.filterEvents_xy(self.previewEvents,xyStretch)
        
        if eventType == "Positive only":
            self.previewEvents = self.filterEvents_npy_p(self.previewEvents,1)
        elif eventType == "Negative only":
            self.previewEvents = self.filterEvents_npy_p(self.previewEvents,0)
        elif eventType == "All events treated equal":
            #Set all polarities to 1:
            self.previewEvents['p'] = 1
            
        logging.info(f"Preview - Displaying {len(self.previewEvents)} events ({eventType}) in the chosen time frame, no finding/fitting.")
        
        #Update the preview panel and localization list:
        self.updateShowPreview(previewEvents=self.previewEvents,timeStretch=timeStretch,frameTime=frameTime)
        
        self.updateProgressBar(overwriteValue = 100)
        #Switch the user to the Preview tab
        utils.changeTab(self, text='Preview run')
    
    def previewRun(self,timeStretch=(0,1000),xyStretch=(0,0,0,0),frameTime=100,eventType="All events"):
        """
        Generates the preview of a run analysis.

        Parameters:
            timeStretch (tuple): A tuple containing the start and end times for the preview.
            xyStretch (tuple): A tuple containing the minimum and maximum x and y coordinates for the preview.
            frametime (int): The frame time in ms.
            eventType (str): "All events", "Positive only", "Negative only"

        Returns:
            None
        """
        #Switch the user to the Run info tab
        utils.changeTab(self, text='Run info')

        self.updateProgressBar(overwriteValue = 5)
        # Empty the event preview list
        self.previewEvents = []

        #Checking if a file is selected rather than a folder:
        if not os.path.isfile(self.dataLocationInput.text()):
            logging.error("Please choose a file rather than a folder for previews.")
            return
        
        #Give a clear logging separation:
        logging.info("")
        logging.info("")
        logging.info("--------------- Preview Run Starting ---------------")
        logging.info("")
        logging.info("")
        
        #Reset the abort to allow for aborting
        abortFlag.value = False
        self.analysis_stop_button.setText("Stop")
        self.analysis_stop_button.setEnabled(True)
        
        #Ensure final output is stored
        self.globalSettings['StoreFinalOutput']['value'] = False

        # reset previewEvents array, every time run is pressed
        self.previewEvents = []
        
        #Create the finding structure:
        self.FindingCompleted = False
        self.findingAnalysis = FindingAnalysis()
        #Create the finding structure:
        self.FittingCompleted = False
        self.fittingAnalysis = FittingAnalysis()
        
        #Change global values so nothing is stored - we just want a preview run. This is later set back to orig values (globalSettingsOrig):
        globalSettingsOrig = copy.deepcopy(self.globalSettings)
        self.globalSettings['StoreConvertedRawData']['value'] = False
        self.globalSettings['StoreFileMetadata']['value'] = False
        # self.globalSettings['StoreFinalOutput']['StoreFinalOutput']['value'] = False
        self.globalSettings['StoreFinalOutput']['value'] = False
        self.globalSettings['StoreFindingOutput']['value'] = False
        self.globalSettings['StoreFittingOutput']['value'] = False
        
        self.FindingCompleted = False
        self.FittingCompleted = False
        
        self.previewTimeStretch = timeStretch
        self.previewXYStretch = xyStretch
        
        #Run the processing on a different thread for GUI proper working
        thread = threading.Thread(target=self.run_preview_i)
        thread.start()

        #Await completion of finding  - i need to do this, since calling/emitting requires a QObject, which cannot be pickled for threading.
        while self.FindingCompleted == False:
            QApplication.processEvents() #continue as normal
        self.updateProgressBar(findorfit='find')
        while self.FittingCompleted == False:
            QApplication.processEvents() #continue as normal
        self.updateProgressBar(findorfit='fit')
        
        logging.info('previewing finished')
        #Reset global settings
        self.globalSettings = globalSettingsOrig

        #we need to find the events to display in the preview:
        if self.dataLocationInput.text().endswith('.hdf5'):
            previewEvents,_ = self.timeSliceFromHDF(self.dataLocationInput.text(),requested_start_time_ms = float(timeStretch[0]),requested_end_time_ms=float(timeStretch[0])+float(timeStretch[1]),howOftenCheckHdfTime = 50000)
        elif self.dataLocationInput.text().endswith('.raw'):
            previewEvents = utils.readRawTimeStretch(self.dataLocationInput.text(),self.globalSettings['MetaVisionPath']['value'],buffer_size = 5e7, n_batches=5e7, timeStretchMs=[float(timeStretch[0])*1000,float(timeStretch[1])*1000])
        elif self.dataLocationInput.text().endswith('.npy'):
            #Load the data:
            previewEvents = np.load(self.dataLocationInput.text())
            #constrict to correct time:
            previewEvents = self.filterEvents_npy_t(previewEvents,timeStretch)
        
        #Remove hotpixels
        hotpixelarray = eval("["+self.globalSettings['HotPixelIndexes']['value']+"]")
        previewEvents = utils.removeHotPixelEvents(previewEvents,hotpixelarray)
        
        self.updateProgressBar(overwriteValue = 95)
        #Check if we have at least 1 event:
        if len(previewEvents) <= 0:
        #     #Log the nr of events found:
        #     logging.info(f"Preview - Found {len(previewEvents)} events in the chosen time frame.")
        # else:
            logging.error("Preview - No events found in the chosen time frame.")
            return
        
        #Load the events in self memory and filter on XY
        self.previewEvents = previewEvents
        self.previewEvents = self.filterEvents_xy(self.previewEvents,xyStretch)
        
        if eventType == "Positive only":
            self.previewEvents = self.filterEvents_npy_p(self.previewEvents,1)
        elif eventType == "Negative only":
            self.previewEvents = self.filterEvents_npy_p(self.previewEvents,0)
        
        logging.info(f"Preview - Displaying {len(self.previewEvents)} events ({eventType}) in the chosen time frame.")
        
        #Update the preview panel and localization list:
        self.updateShowPreview(previewEvents=self.previewEvents,timeStretch=timeStretch,frameTime=frameTime)
        self.updateLocList()
        
        self.updateProgressBar(overwriteValue = 100)
        self.updateGUIafterNewResults()
        #Switch the user to the Preview tab
        utils.changeTab(self, text='Preview run')

    def updateProgressBar(self,overwriteValue = None,findorfit='fit'):
        #self.signalEmitArray is an array, which looks like this:
        #[current_pol_run, total_number_pol_run, progress_of_this_pol_run]
        #This is separate for finding and fitting.
        #E.g. a finding can give [1,1,0.5], which would mean that it's 50% complete of the second pol run of two total pol runs
        if overwriteValue == None:
            if self.signalEmitArray is not None:
                if findorfit == 'find': #We're still in finding!, thus 0-50 range
                    self.analysis_progressbar.setValue(self.signalEmitArray[2]*((self.signalEmitArray[0]+1)/(self.signalEmitArray[1]+1))*100*0.5)
                else:#We're in fitting, thus 50-100 range
                    self.analysis_progressbar.setValue(self.signalEmitArray[2]*((self.signalEmitArray[0]+1)/(self.signalEmitArray[1]+1))*100*0.5+50)
                    
                QApplication.processEvents() #progress the changes to the gui
                
                self.signalEmitArray = None
        else:
            self.analysis_progressbar.setValue(overwriteValue)
            QApplication.processEvents() #progress the changes to the gui

    def findingAnalysisComplete(self):
        #Function is run as soon as finding is completed
        self.currentFileInfo['FindingTime'] = time.time() - self.analysisStartTime 
        self.storeFindingOutput()
    
    def fittingAnalysisComplete(self):
        #Function is run as soon as fitting is completed
        #Perfect time to save data and such :)
        self.updateProgressBar(overwriteValue = 95)
        self.currentFileInfo['FittingTime'] = time.time() - self.currentFileInfo['FindingTime']
        #Update the GUI results
        self.updateGUIafterNewResults()
        #Store data... These functions have checks on what to store
        # if self.globalSettings['StoreFinalOutput']['value'] > 0:
        self.storeLocalizationOutput()
        self.createAndStoreFileMetadata()
        
        #restore global settings (can be changed in run_processing_i)
        self.globalSettings = self.globalSettingsBeforeRun
        self.updateProgressBar(overwriteValue = 100)
        logging.info('Analysis completed!')

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

    def getStoreLocationPartial(self):
        if 'ExistingFinding' in self.data['FindingMethod']:
            FindingResultFileName = self.data['FindingMethod'][self.data['FindingMethod'].index('File_Location="')+len('File_Location="'):self.data['FindingMethod'].index('")')]
            storeLocationPartial = FindingResultFileName[:-7]
        else:
            #Find the last . and remove everything after it - i.e. removing the extension:
            storeLocationPartial = self.currentFileInfo['CurrentFileLoc'][:self.currentFileInfo['CurrentFileLoc'].rfind('.')]
        return storeLocationPartial

    def storeLocalization(self,storeLocation,localizations,outputType='thunderstorm'):
        if outputType == 'minimal':
            localizations.to_csv(storeLocation)
        elif outputType == 'thunderstorm':
            localizations = localizations.copy()
            #remove all entries that have a nan somewhere:
            localizations = localizations.dropna()
            
            #ensure that every entry that is currently empty will be populated by '1':
            localizations = localizations.applymap(lambda x: '1' if x == '' else x)
            
            #Add a frame column to fittingResult:
            localizations['frame'] = localizations['t'].apply(round).astype(int)
            localizations['frame'] -= min(localizations['frame'])-1
            #Create thunderstorm headers
            headers = list(localizations.columns)
            headers = ['\"x [nm]\"' if header == 'x' else '\"y [nm]\"' if header == 'y' else '\"z [nm]\"' if header == 'z' else '\"t [ms]\"' if header == 't' else 'Δx [nm]' if header == 'del_x' else 'Δy [nm]' if header == 'del_y' else 'Δt [ms]' if header == 'del_t' else 'x_dim [px]' if header == 'x_dim' else 'y_dim [px]' if header == 'y_dim' else 't_dim [ms]' if header == 't_dim' else header for header in headers]

            #Store the csv with ID as index:
            localizations.to_csv(storeLocation, header=headers, quoting=csv.QUOTE_NONE, index=False, index_label='\"id\"')
        else:
            #default to minimal
            localizations.to_csv(storeLocation)

    def storeLocalizationOutput(self):
        #Storing the .CSV
        if self.globalSettings['StoreFinalOutput']['value'] > 0 and len(self.data['FittingResult'][0]) > 0:
            logging.debug('Attempting to store fitting results output')
            storeLocation = self.getStoreLocationPartial()+'_FitResults_'+self.storeNameDateTime+'.csv'
            #Store the localization output
            localizations = self.data['FittingResult'][0].dropna(axis=0, ignore_index=True)
            localizations = localizations.drop('fit_info', axis=1)

            #Actually store
            self.storeLocalization(storeLocation,localizations,outputType=self.globalSettings['OutputDataFormat']['value'])

        #Also store pickle information:
        #Also save pos and neg seperately if so useful:
        if self.globalSettings['StoreFittingOutput']['value'] > 0 and len(self.data['FittingResult'][0]) > 0:
            if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
                try:
                    allPosFittingResults = self.data['FittingResult'][0][self.data['FittingResult'][0]['p']==1]
                    allNegFittingResults = self.data['FittingResult'][0][self.data['FittingResult'][0]['p']==0]
                    if len(allPosFittingResults) > 0 and len(allNegFittingResults) > 0:
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

    def storeFindingOutput(self):
        if self.globalSettings['StoreFindingOutput']['value'] > 0 and len(self.data['FindingResult'][0]) > 0:
            logging.debug('Attempting to store finding results output')
            #Store the Finding results output
            file_path = self.currentFileInfo['CurrentFileLoc'][:-4]+'_FindingResults_'+self.storeNameDateTime+'.pickle'
            self.data['refFindingFile'] = file_path
            with open(file_path, 'wb') as file:
                pickle.dump(self.data['FindingResult'][0], file)


            #Also save pos and neg seperately if so useful:
            if self.dataSelectionPolarityDropdown.currentText() == self.polarityDropdownNames[3]:
                #Determine how many positive/negative candidates are found:
                #loop over the found candidates and check where the 'p' of the first index is still 1
                nrPosLocs = max(np.where([value['events']['p'].iloc[0] for key, value in self.data['FindingResult'][0].items()])[0])
                try:
                    #Check that we have both positive and negative finding results
                    if nrPosLocs > 0 and len(self.data['FindingResult'][0]) > nrPosLocs:
                        allPosFindingResults = {key: value for key, value in self.data['FindingResult'][0].items() if key <= nrPosLocs}
                        allNegFindingResults = {key-nrPosLocs-1: value for key, value in self.data['FindingResult'][0].items() if key > nrPosLocs}

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

    def readableGlobalSettings(self):
        #Returns readable global settings for metadata and such
        output = ''
        for key, value in self.globalSettings.items():
            if key not in self.globalSettings['IgnoreInOptions']:
                output+=f"{key}: {self.globalSettings[key]['value']}\n"
        return output

    def createAndStoreFileMetadata(self):
        if self.globalSettings['StoreFileMetadata']['value'] > 0 :
            logging.debug('Attempting to create and store file metadata')
            try:
                metadatastring = (f"""\
                Metadata information for file {self.currentFileInfo['CurrentFileLoc']}
                Analysis routine started at {self.storeNameDateTime[0:4]}-{self.storeNameDateTime[4:6]}-{self.storeNameDateTime[6:8]} {self.storeNameDateTime[9:11]}:{self.storeNameDateTime[11:13]}:{self.storeNameDateTime[13:15]}
                Analysis routine finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                {self.data['MetaDataOutput']}
                
                ---- Global settings used ----
                
                {self.readableGlobalSettings()}
                
                ---- Additional custom output ----
                
                Custom output from finding function:\n""")\
                + f"""{self.data['FindingResult'][1]}\n\n""" + (f"""\

                Custom output from fitting function:\n""")\
                + f"""{self.data['FittingResult'][1]}
                """
                #Store this metadatastring:
                with open(self.getStoreLocationPartial()+'_RunInfo_'+self.storeNameDateTime+'.txt', 'w') as f:
                    metadatastringb = dedent(metadatastring)
                    metadatastringb = metadatastringb.replace('                ', '')
                    f.write(dedent(metadatastringb))
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
                    methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(self,f"distKwarg_name_to_displayName_map"),typev='distOrTime'))
                # add timeKwarg choice to Kwargs if given
                if ("ComboBox" in widget.objectName()) and widget.isVisibleTo(self.tab_processing) and 'time_kwarg' in widget.objectName():
                    methodKwargNames_method.append('time_kwarg')
                    methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(self,f"timeFit_name_to_displayName_map"),typev='distOrTime'))
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
                            methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(self,f"distKwarg_name_to_displayName_map"),typev='distOrTime'))
                        # add timeKwarg choice to Kwargs if given
                        if ("ComboBox" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.tab_processing) and 'time_kwarg' in widget_sub.objectName():
                            methodKwargNames_method.append('time_kwarg')
                            methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(self,f"timeFit_name_to_displayName_map"),typev='distOrTime'))

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
        logging.debug(widget)


    def save_entries_to_json_core(self):
        self.entries = {}
        allEditableFields = self.get_editable_fields().items()
        # Iterate over all editable fields and store their values in the entries dictionary
        for field_name, field_widget in allEditableFields:
            # only store editable fields that have a name
            if not field_name=="":
                if isinstance(field_widget, QLineEdit):
                    self.entries[field_name] = field_widget.text()
                if isinstance(field_widget, QCheckBox):
                    self.entries[field_name] = field_widget.isChecked()
                elif isinstance(field_widget, QComboBox):
                    self.entries[field_name] = field_widget.currentText()

    def saveas_entries_to_json(self):
        #Do save_entries_to_json, but to a user-definable location
        #Do the bulk of the work
        self.save_entries_to_json_core()
    
        #Get a location from the user
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select storage location", "", "JSON Files (*.json)", options=options)
        if file_path:
            with open(file_path, 'w') as json_file:
                json.dump(self.entries, json_file)
            #Also store the global settings:
            self.advancedSettingsWindow.save_global_settings(jsonLocation=file_path[:-5]+'_advancedSettings.json')
        else:
            pass

        
    def save_entries_to_json(self):
        #Do the bulk of the work
        self.save_entries_to_json_core()
        # Specify the path and filename for the JSON file
        json_file_path = self.globalSettings['JSONGUIstorePath']['value']
        # Write the entries dictionary to the JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(self.entries, json_file)
        
        #Also store the global settings:
        self.advancedSettingsWindow.save_global_settings()

    def loadas_entries_from_json(self):
        # Specify the path and filename for the JSON file
        #Get a location from the user
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select storage location", "", "JSON Files (*.json)", options=options)
        if file_path:
            #First set all comboboxes
            self.load_entries_from_json_single(json_file_path=file_path,runParams=['QComboBox'])
            #Then set all line edits
            self.load_entries_from_json_single(json_file_path=file_path,runParams=['QLineEdit'])
            #Then set all checkboxes
            self.load_entries_from_json_single(json_file_path=file_path,runParams=['QCheckBox'])
            #try to load global settings next to it:
            try:
                self.advancedSettingsWindow.load_global_settings(jsonLocation=file_path[:-5]+'_advancedSettings.json')
            except:
                logging.warning('Could not load global settings from '+file_path[:-5]+'_advancedSettings.json')
                
        else:
            pass

    def load_entries_from_json(self):
        # Specify the path and filename for the JSON file
        json_file_path = self.globalSettings['JSONGUIstorePath']['value']
        #First set all comboboxes
        self.load_entries_from_json_single(json_file_path=json_file_path,runParams=['QComboBox'])
        #Then set all line edits
        self.load_entries_from_json_single(json_file_path=json_file_path,runParams=['QLineEdit'])
        #Then set all checkboxes
        self.load_entries_from_json_single(json_file_path=json_file_path,runParams=['QCheckBox'])
        #Also load global settings
        self.advancedSettingsWindow.load_global_settings()
        
    def load_entries_from_json_single(self,json_file_path=None,runParams=['QLineEdit','QComboBox','QCheckBox']):
        if json_file_path is None:
            json_file_path = self.globalSettings['JSONGUIstorePath']['value']
        try:
            # Load the entries from the JSON file
            with open(json_file_path, "r") as json_file:
                self.entries = json.load(json_file)

            for polVal in ['Pos']:#["Pos","Neg","Mix"]:
                # Set the values of the editable fields from the loaded entries
                allEditableFields = self.get_editable_fields().items()
                for field_name, field_widget in allEditableFields:
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
                                        elif 'time_kwarg' in field_widget.objectName():
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
        all_widgets = self.findChildren((QLineEdit, QComboBox, QCheckBox))
        for widget in all_widgets:
            fields[widget.objectName()] = widget
        
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

""" Fitting/Finding logic """

class FindingFittingAnalysis():
    """
    General class for both finding and fitting analysis - expanded later for specific finding or fitting.
    """
    def __init__(self):
        # super().__init__()
        #Initiate empy variables in self:
        self.polarityAnalysis = 'Both' #'Pos','Neg','Mix', or 'Both'
        self.parrallellized = False
        self.GPU = False #Not yet implemented
        self.settings = None #Global settings of the GUI
        self.GUIinfo = None #all GUI info 
        self.EvalText = '' #Textual string of what function to run, including all params
        self.Results = {} #Where the fitting/finding results will be stored eventually
        self.disableRun = False #Set to True if you want to skip the run entirely.
        self.currentProgress = 0 #Progress of only this step from 0-1
        self.currentProgressStep = 0 #To do with progress
        self.totalProgressNrSteps = 1#To do with progress - current step, e.g. finding1-finding2 is 2 steps-  WHICH is 1 in python-counting
        
        global abortFlag
    
    #Some functions to set variables/values:
    def set_polarityAnalysis(self,polarityAnalysis):
        self.polarityAnalysis = polarityAnalysis
    
    def set_parrallellized(self,parrallellized):
        self.parrallellized = parrallellized
    
    def set_GPU(self,GPU):
        self.GPU = GPU
    
    def set_globalSettings(self,settings):
        #Settings should be the globalSettings structure from the GUI
        self.settings = settings
    
    def set_GUIinfo(self,GUIinfo):
        self.GUIinfo = GUIinfo
    
    def get_Results(self):
        return self.Results

    def set_EvalText(self,EvalText):
        self.EvalText = EvalText
    
    def getFunctionEvalText(self,GUIinfo,className,p1,p2,polarity):
        #Get the dropdown info
        moduleMethodEvalTexts = []
        all_layouts = GUIinfo.findChild(QWidget, "groupbox"+className+polarity).findChildren(QLayout)[0]


        methodKwargNames_method = []
        methodKwargValues_method = []
        methodName_method = ''
        # Iterate over the items in the layout
        for index in range(all_layouts.count()):
            item = all_layouts.itemAt(index)
            widget = item.widget()
            if widget is not None:#Catching layouts rather than widgets....
                if ("LineEdit" in widget.objectName()) and widget.isVisibleTo(GUIinfo.tab_processing):
                    # The objectName will be along the lines of foo#bar#str
                    #Check if the objectname is part of a method or part of a scoring
                    split_list = widget.objectName().split('#')
                    methodName_method = split_list[1]
                    methodKwargNames_method.append(split_list[2])

                    #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                    methodKwargValues_method.append(widget.text().replace('\\','/'))

                # add distKwarg choice to Kwargs if given
                if ("ComboBox" in widget.objectName()) and widget.isVisibleTo(GUIinfo.tab_processing) and 'dist_kwarg' in widget.objectName():
                    methodKwargNames_method.append('dist_kwarg')
                    methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(GUIinfo,f"distKwarg_name_to_displayName_map"),typev='distOrTime'))
                # add timeKwarg choice to Kwargs if given
                if ("ComboBox" in widget.objectName()) and widget.isVisibleTo(GUIinfo.tab_processing) and 'time_kwarg' in widget.objectName():
                    methodKwargNames_method.append('time_kwarg')
                    methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(GUIinfo,f"timeFit_name_to_displayName_map"),typev='distOrTime'))
            else:
                #If the item is a layout instead...
                if isinstance(item, QLayout):
                    for index2 in range(item.count()):
                        item_sub = item.itemAt(index2)
                        widget_sub = item_sub.widget()
                        if ("LineEdit" in widget_sub.objectName()) and widget_sub.isVisibleTo(GUIinfo.tab_processing):
                            # The objectName will be along the lines of foo#bar#str
                            #Check if the objectname is part of a method or part of a scoring
                            split_list = widget_sub.objectName().split('#')
                            methodName_method = split_list[1]
                            methodKwargNames_method.append(split_list[2])

                            #Widget.text() could contain a file location. Thus, we need to swap out all \ for /:
                            methodKwargValues_method.append(widget_sub.text().replace('\\','/'))

                        # add distKwarg choice to Kwargs if given
                        if ("ComboBox" in widget_sub.objectName()) and widget_sub.isVisibleTo(GUIinfo.tab_processing) and 'dist_kwarg' in widget_sub.objectName():
                            methodKwargNames_method.append('dist_kwarg')
                            methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(GUIinfo,f"distKwarg_name_to_displayName_map"),typev='distOrTime'))
                            # add timeKwarg choice to Kwargs if given
                        if ("ComboBox" in widget_sub.objectName()) and widget_sub.isVisibleTo(GUIinfo.tab_processing) and 'time_kwarg' in widget_sub.objectName():
                            methodKwargNames_method.append('time_kwarg')
                            methodKwargValues_method.append(utils.functionNameFromDisplayName(widget.currentText(),getattr(GUIinfo,f"timeFit_name_to_displayName_map"),typev='distOrTime'))

        #If at this point there is no methodName_method, it means that the method has exactly 0 req or opt kwargs. Thus, we simply find the value of the QComboBox which should be the methodName:
        if methodName_method == '':
            for index in range(all_layouts.count()):
                item = all_layouts.itemAt(index)
                widget = item.widget()
                if isinstance(widget,QComboBox) and widget.isVisibleTo(GUIinfo.tab_processing) and className in widget.objectName():
                    if className == 'Finding':
                        methodName_method = utils.functionNameFromDisplayName(widget.currentText(),getattr(GUIinfo,f"Finding_functionNameToDisplayNameMapping{polarity}"))
                        break
                    elif className == 'Fitting':
                        methodName_method = utils.functionNameFromDisplayName(widget.currentText(),getattr(GUIinfo,f"Fitting_functionNameToDisplayNameMapping{polarity}"))
                        break

        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = utils.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method,partialStringStart=str(p1)+','+str(p2))
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)

        if moduleMethodEvalTexts is not None and len(moduleMethodEvalTexts) > 0:
            return moduleMethodEvalTexts[0]
        else:
            return None

    def setProgressInfo(self,progressInfo):
        self.progressInfo = progressInfo
        self.GUIinfo.signalEmitArray = [self.currentProgressStep, self.totalProgressNrSteps, self.progressInfo]

class FindingAnalysis(FindingFittingAnalysis):
    """
    General class that runs the finding analysis. Should take in parameters, based on the file location (NOT the events), and spit out the finding results. Should handle all file loading/data handling, splitting of data, multiprocessing, etc etc.
    """
    def __init__(self):
        super().__init__()
        #Initiate empy variables in self:
        self.fileLocation = '' #location of file (.hdf5, .raw, .npy)
        self.timeStretchMs = [0,np.inf] #Time stretch in milliseconds
        self.xyStretch = [[0,np.inf],[0,np.inf]] #XY stretch in pixels
        self.chunkingTime = [np.inf,0]
        
    def set_xyStretch(self,xyStretch):
        self.xyStretch = xyStretch
    
    def set_fileLocation(self,fileLocation):
        #Check if ends on .hdf5, .raw, .npy:
        if not fileLocation.endswith('.hdf5') and not fileLocation.endswith('.raw') and not fileLocation.endswith('.npy'):
            logging.error('File location must end with .hdf5, .raw, or .npy')
        else:
            self.fileLocation = fileLocation
            if fileLocation.endswith('.npy'):
                if self.parrallellized == True:
                    self.parrallellized = False
                    logging.warning('Parrallellization for FINDING in NPY turned off!')
            if fileLocation.endswith('.raw'):
                if self.parrallellized == True:
                    self.parrallellized = False
                    logging.warning('Parrallellization for FINDING in RAW turned off!')
    
    def set_timeStretchMs(self,timeStretchMs):
        self.timeStretchMs = timeStretchMs
    
    def set_chunkingTime(self,chunkingTime):
        self.chunkingTime = chunkingTime
    #Real functions:
    """
    Callable functions
    """
    def analyse(self):
        logging.info(f"--- Starting finding analysis ---")
        #First, check on which polarities we should run:
        if self.polarityAnalysis == 'Pos':
            self.totalProgressNrSteps = 0
            self.currentProgressStep = 0
            self.runFindingOnPolarity('Pos')
        elif self.polarityAnalysis == 'Neg':
            self.totalProgressNrSteps = 0
            self.currentProgressStep = 0
            self.runFindingOnPolarity('Neg')
        elif self.polarityAnalysis == 'Mix':
            self.totalProgressNrSteps = 0
            self.currentProgressStep = 0
            self.runFindingOnPolarity('Mix')
        elif self.polarityAnalysis == 'Both':
            self.totalProgressNrSteps = 1
            self.currentProgressStep = 0
            self.runFindingOnPolarity('Pos')
            self.setProgressInfo(1)
            self.currentProgressStep = 1
            self.runFindingOnPolarity('Neg')
        else:
            logging.error('Polarity analysis must be either Pos, Neg, Mix, or Both')
        
        #Tell the GUI the finding is complete!
        self.setProgressInfo(1)
        self.GUIinfo.FindingCompleted = True
        pass
    
    def loadPickleFinding(self,evalText,singlePolarity):
        #Function run when loading a pickle file for a single polarity 
        self.events = None
        pickleLoad = eval(evalText)
        
        #Store as results:
        if self.Results == {}:
            self.Results = pickleLoad
        else: #if there is some data already:
            newResults = self.Results
            offsetlen = len(self.Results[0])
            self.Results={}
            self.Results[0]=newResults[0]
            #we should append it:
            for k in pickleLoad[0]:
                self.Results[0][k+offsetlen] = pickleLoad[0][k]
            self.Results[1] = newResults[1]+'\n\n\n'+pickleLoad[1]
        
        return
    
    def checkForRunDisableLoadPickleMismatch(self,singlePolarity):
        #If the FITTING is loading a PICKLE, but the FINDING is NOT loading a pickle, the FINDING is skipped.
        #If the FITTING is loading a PICKLE, but the FINDING is loading a pickle, continue as normal.
        #If the FITTING is NOT loading a PICKLE, continue as normal
        findingeval = self.getFunctionEvalText(self.GUIinfo,'Finding',"","",singlePolarity)
        fittingeval = self.getFunctionEvalText(self.GUIinfo,'Fitting',"","",singlePolarity)
        if not 'LoadExistingFitting' in fittingeval:
            self.disableRun = False
        else:
            if 'LoadExistingFinding' in findingeval:
                self.disableRun = False
            else:
                self.disableRun = True
    
    def runFindingOnPolarity(self,singlePolarity):
        logging.info(f"Starting finding of polarity {singlePolarity}")
        if abortFlag.value == True:
            logging.error("Aborting fitting analysis due to abortFlag")
            return
        starttime = time.time()
        #check for disabling the run
        self.checkForRunDisableLoadPickleMismatch(singlePolarity)
        
        #Run the analysis
        if self.disableRun == True:
            logging.warning('Finding not run because disableRun was set to True! (pickle load mis-match)')
            #Still need to enter something as finding result:
            if self.Results == {}:
                self.Results={}
                self.Results[0]={}
                self.Results[1]='Finding not run because of pickle load mis-match'
            else:
                newmetadata = self.Results[1]+'\n\n\nFinding not run because of pickle load mis-match'
                newResults = self.Results
                self.Results = {}
                self.Results[0] = newResults[0]
                self.Results[1] = newmetadata
            #Tell the GUI the finding is complete!
            self.GUIinfo.FindingCompleted = True
            return
        
        #Run the finding on a single polarity
        #First, get the evaluation text:
        evalText = self.getFunctionEvalText(self.GUIinfo,'Finding',"self.events","self.settings",singlePolarity)
        self.set_EvalText(evalText)
        
        #Set this to the GUI
        self.GUIinfo.data['FindingMethod'] = evalText
        
        #Run an entirely different function if a pickle file needs to be loaded: - we don't want to load any npy/raw/hdf5 data in this case.
        if "LoadExistingFinding" in evalText:
            self.loadPickleFinding(evalText,singlePolarity)
            return #end the runFindingOnPolarity function
        
        #Then, we initialize the finding, based on file type and parrallellization
        if self.GPU == False:
            if self.parrallellized == False:
                if self.fileLocation.endswith('.hdf5'):
                    
                    if self.chunkingTime[0] == np.inf:
                        #Function that returns all the start/stop indexes for all slices:
                        hdf5_startstopindeces = utils.findIndexFromTimeSliceHDF(self.fileLocation,requested_start_time_ms_arr = [self.timeStretchMs[0]],requested_end_time_ms_arr=[self.timeStretchMs[1]])
                    else:
                        #Function that returns all the start/stop indexes for all slices:
                        requested_start_time_ms_arr, requested_end_time_ms_arr = utils.determineAllStartStopTimesHDF(self.fileLocation,timeChunkMs=self.chunkingTime[0],timeChunkOverlapMs=self.chunkingTime[1],chunkStartStopTime = self.timeStretchMs)
                        
                        hdf5_startstopindeces = utils.findIndexFromTimeSliceHDF(self.fileLocation,requested_start_time_ms_arr = requested_start_time_ms_arr,requested_end_time_ms_arr=requested_end_time_ms_arr)
                    
                    #Now we have 1 or multiple chunks we have to loop over:
                    totalFindingResults = {}
                    totalFindingResults[0] = {}
                    totalFindingResults[1]=''
                    for chunk in range(len(hdf5_startstopindeces)):
                        self.setProgressInfo(chunk/len(hdf5_startstopindeces))
                        #Get the events from these indeces:
                        self.events = utils.timeSliceFromHDFFromIndeces(self.fileLocation,hdf5_startstopindeces,index=chunk)
                        
                        hotpixelarray = eval("["+self.settings['HotPixelIndexes']['value']+"]")
                        self.events = utils.removeHotPixelEvents(self.events,hotpixelarray)
                        
                        self.splitOnPolarity(singlePolarity)
                        self.splitOnXY()
                        
                        #Run the finding on this:
                        findingResults = self.runFinding(storeasselfResults=False)
                        
                        #If we have chunking - we should ensure we only keep the candidates that are in this 'real chunk' part of the chunk:
                        if len(hdf5_startstopindeces) > 0:
                            origfindingResults = findingResults
                            chunking_limits = [[requested_start_time_ms_arr[chunk]+self.chunkingTime[1],requested_end_time_ms_arr[chunk]-self.chunkingTime[1]],[requested_start_time_ms_arr[chunk],requested_end_time_ms_arr[chunk]]]
                            chunking_limits = np.multiply(chunking_limits,1000) #ms to us
                            
                            #Loop over all candidates:
                            for k in reversed(range(len(origfindingResults[0]))):
                                candidate = origfindingResults[0][k]
                                if self.filter_finding_on_chunking(candidate,chunking_limits) == False:
                                    #pop it
                                    origfindingResults[0].pop(k)
                            
                            #Reset the index values:
                            findingResults = {}
                            findingResults[0] = {index: value for index, value in enumerate(origfindingResults[0].values())}
                            findingResults[1] = origfindingResults[1]
                        
                        #And append
                        index_offset = len(totalFindingResults[0])
                        for k in findingResults[0]:
                            totalFindingResults[0][k + index_offset] = findingResults[0][k]
                        totalFindingResults[1] += '\n'+findingResults[1]
                    
                    #Store as results:
                    if self.Results == {}:
                        self.Results = totalFindingResults
                    else: #if there is some data already:
                        offsetlen = len(self.Results[0])
                        #we should append it:
                        for k in totalFindingResults[0]:
                            self.Results[0][k+offsetlen] = totalFindingResults[0][k]
                        self.Results[1] += '\n\n\n'+totalFindingResults[1]
                
                elif self.fileLocation.endswith('.raw'):
                    if self.chunkingTime[0] == np.inf or self.settings['FindingBatching']['value']==0:
                        #If no chunking, simply load the entire file in memory: Note that this also loads a .npy with the same name if available.
                        self.events = utils.RawToNpy(filepath=self.fileLocation,metaVisionPath=self.GUIinfo.globalSettings['MetaVisionPath']['value'],storeConvertedData = self.GUIinfo.globalSettings['StoreConvertedRawData']['value']>0)
                        
                        #Remove hot pixels
                        hotpixelarray = eval("["+self.settings['HotPixelIndexes']['value']+"]")
                        self.events = utils.removeHotPixelEvents(self.events,hotpixelarray)
                        
                        #get the correct time area:
                        #Keep only events between timeStretchMs[0] and [1]:
                        self.events = self.events[np.where((self.events['t']>self.timeStretchMs[0]*1000)&(self.events['t']<self.timeStretchMs[1]*1000))]
                        #Split on polarity, xy:
                        self.splitOnPolarity(singlePolarity)
                        self.splitOnXY()
                        
                        #Run the finding on this:
                        findingResults = self.runFinding()
                    else:
                        chunkFinished = False
                        curr_chunk = 0
                        findingResultsChunks = {}
                        while chunkFinished == False:
                            #determine min/max time based on chunk and chunking time, but don't go past timeStretchMs[1] or below zero:
                            minTime = max(0,curr_chunk*self.chunkingTime[0]+self.timeStretchMs[0]-self.chunkingTime[1])
                            maxTime = (curr_chunk+1)*self.chunkingTime[0]+self.timeStretchMs[0]+self.chunkingTime[1]
                            if maxTime > self.timeStretchMs[1]:
                                maxTime = self.timeStretchMs[1]
                                #Make this the last chunk to be run:
                                chunkFinished = True
                            logging.info(f"Chunking time: {minTime} - {maxTime}")
                            #Read in chunks:
                            self.events = utils.readRawTimeStretch(filepath=self.fileLocation,metaVisionPath=self.GUIinfo.globalSettings['MetaVisionPath']['value'], timeStretchMs=[minTime,maxTime+minTime])
                            
                            #Remove hot pixels
                            hotpixelarray = eval("["+self.settings['HotPixelIndexes']['value']+"]")
                            self.events = utils.removeHotPixelEvents(self.events,hotpixelarray)
                            
                            if len(self.events) == 0:
                                logging.info('Final chunk reached!')
                                chunkFinished = True
                                break
                            
                            #Split on polarity, xy:
                            self.splitOnPolarity(singlePolarity)
                            self.splitOnXY()
                            
                            #Run the finding on this:
                            findingResultsChunks[curr_chunk] = self.runFinding(storeasselfResults=False)
                            curr_chunk+=1
                        
                        #Combine all finding results...
                        #First we only keep finding results in their 'correct chunk' (i.e. allowing for overlap)
                        totalFindingResults = {}
                        totalFindingResults[0] = {}
                        totalFindingResults[1] = ''
                        for chunk in range(len(findingResultsChunks)):
                            chunkmintime = max(0,chunk*self.chunkingTime[0]+self.timeStretchMs[0])
                            chunkmaxtime = (chunk+1)*self.chunkingTime[0]+self.timeStretchMs[0]
                            chunking_limits = [[chunkmintime,chunkmaxtime],[chunkmintime-self.chunkingTime[1],chunkmaxtime+self.chunkingTime[1]]]
                            chunking_limits = np.multiply(chunking_limits,1000) #ms to us
                            #Integer and non-negative:
                            chunking_limits = [[int(max(0,x)) for x in sub_array] for sub_array in chunking_limits]
                            
                            #Get the finding results
                            origfindingResults = findingResultsChunks[chunk][0]
                            #Reset the index:
                            origfindingResults= {index: value for index, value in enumerate(origfindingResults.values())}
                        
                            #Loop over all candidates:
                            for k in reversed(range(len(origfindingResults))):
                                candidate = origfindingResults[k]
                                if self.filter_finding_on_chunking(candidate,chunking_limits) == False:
                                    #pop it
                                    origfindingResults.pop(k)
                                    
                            #Reset the index again:
                            origfindingResults= {index: value for index, value in enumerate(origfindingResults.values())}
                            
                            #And append
                            index_offset = len(origfindingResults)
                            for k in origfindingResults:
                                totalFindingResults[0][k + index_offset] = origfindingResults[k]
                            totalFindingResults[1] += '\n'+findingResultsChunks[chunk][1]
                        
                        #Reset the index:
                        totalFindingResults[0]= {index: value for index, value in enumerate(totalFindingResults[0].values())}
                        
                        #Set or append to self.results (i.e. neg after pos)
                        if self.Results == {}:
                            self.Results = totalFindingResults
                        else:
                            newResults = self.Results
                            offsetlen = len(newResults[0])
                            #we should append it:
                            for k in totalFindingResults[0]:
                                newResults[0][k+offsetlen] = totalFindingResults[0][k]
                            newResults[1] += '\n\n\n'+totalFindingResults[1]
                            self.Results={}
                            self.Results = newResults
                    pass 
                
                elif self.fileLocation.endswith('.npy'):
                    #Load the data
                    self.events = np.load(self.fileLocation)
                    #Remove hot pixels
                    hotpixelarray = eval("["+self.settings['HotPixelIndexes']['value']+"]")
                    self.events = utils.removeHotPixelEvents(self.events,hotpixelarray)
                    
                    #get the correct time area:
                    #Keep only events between timeStretchMs[0] and [1]:
                    self.events = self.events[np.where((self.events['t']>self.timeStretchMs[0]*1000)&(self.events['t']<self.timeStretchMs[1]*1000))]
                    #Split on polarity, xy:
                    self.splitOnPolarity(singlePolarity)
                    self.splitOnXY()
                    
                    #Run the finding on this:
                    findingResults = self.runFinding()
                    
                    pass 
                else:
                    logging.error('File location must end with .hdf5, .raw, or .npy')
                    pass
                
            elif self.parrallellized == True: #Parrallel-CPU processing
                if self.fileLocation.endswith('.hdf5'):
                    
                    if self.settings['FindingBatching']['value']>0:
                        #Find the start/end time arrays based on the slicing
                        requested_start_time_ms_arr, requested_end_time_ms_arr = utils.determineAllStartStopTimesHDF(self.fileLocation,timeChunkMs = self.chunkingTime[0],timeChunkOverlapMs = self.chunkingTime[1],chunkStartStopTime = self.timeStretchMs)
                    else:#if no batching, just use the whole file
                        #Find the start/end time arrays based on the slicing
                        requested_start_time_ms_arr, requested_end_time_ms_arr = utils.determineAllStartStopTimesHDF(self.fileLocation,timeChunkMs = np.inf,timeChunkOverlapMs = 0,chunkStartStopTime = self.timeStretchMs)
                    
                    #Function that returns all the start/stop indexes for all slices:
                    hdf5_startstopindeces = utils.findIndexFromTimeSliceHDF(self.fileLocation,requested_start_time_ms_arr = requested_start_time_ms_arr,requested_end_time_ms_arr=requested_end_time_ms_arr)
                    
                    #myGUI and pyqtsignal cannot be passed, so has to be removed:
                    GUIinfo = self.GUIinfo
                    self.GUIinfo = {}
                    SelfResults = self.Results
                    self.Results = {}
                    
                    results = None
                    #Run the analysis in parallel over all cpu cores
                    with parallel_backend('loky', n_jobs=-1):
                        results = Parallel(max_nbytes=None,timeout=1e6)(delayed(self.process_hdf5_chunk_joblib)(n, hdf5_startstopindeces,singlePolarity,abortFlag) for n in range(0,len(hdf5_startstopindeces)))
                    
                    #reset GUIinfo in case we need it:
                    self.GUIinfo = GUIinfo
                    self.Results = SelfResults
                    #If we have chunking - we should ensure we only keep the candidates that are in this 'real chunk' part of the chunk:
                    if len(hdf5_startstopindeces) > 0:
                        
                        #We loop over all results:
                        for chunk in range(0,len(results)):
                            result = results[chunk]
                            if result == None:
                                logging.error('No data found for this chunk')
                                break
                            origfindingResults = result
                            
                            chunking_limits = [[requested_start_time_ms_arr[chunk]+self.chunkingTime[1],requested_end_time_ms_arr[chunk]-self.chunkingTime[1]],[requested_start_time_ms_arr[chunk],requested_end_time_ms_arr[chunk]]]
                            chunking_limits = np.multiply(chunking_limits,1000) #ms to us
                        
                            #Loop over all candidates:
                            origfindingResultsKeyCopy = list(origfindingResults[0].keys())
                            for k in reversed(origfindingResultsKeyCopy):#reversed(range(len(origfindingResults[0]))):
                                candidate = origfindingResults[0][k]
                                if self.filter_finding_on_chunking(candidate,chunking_limits) == False:
                                    #pop it
                                    origfindingResults[0].pop(k)
                            
                            #Reset the index values:
                            result = {}
                            result[0] = {index: value for index, value in enumerate(origfindingResults[0].values())}
                            result[1] = origfindingResults[1]
                            
                            #Store back in original results
                            results[chunk] = result
                            
                    parr_batch_results = None
                    parr_batch_metadata = None
                    #Get all results of all parrallell runs:
                    parr_batch_results = [result[0] for result in results if result is not None]
                    parr_batch_metadata = [result[1] for result in results if result is not None]
                    
                    if parr_batch_results is not None and len(parr_batch_results) > 0:
                        #Create the combined dictionary and metadata of all results:
                        combined_dict = {}
                        combined_metadata = ''
                        
                        #Ugliest way to add all to a single dictionary, but its fast and it works
                        for n in range(0,len(results)):
                            newent = parr_batch_results[n]
                            for k in newent:
                                combined_dict[len(combined_dict)] = newent[k]
                            combined_metadata += '\n'+parr_batch_metadata[n]

                        findingResults = {}
                        findingResults[0] = combined_dict
                        findingResults[1] = combined_metadata
                        #Finally, store it as the Results data, or append it if Both polarity
                        if self.Results == {}: #If it's the first data, just store it
                            self.Results = findingResults
                        else: #Otherwise append to previous data (i.e. first pos, then neg)
                            oldResults = self.Results
                            self.Results = {}
                            combined_dict_index_offset = len(oldResults[0])
                            self.Results[0] = oldResults[0]
                            for k in combined_dict:
                                self.Results[0][k + combined_dict_index_offset] = combined_dict[k]
                            self.Results[1] = oldResults[1]+'\n\n'+combined_metadata
                        
                        pass
                    else:
                        logging.error('No data found for any chunks')
                        pass
                    
                elif self.fileLocation.endswith('.raw'):
                    #To be implemented
                    logging.error('NOT YET IMPLEMENTED YOU SHOULDNT REACH THIS -RAW')
                    pass 
                
                elif self.fileLocation.endswith('.npy'):
                    #To be implemented
                    logging.error('NOT YET IMPLEMENTED YOU SHOULDNT REACH THIS -RAW')
                    pass 
                else:
                    logging.error('File location must end with .hdf5, .raw, or .npy')
                    pass
                
        else: #GPU running
            #To be implemented
            pass
        
        totaltime = time.time()-starttime
        #Logging feedback
        if 'findingResults' in locals():
            logging.info(f"Finding of polarity {singlePolarity} complete. {len(findingResults[0])} candidates found!")
            #Add metadata info
            self.GUIinfo.data['MetaDataOutput'] += '\n-----Information on finding of polarity '+singlePolarity+': -----\nMethodology Used:\n' +evalText+'\n\nNumber of candidates found: '+str(len(findingResults[0]))+'\nCandidate fitting took '+str(totaltime)+' seconds.\n\n'
        else:
            if len(self.Results)>0:
                logging.info(f"Finding of polarity {singlePolarity} complete. {len(self.Results[0])} candidates found!")
                #Add metadata info
                self.GUIinfo.data['MetaDataOutput'] += '\n-----Information on finding of polarity '+singlePolarity+': -----\nMethodology Used:\n' +evalText+'\n\nNumber of candidates found: '+str(len(self.Results[0]))+'\nCandidate fitting took '+str(totaltime)+' seconds.\n\n'
        pass
    

    def splitOnPolarity(self,singlePolarity):
        #Split the self.events on p data:
        if singlePolarity == 'Pos':
            self.events = utils.filterEvents_p(self.events,pValue=1)
        elif singlePolarity == 'Neg':
            self.events = utils.filterEvents_p(self.events,pValue=0)

    def splitOnXY(self):
        #Split the events on xy:
        self.events = utils.filterEvents_xy(self.events,xyStretch = (self.xyStretch[0][0], self.xyStretch[0][1], self.xyStretch[1][0], self.xyStretch[1][1]))

    """
    Functions to chunk the data (in time):
    """    
    def filter_finding_on_chunking(self,candidate,chunking_limits):
        #Return true if it should be in this chunk, false if not

        #Looking at end of chunk:

        #When using finding with bbox padding, sometimes empty candidates are passed to this method
        if len(candidate['events']['t']) == 0:
            logging.warning('this candidate is empty, returning false')
            return False

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

    """
    Finding itself
    """
    def runFinding(self,storeasselfResults=True):
        #Check if we have events left:
        if len(self.events) == 0 :
            logging.error("No events found! Check time, xy, polarity filtering!")
        else:
            if self.EvalText is not None:
                FindingResult = eval(str(self.EvalText))
                
                #Only keep the results that should be in this chunk:
                # filter_finding_on_chunking(candidate,chunking_limits)
                if storeasselfResults:
                    if self.Results == {}:
                        self.Results = FindingResult
                    else: #if there already is some finding result (i.e. first pos, then next), it should be appended 
                        newResults = {}
                        newResults[0] = self.Results[0].copy()
                        newResults[1] = self.Results[1]
                        combined_dict_index_offset = len(newResults[0])
                        for k in FindingResult[0]:
                            newResults[0][k + combined_dict_index_offset] = FindingResult[0][k]
                        newResults[1] += '\n\n'+FindingResult[1]
                        self.Results={}
                        self.Results[0] = newResults[0]
                        self.Results[1] = newResults[1]
                        pass
                
                return FindingResult
            else:
                logging.error("FindingEvalText is None")
                pass

    def process_hdf5_chunk_joblib(self,n,hdf5startstopindeces,singlePolarity,abortFlag):
        #Allowing for multicore hdf5 chunking, or single-core
        # print('Running finding chunk '+str(n)+' of '+str(len(hdf5startstopindeces)) )
        if abortFlag.value == True:
            logging.error("Aborting fitting analysis due to abortFlag")
            return
        st_time = time.time()
        #Get the events from these indeces (example at 0):
        self.events = utils.timeSliceFromHDFFromIndeces(self.fileLocation,hdf5startstopindeces,index=n)
        # print(f"Worker{n} events loaded time {time.time()}")
        #Remove hot pixels
        hotpixelarray = eval("["+self.settings['HotPixelIndexes']['value']+"]")
        self.events = utils.removeHotPixelEvents(self.events,hotpixelarray)
        # print(f"Worker{n} hot pixels filtered time {time.time()}")
        
        msk=(self.events['p']==0)
        fractionPosEvents = (np.sum(msk==False)*100.0/len(msk))
        # print(f"fraction of positive events: {fractionPosEvents:.2f} at time {self.events['t'][0]/1e6}us")
        # print(f"Worker{n} event filter started time {time.time()}")
        #Split the events on p data:
        if singlePolarity == 'Pos':
            self.events = utils.filterEvents_p(self.events,pValue=1)
        elif singlePolarity == 'Neg':
            self.events = utils.filterEvents_p(self.events,pValue=0)
        # print(f"Worker{n} event filter done time {time.time()}")
        #Split the events on xy:
        self.events = utils.filterEvents_xy(self.events,xyStretch = (self.xyStretch[0][0], self.xyStretch[0][1], self.xyStretch[1][0], self.xyStretch[1][1]))
        # print(f"Worker{n} xy filter done time {time.time()}")
        
        #Run the finding on this:
        FindingResult = self.runFinding(storeasselfResults=False)
        
        #Return them
        return FindingResult

class FittingAnalysis(FindingFittingAnalysis):
    """
    General class that runs the fitting analysis. Should take in parameters, based on the finding results, and spit out the fitting results. 
    """
    def __init__(self):
        super().__init__()
        self.findingResult = None
    
    def set_findingResult(self,findingResult):
        self.findingResult = findingResult
        
    def analyse(self):
        logging.info(f"--- Starting fitting analysis ---")
        #First, check on which polarities we should run:
        if self.polarityAnalysis == 'Pos':
            self.totalProgressNrSteps = 0
            self.currentProgressStep = 0
            self.runFittingOnPolarity('Pos')
        elif self.polarityAnalysis == 'Neg':
            self.totalProgressNrSteps = 0
            self.currentProgressStep = 0
            self.runFittingOnPolarity('Neg')
        elif self.polarityAnalysis == 'Mix':
            self.totalProgressNrSteps = 0
            self.currentProgressStep = 0
            self.runFittingOnPolarity('Mix')
        elif self.polarityAnalysis == 'Both':
            self.totalProgressNrSteps = 1
            self.currentProgressStep = 0
            self.runFittingOnPolarity('Pos')
            logging.info('Pos done')
            self.setProgressInfo(1)
            self.currentProgressStep = 1
            self.runFittingOnPolarity('Neg')
            logging.info('Neg done')
        else:
            logging.error('Polarity analysis must be either Pos, Neg, Mix, or Both')
        
        #Tell the GUI the fitting is complete!
        self.setProgressInfo(1)
        self.GUIinfo.FittingCompleted = True
        pass
        
    def runFittingOnPolarity(self,singlePolarity):
        logging.info(f"Starting fitting of polarity {singlePolarity}")
        if abortFlag.value == True:
            logging.error("Aborting fitting analysis due to abortFlag")
            return
        starttime = time.time()
        #Run the finding on a single polarity
        
        if len(self.findingResult) > 0 and len(self.findingResult[0]) > 0:
            #Remove all finding results that are not this polarity:
            if singlePolarity != 'Mix':
                self.partialFindingResults = self.findingResult[0].copy()
                #Loop over all entries reversed (for popping)
                for k in reversed(range(len(self.partialFindingResults))):
                    polval = np.unique(self.partialFindingResults[k]['events']['p'])
                    if len(polval) > 1:
                        #pop it!
                        self.partialFindingResults.pop(k)
                    else:
                        if singlePolarity == 'Pos':
                            if polval[0] == 0: #pop it if it's negative
                                self.partialFindingResults.pop(k)
                        elif singlePolarity == 'Neg':
                            if polval[0] == 1: #pop it if it's positive
                                self.partialFindingResults.pop(k)
            else:
                self.partialFindingResults = self.findingResult[0].copy()
            
            #fitting require indexes to start at 0, so correcting for this:
            try:
                first_key = next(iter(self.partialFindingResults))
            except:
                first_key=0
            self.fittingAdjustValue = first_key
            self.partialFindingResults = {key - first_key: value for key, value in self.partialFindingResults.items()}
        else:
            logging.warning('No finding results found!')
            self.partialFindingResults = None
        
        #First, get the evaluation text:
        self.set_EvalText(self.getFunctionEvalText(self.GUIinfo,'Fitting',"self.partialFindingResults","self.settings",singlePolarity))
        #Set it in the GUI
        self.GUIinfo.data['FittingMethod'] = str(self.getFunctionEvalText(self.GUIinfo,'Fitting',"self.partialFindingResults","self.settings",singlePolarity))
        
        #Set a 0-offset for the fitting if a file is loaded, otherwise it appends and creates a gap.
        if "LoadExistingFitting" in self.EvalText:
            self.fittingAdjustValue = 0
        
        #Run the fitting
        if self.partialFindingResults is not None:
            fittingResults = self.runFitting()
            logging.info(f"Fitting of polarity {singlePolarity} complete. {len(fittingResults[0].dropna())} valid localizations found!")
            nLoc = len(fittingResults[0].dropna())
        else:
            fittingResults = pd.DataFrame()
            self.Results = {}
            self.Results[0] = fittingResults
            self.Results[1] = fittingResults
            logging.info(f"Fitting of polarity {singlePolarity} complete. No localizations found!")
            nLoc = 0
            pass

        totaltime = time.time() - starttime
        self.GUIinfo.data['MetaDataOutput'] += '\n-----Information on fitting of polarity '+singlePolarity+': -----\nMethodology Used:\n' +self.GUIinfo.data['FittingMethod']+'\n\nNumber of valid localizations found: '+str(nLoc)+'\nCandidate fitting took '+str(totaltime)+' seconds.\n\n'
        
    """
    Fitting itself
    """
    def runFitting(self):
        if self.EvalText is not None:
            FittingResult = eval(str(self.EvalText))
            resbackup = self.Results
            if self.Results == {}:
                self.Results = FittingResult 
            else: #if there already is some fitting result, it should be appended
                if not hasattr(self,'fittingAdjustValue'):
                    self.fittingAdjustValue = len(self.Results[0])
                #Add the candidate_id offset correctly:
                if not FittingResult[0].empty:
                    FittingResult[0]['candidate_id'] = FittingResult[0]['candidate_id'] + self.fittingAdjustValue
                    origResults = self.Results
                    #append to self.results:
                    #Need to reset self.Result to avoid errors
                    self.Results = {}
                    self.Results[0] = pd.concat([origResults[0],FittingResult[0]],ignore_index=True)
                    self.Results[1] = origResults[1]+'\n\n\n'+FittingResult[1]
                else: #if empty fitting results:
                    self.Results = (resbackup[0], resbackup[1] + '\n\n\n' + FittingResult[1])

                pass
            
            return FittingResult
        else:
            logging.error("FittingEvalText is None")
            pass

""" Other GUI stuff"""

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
        load_button = QPushButton("Load stored Global Settings")
        load_button.clicked.connect(self.load_global_settings)
        layout.addWidget(load_button,currentRow+1,0,1,2)
        load_as_button = QPushButton("Load-as Global Settings")
        load_as_button.clicked.connect(self.load_as_global_settings)
        layout.addWidget(load_as_button,currentRow+2,0,1,2)

        #Add a full-reset button:
        full_reset_button = QPushButton("Fully reset GUI and global settings")
        full_reset_button.clicked.connect(self.confirm_full_reset_GUI_GlobSettings)
        layout.addWidget(full_reset_button,currentRow+3,0,1,2)

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
                try:
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
                except:
                    logging.debug(f"Error with setting {setting}")

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

    def save_global_settings(self,jsonLocation=None):
        if jsonLocation == None or jsonLocation == False:
            # Specify the path and filename for the JSON file
            json_file_path = self.parent.globalSettings['GlobalOptionsStorePath']['value']
        else:
            json_file_path = jsonLocation
        #Set the current GUI position/size to the global settings:
        self.parent.globalSettings['GUIWindowWize']['value'] = [self.parent.pos().x(),self.parent.pos().y(),self.parent.width(),self.parent.height()]
        

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

        #Also update the global settings (required if called from outside the window, i.e. hotpixel removal utility)
        self.load_global_settings()
        
        #Close after saving
        logging.info('Global settings saved!')
        self.close()

    def load_global_settings(self,jsonLocation=None):
        if jsonLocation == None or jsonLocation == False:
            # Specify the path and filename for the JSON file
            json_file_path = self.parent.globalSettings['GlobalOptionsStorePath']['value']
        else:
            json_file_path = jsonLocation
        try:    # Load the globalSettings dictionary from the JSON file
            with open(json_file_path, "r") as json_file:
                loaded_settings = json.load(json_file)

            to_rem = []
            # Update the 'input' value in self.parent.globalSettings, if it exists
            for key, value in loaded_settings.items():
                if key != 'IgnoreInOptions':
                    if key != "JSONGUIstorePath" and key != "GlobalOptionsStorePath" and key != "MetaVisionPath" and key != "LoggingFilePath":
                        if key in self.parent.globalSettings:
                            # self.parent.globalSettings[key]['input'] = value['input']
                            value['input'] = self.parent.globalSettings[key].get('input', None)
                    else:
                        #pop from loaded settings:
                        #concat key to to_rem:
                        to_rem.append(key)
            
            for key in to_rem:
                #Remove this key from loaded_settings:
                loaded_settings.pop(key)

            self.parent.globalSettings.update(loaded_settings)
            #Update the values of the globalSettings GUI
            self.updateGlobSettingsGUIValues()
        except:
            self.save_global_settings()
            logging.info('No global settings storage found, new one created.')
            
            
    def load_as_global_settings(self):
        ## Have a file load dialog to open json:
        # Open the file dialog
        jsonLocation, _ = QFileDialog.getOpenFileName(self, "Load Advanced settings file", "", "Settings File (*.json);;All Files (*);")
        json_file_path = jsonLocation
        try:    # Load the globalSettings dictionary from the JSON file
            with open(json_file_path, "r") as json_file:
                loaded_settings = json.load(json_file)

            to_rem = []
            # Update the 'input' value in self.parent.globalSettings, if it exists
            for key, value in loaded_settings.items():
                if key != 'IgnoreInOptions':
                    if key != "JSONGUIstorePath" and key != "GlobalOptionsStorePath" and key != "MetaVisionPath" and key != "LoggingFilePath":
                        if key in self.parent.globalSettings:
                            # self.parent.globalSettings[key]['input'] = value['input']
                            loaded_settings[key]['input'] = self.parent.globalSettings[key].get('input', None)
                    else:
                        #pop from loaded settings:
                        #concat key to to_rem:
                        to_rem.append(key)
            
            for key in to_rem:
                #Remove this key from loaded_settings:
                loaded_settings.pop(key)

            self.parent.globalSettings.update(loaded_settings)
            #Update the values of the globalSettings GUI
            self.updateGlobSettingsGUIValues()
        except:
            logging.error(f"Problem with loading adv settings from file {json_file_path}")

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

class ClickableGroupBox(QGroupBox):
    #Pressing clickable group boxes collapses them
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
        self.mainlayout = QGridLayout()
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
        self.mainlayout.layout().addWidget(self.VisualisationGroupbox,1,1,1,2)

        #------------End of GUI dynamic layout -----------------


        #Create a new 'this is the events under the cursor' textbox
        self.underCursorInfo = QLabel("")
        self.mainlayout.addWidget(self.underCursorInfo,2,2,1,1)
        
        # Create a napari viewer
        self.napariviewer = Viewer(show=False)
        #Add a napariViewer to the layout
        self.viewer = QtViewer(self.napariviewer)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        qvbox = QVBoxLayout()
        qvbox.addWidget(self.viewer.controls)
        
        #Create a button to reset zoom/position:
        button = QPushButton("Reset View", self)
        button.setObjectName("VisualisationResetViewButtonKEEP")
        button.clicked.connect(lambda text, parent=parent: self.reset_visualisation_view(parent, partialOrFull='Partial'))
        qvbox.addWidget(button)
        #Create a button to reset zoom/position:
        button = QPushButton("Reset entire visualisation", self)
        button.setObjectName("VisualisationResetViewFullButtonKEEP")
        button.clicked.connect(lambda text, parent=parent: self.reset_visualisation_view(parent, partialOrFull='Full'))
        qvbox.addWidget(button)
        #add a vertical stretch:
        qvbox.addStretch()
        
        self.mainlayout.addLayout(qvbox,3,1,1,1)
        self.viewer.controls.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.mainlayout.addWidget(self.viewer,3,2,1,1)
        
        self.timeOfLastCursorUpdate = 0
        self.underCursorInfoDF = {}
        self.underCursorInfoDF['current_pixel'] = {}
        self.viewer.on_mouse_move = lambda event: self.currently_under_cursor(event)

        logging.info('VisualisationNapari init')

    def currently_under_cursor(self,event):
        """
        Class that determines which pixel is currently under the cursor. The main task of this function is to go from cavnas position to image position.

        Args:
            event (Event): general Vispy event, containing, amongst others, the xy position of the curosr in the canvas.
        """
        #Vispy mouse position

        timeBetweenCursorUpdates = 0.0 #in seconds
        if self.timeOfLastCursorUpdate == 0 or (time.time() - self.timeOfLastCursorUpdate) >= timeBetweenCursorUpdates:
            #We get the canvas size/position in image pixel units
            canvas_size_in_px_units = event.source.size/self.viewer.camera.zoom
            # camera_coords = [self.napariviewer.camera.center[2]+.5, self.napariviewer.camera.center[1]+.5]
            camera_coords = [self.viewer.camera.center[2], self.viewer.camera.center[1]]
            canvas_pos = np.vstack([camera_coords-(canvas_size_in_px_units/2),camera_coords+(canvas_size_in_px_units/2)])
            #And we can normalize the cursor position to image pixels
            cursor_unit_norm = event._pos/event.source.size
            #Thus, we can find the position in um - at the moment we simply calculate this
            pos_nm=np.zeros((2,))
            pos_nm[0] = (cursor_unit_norm[0]*canvas_size_in_px_units[0]+canvas_pos[0][0])*1000
            
            pos_nm[1] = (cursor_unit_norm[1]*canvas_size_in_px_units[1]+canvas_pos[0][1])*1000

            #Yep we're doing this.
            GUIobject = self.parent().parent().parent().parent().parent().parent().parent().parent()
            pxsize = float(GUIobject.globalSettings['PixelSize_nm']['value'])
            
            pos_EBSpx = np.array(np.round(pos_nm/pxsize)).astype(int)
            
            #Set the text
            self.underCursorInfo.setText(f"Position: {int(pos_nm[0])},{int(pos_nm[1])} nm, EBS px: {pos_EBSpx[0]}, {pos_EBSpx[1]}")
            self.timeOfLastCursorUpdate = time.time()

    def reset_visualisation_view(self,parent,partialOrFull='Partial'):
        logging.info('Resetting visualisation view')
        #Reset the view of the napari viewer
        if partialOrFull == 'Partial':
            self.napariviewer.reset_view()
        elif partialOrFull == 'Full':
            self.napariviewer.reset_view()
            for layer in self.napariviewer.layers:
                layer.rendering = 'attenuated_mip'
                layer.visible = True
                layer.reset_contrast_limits()
                layer.colormap = 'gray'
                layer.gamma = 1
                layer.opacity = 1
        else:
            logging.error('Invalid partialOrFull argument passed to reset_visualisation_view')
            return

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
        logging.debug(FunctionEvalText)

        #Store the history of postprocessing
        current_postprocessinghistoryid = len(self.postProcessingHistory)
        self.postProcessingHistory[current_postprocessinghistoryid] = {}
        self.postProcessingHistory[current_postprocessinghistoryid][0] = self.parent.data['FittingResult'][0]
        self.postProcessingHistory[current_postprocessinghistoryid][1] = FunctionEvalText
        self.postProcessingHistory[current_postprocessinghistoryid][2] = [len(self.postProcessingHistory[current_postprocessinghistoryid][0]),-1]

        postProcessingResult = eval(FunctionEvalText)
        
        if postProcessingResult is not None:
            if len(self.parent.data['FittingResult']) > 1:
                metadata = self.parent.data['FittingResult'][1] + postProcessingResult[1]
            else:
                metadata = postProcessingResult[1]
            # self.parent.data['FittingResult'][0] = postProcessingResult[0]
            self.parent.data['FittingResult'] = (postProcessingResult[0],) + (metadata,)
            
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
            self.parent.parent.data['FittingResult'] = self.parent.postProcessingHistory[historyId]
            # self.parent.parent.data['FittingResult'][0] = self.parent.postProcessingHistory[historyId][0]
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
                        elif ("ComboBox" in widget_sub.objectName()) and widget_sub.isVisibleTo(self.PostProcessingGroupbox):
                            # The objectName will be along the lines of foo#bar#str
                            #Check if the objectname is part of a method or part of a scoring
                            split_list = widget_sub.objectName().split('#')
                            methodName_method = split_list[1]
                            methodKwargNames_method.append(split_list[2])

                            #We get the current choice of the dropdown.
                            methodKwargValues_method.append(widget_sub.currentText().replace('\\','/'))
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
        self.mainlayout = QGridLayout()
        self.mainRightlayout = QVBoxLayout()
        # Set the layout for the main widget
        self.setLayout(self.mainlayout)

        self.viewer = QtViewer(self.napariviewer)

        #Create a new 'this is the events under the cursor' textbox
        self.underCursorInfo = QLabel("")

        self.viewer.on_mouse_move = lambda event: self.currently_under_cursor(event)
        self.viewer.on_mouse_double_click = lambda event: self.napari_doubleClicked(event)

        #Add widgets to the main layout
        self.mainRightlayout.addWidget(self.underCursorInfo) #Text box with info under cursor
        #Make it expand:
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainRightlayout.addWidget(self.viewer) #The view itself
        
        controlVstack = QVBoxLayout()
        controlVstack.addWidget(self.viewer.controls)#The controls for the viewer (contrast, etc)
        self.viewer.controls.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        buttonResetView = QPushButton("Reset View")
        buttonResetView.clicked.connect(self.reset_view)
        controlVstack.addWidget(buttonResetView)
        buttonFullResetView = QPushButton("Reset entire visualisation")
        controlVstack.addWidget(buttonFullResetView)
        buttonFullResetView.clicked.connect(self.reset_full_view)
        #Add a spacer so everything is pushed to top:
        controlVstack.addStretch(1)
        self.mainlayout.addLayout(controlVstack,1,1) 
        self.mainlayout.addLayout(self.mainRightlayout,1,2)

        #Possible layouts to give more napari information
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
        self.settings = self.parent.globalSettings
        self.timeStretch=[]
        self.events = {}
        self.frametime_ms = 0
        self.maxFrames = 0
        
        self.timeOfLastCursorUpdate = 0

        #An array holding info about what's under the cursor
        underCursorInfo = {
            'current_pixel': [[-np.inf,-np.inf]],
            'current_pixel_EBSCoord': [[-np.inf,-np.inf]],
            'current_time': [[-1,-1]],
            'current_candidate': [[-1]]
        }

        self.underCursorInfoDF = pd.DataFrame(underCursorInfo)

    def reset_view(self):
        """
        Reset the view (i.e. reset zoom/pan/etc)
        """
        self.napariviewer.reset_view()
    
    def reset_full_view(self):
        """
        Fully reset the view (i.e. reset zoom/pan/etc - also reset all colormaps/gamma/opacity/etc)
        """
        self.napariviewer.reset_view()
        for layer in self.napariviewer.layers:
            layer.rendering = 'attenuated_mip'
            layer.visible = True
            layer.colormap = 'gray'
            layer.gamma = 1
            layer.opacity = 1

    def napari_doubleClicked(self, event:Event):
        """Callback when the mouse is double clicked. Brings someone to the candidate preview tab if a candidate is clicked

        """
        if self.underCursorInfoDF['current_candidate'][0][0] > -1:
            logging.info(self.underCursorInfoDF['current_candidate'][0][0])

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

        timeBetweenCursorUpdates = 0.0 #in seconds
        if self.timeOfLastCursorUpdate == 0 or (time.time() - self.timeOfLastCursorUpdate) >= timeBetweenCursorUpdates:
            #We get the canvas size/position in image pixel units
            canvas_size_in_px_units = event.source.size/self.napariviewer.camera.zoom
            # camera_coords = [self.napariviewer.camera.center[2]+.5, self.napariviewer.camera.center[1]+.5]
            camera_coords = [self.napariviewer.camera.center[2], self.napariviewer.camera.center[1]]
            canvas_pos = np.vstack([camera_coords-(canvas_size_in_px_units/2),camera_coords+(canvas_size_in_px_units/2)])
            #And we can normalize the cursor position to image pixels
            cursor_unit_norm = event._pos/event.source.size
            #Thus, we can find the pixel index - at the moment we simply calculate this
            highlighted_px_index=np.zeros((2,))
            highlighted_px_index[0] = (cursor_unit_norm[0]*canvas_size_in_px_units[0]+canvas_pos[0][0])/((float(self.settings['PixelSize_nm']['value']))/1000)+.5
            
            highlighted_px_index[1] = (cursor_unit_norm[1]*canvas_size_in_px_units[1]+canvas_pos[0][1])/((float(self.settings['PixelSize_nm']['value']))/1000)+.5

            #Here's the calculated pixel index in x,y coordinate.
            pixel_index = np.floor(highlighted_px_index).astype(int)

            self.underCursorInfoDF['current_pixel'][0] = np.floor(highlighted_px_index)
            #Also get the pixel index from the original EBS size:
            self.underCursorInfoDF['current_pixel_EBSCoord'][0] = [min(self.events['x']),min(self.events['y'])]+self.underCursorInfoDF['current_pixel'][0]
            
            self.updateUnderCursorInfo()
            self.timeOfLastCursorUpdate = time.time()

    def updateUnderCursorInfo(self):
        """
        Updates the under cursor information with details about time, pixel, and events. 
        """
        fullText = ''
        #Only update info if there are events
        if len(self.events)>0:
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

                minx = min(events['x'])
                miny = min(events['y'])
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
                        x_min = min(self.findingResult[f]['events']['x'])-minx
                        x_max = max(self.findingResult[f]['events']['x'])-minx
                        y_min = min(self.findingResult[f]['events']['y'])-miny
                        y_max = max(self.findingResult[f]['events']['y'])-miny

                        if (x_min <= self.underCursorInfoDF['current_pixel'][0][0] <= x_max) and (y_min <= self.underCursorInfoDF['current_pixel'][0][1] <= y_max):
                            self.underCursorInfoDF['current_candidate'][0][0] = f

                if self.underCursorInfoDF['current_candidate'][0][0] > -1:
                    fullText += f"; Candidate: {self.underCursorInfoDF['current_candidate'][0][0]}"

            if self.underCursorInfoDF['current_pixel_EBSCoord'][0][0] > -np.inf and self.underCursorInfoDF['current_pixel_EBSCoord'][0][1] > -np.inf and self.underCursorInfoDF['current_pixel_EBSCoord'][0][0] < np.inf and self.underCursorInfoDF['current_pixel_EBSCoord'][0][1] < np.inf:
                fullText += f"; Pixel coords: {int(self.underCursorInfoDF['current_pixel_EBSCoord'][0][0])}, {int(self.underCursorInfoDF['current_pixel_EBSCoord'][0][1])}"
            
            hotpixelarray = eval("["+self.settings['HotPixelIndexes']['value']+"]")
            for hotpixel in hotpixelarray:
                if self.underCursorInfoDF['current_pixel_EBSCoord'][0][0] == hotpixel[0] and self.underCursorInfoDF['current_pixel_EBSCoord'][0][1] == hotpixel[1]:
                    fullText += f"; Hot pixel - removed!"
                    break
                
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
        #Create a fullh istogram to get the min/max xy pos and such
        self.hist_xy = eventDistributions.SumPolarity(events)
        #Start keeping running total of min/max contrast limits
        minContrastLimit,maxContrastLimit = 0,0
        contrast_limit_percentile = 0.01
        for n in range(0,n_frames):
            #Get the events on this 'frame'
            events_this_frame = events[(events['t']>(float(timeStretch[0])*1000+n*frametime_ms*1000)) & (events['t']<(float(timeStretch[0])*1000+(n+1)*frametime_ms*1000))]
            #Create a 2d histogram out of this
            # thisHist_xy = eventDistributions.SumPolarity(events_this_frame)
            thisHist_xy = self.hist_xy(events_this_frame)
            #Add it to our image
            preview_multiD_image.append(thisHist_xy[0])
            #Keep a running total of minimum and maximum contrast limits
            minContrastVal = np.percentile(thisHist_xy[0], contrast_limit_percentile)
            maxContrastVal = np.percentile(thisHist_xy[0], 100-contrast_limit_percentile)
            if minContrastVal < minContrastLimit:
                minContrastLimit = minContrastVal
            if maxContrastVal > maxContrastLimit:
                maxContrastLimit = maxContrastVal

        #Scale should be the scale of pixel - to - um. E.g. a scale of 0.01 means 100 pixels = 1 um
        scale = ((float(settings['PixelSize_nm']['value']))/1000)
    
        #add this image to the napariviewer
        self.napariviewer.add_image(np.asarray(preview_multiD_image), multiscale=False,scale=(1,scale,scale))
        # self.napariviewer.add_image(np.asarray(preview_multiD_image), multiscale=False)

        #Append the finding/fitting overlay on top of this:
        self.napariviewer.layers.append(self.findingFitting_overlay)

        #Add a scalebar
        self.napariviewer.scale_bar.visible = True
        self.napariviewer.scale_bar.unit = "um"
        
        #Select the original image-layer as selected
        self.napariviewer.layers.selection.active = self.napariviewer.layers[0]
        
        #Set a proper brightness/contrast range:
        self.napariviewer.layers[0].contrast_limits = (minContrastLimit-2,maxContrastLimit+2)
        
        self.update_visibility()
        self.napariviewer.dims.events.current_step.connect(self.update_visibility)

    def update_visibility(self):
        """
        Function to update visibility based on the current layer
        """
        #Reset all data in the findingFitting overlay:
        self.findingFitting_overlay.data=[]
        
        settings=self.settings

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
        candidate_polygons_pos = []
        candidate_polygons_neg = []
        unfitted_candidate_polygons = []
        fitting_polygons = []
        candidates_ids_pos = []
        candidates_ids_neg = []
        candidates_ids = []
        unfitted_candidate_ids = []
        fitting_ids = []

        for f in findingResults:
            #Check if this findingResult should be displayed:
            t_min = min(findingResults[f]['events']['t'])/1000
            t_max = max(findingResults[f]['events']['t'])/1000
            if t_min < curr_time_bounds[1] and t_max > curr_time_bounds[0]:

                #Get the finding info...
                y_min = (min(findingResults[f]['events']['x']) - self.hist_xy.xlim[0])*(pxsize/1000)
                y_max = (max(findingResults[f]['events']['x']) - self.hist_xy.xlim[0])*(pxsize/1000)
                x_min = (min(findingResults[f]['events']['y']) - self.hist_xy.ylim[0])*(pxsize/1000)
                x_max = (max(findingResults[f]['events']['y']) - self.hist_xy.ylim[0])*(pxsize/1000)

                #Check if it succesfully fitted:
                if not np.isnan(fittingResults['x'].iloc[f]):
                    if len(np.unique(findingResults[f]['events']['p'])) == 1:
                        if np.all(np.unique(findingResults[f]['events']['p'])) == 0:
                            #Negative
                            candidates_ids_neg.append(str(f))
                            candidate_polygons_neg.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
                        elif np.all(np.unique(findingResults[f]['events']['p'])) == 1:
                            #Positive
                            candidates_ids_pos.append(str(f))
                            candidate_polygons_pos.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
                    else:
                        #Unknown or mixed
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
                #Getting the x,y coordinates
                xcoord_pxcoord = (fittingResults['x'].iloc[f]/float(pxsize)-self.hist_xy.xlim[0])*(pxsize/1000)
                ycoord_pxcoord = (fittingResults['y'].iloc[f]/float(pxsize)-self.hist_xy.ylim[0])*(pxsize/1000)

                #Variable for how big the cross itself should be drawn at
                fitting_cross_size = 1*(pxsize/1000)
                #Drawing the crosses
                fitting_polygons.append(np.array([[ycoord_pxcoord-fitting_cross_size,xcoord_pxcoord-fitting_cross_size],[ycoord_pxcoord+fitting_cross_size,xcoord_pxcoord+fitting_cross_size]]))
                fitting_polygons.append(np.array([[ycoord_pxcoord-fitting_cross_size,xcoord_pxcoord+fitting_cross_size],[ycoord_pxcoord+fitting_cross_size,xcoord_pxcoord-fitting_cross_size]]))

                #Need to append empty values to prevent NaNs to show up text-wise.
                fitting_ids.append("")
                fitting_ids.append("")

        all_ids = candidates_ids_neg+candidates_ids_pos+candidates_ids+unfitted_candidate_ids+fitting_ids
        # create features
        features = {
            'candidate_ids': all_ids,
        }
        text = {
            'string': '{candidate_ids}',
            'anchor': 'upper_left',
            'translation': [0, 0],
            'size': 8,
            'color':'cyan'
        }

        #Add the finding result
        self.findingFitting_overlay.add_rectangles(candidate_polygons_neg, edge_width=1*((float(settings['PixelSize_nm']['value']))/1000),edge_color=(79/255, 39/255, 119/255),face_color='transparent')
        self.findingFitting_overlay.add_rectangles(candidate_polygons_pos, edge_width=1*((float(settings['PixelSize_nm']['value']))/1000),edge_color=(30/255, 93/255, 73/255),face_color='transparent')
        self.findingFitting_overlay.add_rectangles(candidate_polygons, edge_width=1*((float(settings['PixelSize_nm']['value']))/1000),edge_color='coral',face_color='transparent')
        self.findingFitting_overlay.add_rectangles(unfitted_candidate_polygons, edge_width=1*((float(settings['PixelSize_nm']['value']))/1000),edge_color='red',face_color='transparent')
        self.findingFitting_overlay.add_lines(fitting_polygons, edge_width=0.5*((float(settings['PixelSize_nm']['value']))/1000),edge_color='red',face_color='transparent')
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
        canPreviewtab_horizontal_container.addWidget(self.prev_buttonCan)

        self.next_buttonCan = QPushButton("Next")
        self.next_buttonCan.clicked.connect(lambda: self.next_candidate_callback(parent))
        canPreviewtab_horizontal_container.addWidget(self.next_buttonCan)

        self.average_candidate_button = QPushButton("Average candidate")
        self.average_candidate_button.clicked.connect(lambda: self.average_candidate_callback(parent))
        canPreviewtab_horizontal_container.addWidget(self.average_candidate_button)
        
        self.adv_average_candidate_button = QPushButton("Store Advanced avg. candidate")
        self.adv_average_candidate_button.clicked.connect(lambda: self.adv_average_candidate_popup_callback(parent))
        canPreviewtab_horizontal_container.addWidget(self.adv_average_candidate_button)

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
        
        #Also update the show_candidate_callback if the dropdown is changed - needs to be after plt figurecanvas creation
        firstCanPrevDropdown.currentTextChanged.connect(lambda: self.show_candidate_callback(parent))
        secondCanPrevDropdown.currentTextChanged.connect(lambda: self.show_candidate_callback(parent))

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
                logging.debug('Tried to visualise candidate, but no ID was given!')
            elif 'FindingMethod' in parent.data and int(self.entryCanPreview.text()) < len(parent.data['FindingResult'][0]):
                self.CandidatePreviewID = int(self.entryCanPreview.text())
                logging.debug(f"Attempting to show candidate {self.CandidatePreviewID}.")
                # Get all localizations that belong to the candidate
                self.CandidatePreviewLocs = parent.data['FittingResult'][0][parent.data['FittingResult'][0]['candidate_id'] == self.CandidatePreviewID]
                # Check if fitting was successful
                if self.CandidatePreviewLocs['fit_info'].iloc[0] != '':
                    if pd.isna(self.CandidatePreviewLocs['x'].iloc[0]):
                        self.fit_info.setText(f"No localization generated due to {self.CandidatePreviewLocs['fit_info'].iloc[0]}")
                    else:
                        self.fit_info.setText(f"Average time was taken instead of fitted time due to {self.CandidatePreviewLocs['fit_info'].iloc[0]}")

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
        
    def adv_average_candidate_popup_callback(self,parent):
        #create a small popup:
            
        window = utils.SmallWindow(parent,windowTitle="Store advanced average candidate")
        window.addDescription("Calculate and store an 'advanced' average candidate. Keep in mind that this will take quite some time: ~ 10x longer than the normal average candidate. You can limit this by setting a max nr candidates if you want")
        
        maxNrCandidates = window.addTextEdit(labelText="MaxNrCandidates:",preFilledText="100000")
        StoreLoc = window.addFileLocation(labelText="Store location:", textAddPrePeriod = "_advAvgCandidate",textPostPeriod="pickle")
        
        
        button = window.addButton("Run")
        button.clicked.connect(lambda: self.adv_average_candidate_callback(parent,storeLoc=StoreLoc.text(),maxnrcandidates = int(maxNrCandidates.text())))
        
        window.show()

    def adv_average_candidate_callback(self,parent,storeLoc="",maxnrcandidates=1e9):
        #Check if storeLoc ends in .pickle, otherwise add it:
        if storeLoc[-7:]!= '.pickle':
            storeLoc = storeLoc + '.pickle'
            
        if 'FindingMethod' in parent.data:
            logging.debug(f"Attempting to calculate and store advanced average candidate.")
            pxsize = float(parent.globalSettings['PixelSize_nm']['value'])

            # get polarity options
            polarity_option = list(parent.data['FittingResult'][0].groupby('p').groups.keys())

            import copy
            # parameter initialization
            
            fullAvgPSF = {}
            fullAvgPSF['pos'] = pd.DataFrame()
            fullAvgPSF['neg'] = pd.DataFrame()
            fullAvgPSF['mix'] = pd.DataFrame()
            
            maxNrCandidates = min(maxnrcandidates,len(parent.data['FittingResult'][0]))
            for loc in range(0,maxNrCandidates):
                if np.mod(loc,100) == 0:
                    logging.info(f"Currently at: {loc} of {maxNrCandidates}, or {100*loc/maxNrCandidates:.2f}%")
                if parent.data['FittingResult'][0].iloc[loc]['fit_info'] != '':
                    continue
                else: 
    
                    candidate_id = parent.data['FittingResult'][0].iloc[loc]['candidate_id']
                    candidate = parent.data['FindingResult'][0][candidate_id]
                    eventsRaw = candidate['events'].copy()

                    eventsRaw['event_id'] = np.arange(1, len(eventsRaw) + 1)
                    eventsRaw['candidate_id'] = candidate_id
                    eventsRaw['pixel_incident_count'] = 0
                    eventsRaw['pixel_incidence_tot'] = 0
                    eventsRaw['t_delay'] = 0

                    eventsRaw.sort_values(by='t', inplace=True)
                    
                    #Extract the indeces of each indiv pixel:
                    pixel_indices = eventsRaw.groupby(['x', 'y']).groups
                    
                    for (x, y), indices in pixel_indices.items():
                        # Process each pixel# Process each pixel
                        pixel_events = eventsRaw.loc[indices]
                        n_px_events = len(pixel_events)
                        
                        # Use loc to set values for all indices at once
                        eventsRaw.loc[indices, 'pixel_incidence_tot'] = n_px_events
                        
                        # Sort pixel events by time
                        pixel_events_sorted = pixel_events.sort_values('t')
                        
                        # Calculate pixel_incidence_count and t_delay
                        eventsRaw.loc[indices, 'pixel_incident_count'] = range(1, int(n_px_events) + 1)
                        eventsRaw.loc[indices[1:], 't_delay'] = pixel_events_sorted['t'].diff().values[1:]

    
                    # eventsRaw.sort_values(by=['x','y','t'], inplace=True)

                    # Vectorized operations for final adjustments
                    fit_result = parent.data['FittingResult'][0].iloc[loc]
                    eventsRaw['x'] -= fit_result['x'] / pxsize
                    eventsRaw['y'] -= fit_result['y'] / pxsize
                    eventsRaw['t'] -= fit_result['t'] * 1000

                    events = eventsRaw
                    events = events.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if 2 in polarity_option: # mixed polarity case
                        fullAvgPSF['mix'] = pd.concat([fullAvgPSF['mix'], events], ignore_index=True)
                    else: # seperate polarity case
                        if all(events['p'] == 1):
                            #append all entries in events to sumalleventspos:
                            fullAvgPSF['pos'] = pd.concat([fullAvgPSF['pos'], events], ignore_index=True)
                        elif all(events['p'] == 0):
                            fullAvgPSF['neg'] = pd.concat([fullAvgPSF['neg'], events], ignore_index=True)
            
            
            #Store this as pickle:
            # picklePath = "\\\\IFMB-NAS\\AG Endesfelder\\Data\\Koen\\Scientific\\EBS_data\\aTub-AF647 normal density 2022-05-04\\AveragePSF.pickle"
            with open(storeLoc, 'wb') as f: 
                pickle.dump(fullAvgPSF, f)

    def average_candidate_callback(self, parent):
        if 'FindingMethod' in parent.data:
            logging.debug(f"Attempting to calculate and plot average candidate.")
            pxsize = float(parent.globalSettings['PixelSize_nm']['value'])

            # get polarity options
            polarity_option = list(parent.data['FittingResult'][0].groupby('p').groups.keys())

            import copy
                        
            #We loop over all localizations:
            if len(parent.data['AveragePSFmix']) == 0 and len(parent.data['AveragePSFpos']) == 0 and len(parent.data['AveragePSFneg']) == 0:
                # parameter initialization
                parent.data['avg_candidates_mix'] = 0
                parent.data['avg_candidates_pos'] = 0
                parent.data['avg_candidates_neg'] = 0
                parent.data['avg_cluster_size_mix'] *= 0.
                parent.data['avg_cluster_size_pos'] *= 0.
                parent.data['avg_cluster_size_neg'] *= 0.
                for loc in range(0,len(parent.data['FittingResult'][0])):
                    if np.mod(loc,1000) == 0:
                        logging.info(f"Currently at: {loc} of {len(parent.data['FittingResult'][0])}, or {100*loc/len(parent.data['FittingResult'][0]):.2f}%")
                    if parent.data['FittingResult'][0].iloc[loc]['fit_info'] != '':
                        continue
                    else: 
                        
                        #we grab its candidate id:
                        candidate_id = parent.data['FittingResult'][0].iloc[loc]['candidate_id']
                        #We find the corresponding candidate from findingResult:
                        candidate = copy.deepcopy(parent.data['FindingResult'][0][candidate_id])
                        #We take the events in there:
                        events = candidate['events'].copy()
                        
                        #Correct the event for the localization x, y, time:
                        events['x'] = events['x'] - parent.data['FittingResult'][0].iloc[loc]['x']/pxsize
                        events['y'] = events['y'] - parent.data['FittingResult'][0].iloc[loc]['y']/pxsize
                        events['t'] = events['t'] - parent.data['FittingResult'][0].iloc[loc]['t']*1000.  

                        if 2 in polarity_option: # mixed polarity case
                            parent.data['AveragePSFmix'] = pd.concat([parent.data['AveragePSFmix'], events], ignore_index=True)
                            parent.data['avg_candidates_mix'] += 1
                            parent.data['avg_cluster_size_mix'] += np.array([np.max(events['x'])-np.min(events['x']), np.max(events['y'])-np.min(events['y']), np.max(events['t'])-np.min(events['t'])])
                        else: # seperate polarity case
                            if all(events['p'] == 1):
                                #append all entries in events to sumalleventspos:
                                parent.data['AveragePSFpos'] = pd.concat([parent.data['AveragePSFpos'], events], ignore_index=True)
                                parent.data['avg_candidates_pos'] += 1
                                parent.data['avg_cluster_size_pos'] += np.array([np.max(events['x'])-np.min(events['x']), np.max(events['y'])-np.min(events['y']), np.max(events['t'])-np.min(events['t'])])
                            elif all(events['p'] == 0):
                                parent.data['AveragePSFneg'] = pd.concat([parent.data['AveragePSFneg'], events], ignore_index=True)
                                parent.data['avg_candidates_neg'] += 1
                                parent.data['avg_cluster_size_neg'] += np.array([np.max(events['x'])-np.min(events['x']), np.max(events['y'])-np.min(events['y']), np.max(events['t'])-np.min(events['t'])])
                if parent.data['avg_candidates_mix'] > 0:
                    parent.data['avg_cluster_size_mix'] /= parent.data['avg_candidates_mix']
                    parent.data['AveragePSFmix'].sort_values(by='t', inplace=True)
                if parent.data['avg_candidates_pos'] > 0:
                    parent.data['avg_cluster_size_pos'] /= parent.data['avg_candidates_pos']
                    parent.data['AveragePSFpos'].sort_values(by='t', inplace=True)
                    # parent.data['AveragePSFpos'].to_csv('C:\\Data\\EBS\\AveragePSF_GaussianT\\AveragePSF_pos.csv', index=False, header=True)
                if parent.data['avg_candidates_neg'] > 0:
                    parent.data['avg_cluster_size_neg'] /= parent.data['avg_candidates_neg']
                    parent.data['AveragePSFneg'].sort_values(by='t', inplace=True)
                    # parent.data['AveragePSFneg'].to_csv('C:\\Data\\EBS\\AveragePSF_GaussianT\\AveragePSF_neg.csv', index=False, header=True)
            
            # First clear text and figures
            self.candidate_info.setText('')
            self.fit_info.setText('')
            self.firstCandidateFigure.clf()
            self.secondCandidateFigure.clf()
            self.firstCandidateCanvas.draw()
            self.secondCandidateCanvas.draw()

            # dependend on the polarity option display different plots
            if 2 in polarity_option: # mixed polarity
                N_events = len(parent.data['AveragePSFmix'])
                self.candidate_info.setText(f"The average candidate has {N_events/parent.data['avg_candidates_mix']} events and has dimensions ({parent.data['avg_cluster_size_mix'][0]:.2f}, {parent.data['avg_cluster_size_mix'][1]:.2f}, {parent.data['avg_cluster_size_mix'][2]:.2f}).")
                FirstFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFmix']", "average_loc", "[]", "self.firstCandidateFigure","parent.globalSettings")
                SecondFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFmix']", "average_loc", "[]", "self.secondCandidateFigure","parent.globalSettings")

            elif 0 in polarity_option and 1 in polarity_option: # both polarities seperately
                N_events_pos = len(parent.data['AveragePSFpos'])
                N_events_neg = len(parent.data['AveragePSFneg'])
                cluster_info = f"The average positive candidate has {N_events_pos/parent.data['avg_candidates_pos']:.2f} events and dimensions ({parent.data['avg_cluster_size_pos'][0]:.2f}, {parent.data['avg_cluster_size_pos'][1]:.2f}, {parent.data['avg_cluster_size_pos'][2]:.2f}).\nThe average negative candidate has {N_events_neg/parent.data['avg_candidates_neg']:.2f} events and has dimensions ({parent.data['avg_cluster_size_neg'][0]:.2f}, {parent.data['avg_cluster_size_neg'][1]:.2f}, {parent.data['avg_cluster_size_neg'][2]:.2f})."
                self.candidate_info.setText(cluster_info)
                FirstFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFpos']", "average_loc", "[]", "self.firstCandidateFigure","parent.globalSettings")
                SecondFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFneg']", "average_loc", "[]", "self.secondCandidateFigure","parent.globalSettings")
                self.firstCandidateFigure.suptitle('Average positive candidate')
                self.secondCandidateFigure.suptitle('Average negative candidate')
            
            elif 1 in polarity_option: # only positive polarity
                N_events_pos = len(parent.data['AveragePSFpos'])
                cluster_info = f"The average positive candidate has {N_events_pos/parent.data['avg_candidates_pos']:.2f} events and dimensions ({parent.data['avg_cluster_size_pos'][0]:.2f}, {parent.data['avg_cluster_size_pos'][1]:.2f}, {parent.data['avg_cluster_size_pos'][2]:.2f})."
                self.candidate_info.setText(cluster_info)
                FirstFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFpos']", "average_loc", "[]", "self.firstCandidateFigure","parent.globalSettings")
                SecondFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFpos']", "average_loc", "[]", "self.secondCandidateFigure","parent.globalSettings")
            
            elif 0 in polarity_option: # only negative polarity
                N_events_neg = len(parent.data['AveragePSFneg'])
                cluster_info = f"The average negative candidate has {N_events_neg/parent.data['avg_candidates_neg']:.2f} events and dimensions ({parent.data['avg_cluster_size_neg'][0]:.2f}, {parent.data['avg_cluster_size_neg'][1]:.2f}, {parent.data['avg_cluster_size_neg'][2]:.2f})."
                self.candidate_info.setText(cluster_info)
                FirstFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFneg']", "average_loc", "[]", "self.firstCandidateFigure","parent.globalSettings")
                SecondFunctionEvalText = self.getCanPrevFunctionEvalText("parent.data['AveragePSFneg']", "average_loc", "[]", "self.secondCandidateFigure","parent.globalSettings")
                 
            eval(FirstFunctionEvalText)
            eval(SecondFunctionEvalText)
            
            # Quick lines if wanted to export the average PSF data
            # parent.data['AveragePSFpos'].to_csv(os.path.join('C:/Data/EBS/', 'AveragePSFpos.csv'))
            # parent.data['AveragePSFneg'].to_csv(os.path.join('C:/Data/EBS/', 'AveragePSFneg.csv'))

            # Improve figure layout and update the first canvases
            # self.firstCandidateFigure.tight_layout()
            self.firstCandidateCanvas.draw()
            # self.secondCandidateFigure.tight_layout()
            self.secondCandidateCanvas.draw()
        else:
            self.candidate_info.setText('Tried to visualise candidate but no data found!')
            logging.error('Tried to visualise candidate but no data found!')

class TableModel(QAbstractTableModel):
    """TableModel that heavily speedsup the table view
    Blatantly taken from https://stackoverflow.com/questions/71076164/fastest-way-to-fill-or-read-from-a-qtablewidget-in-pyqt5
    """

    def __init__(self, table_data, parent=None, header_names = None):
        super().__init__(parent)
        self.table_data = table_data
        self._header = header_names if header_names else table_data.columns

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return self.table_data.shape[0]

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return self.table_data.shape[1]

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if role == Qt.DisplayRole:
            return str(self.table_data.loc[index.row()][index.column()])

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._header[section])
        return super().headerData(section, orientation, role)

    def setColumn(self, col, array_items):
        """Set column data"""
        self.table_data[col] = array_items
        # Notify table, that data has been changed
        self.dataChanged.emit(QModelIndex(), QModelIndex())

    def getColumn(self, col):
        """Get column data"""
        return self.table_data[col]

class DataAnalysisWidget(QWidget):
    """
    Class that handles the Data Analysis tab.
    Allows dynamic loading of analysis scripts, execution, visualization, and saving.
    """
    def __init__(self, parent):
        self.parent = parent
        super().__init__()
        
        # Main layout
        self.mainlayout = QVBoxLayout()
        self.setLayout(self.mainlayout)
        
        # --- Control Panel (Analysis Settings) ---
        self.settingsGroupbox = QGroupBox("Analysis Settings")
        self.settingsGroupbox.setLayout(QGridLayout())
        self.settingsGroupbox.setObjectName("DataAnalysisSettingsGroupboxKEEP")
        
        # Dropdown for function selection
        self.analysisDropdown = QComboBox(self)
        self.analysisDropdown.setMaxVisibleItems(30)
        self.analysisDropdown.setObjectName("DataAnalysisDropdownKEEP")
        
        # Get options dynamically
        options = utils.functionNamesFromDir('DataAnalysis')
        self.displaynames, self.functionNameToDisplayNameMapping = utils.displayNamesFromFunctionNames(options, '')
        self.analysisDropdown.addItems(self.displaynames)
        
        # Connect dropdown change to dynamic layout update
        self.analysisDropdown.currentTextChanged.connect(
            lambda text: utils.changeLayout_choice(
                self.settingsGroupbox.layout(),
                "DataAnalysisDropdownKEEP",
                self.functionNameToDisplayNameMapping,
                parent=self,
                ignorePolarity=True
            )
        )
        
        # Add dropdown to layout
        self.settingsGroupbox.layout().addWidget(self.analysisDropdown, 0, 0, 1, 6)
        
        # Initial layout update
        utils.changeLayout_choice(
            self.settingsGroupbox.layout(),
            "DataAnalysisDropdownKEEP",
            self.functionNameToDisplayNameMapping,
            parent=self,
            ignorePolarity=True
        )

        # Batch Mode Checkbox
        self.batchModeCheckbox = QCheckBox("Batch Mode (Recursive Folder)")
        self.batchModeCheckbox.setObjectName("DataAnalysisBatchModeCheckboxKEEP")
        self.settingsGroupbox.layout().addWidget(self.batchModeCheckbox, 1, 0, 1, 6)
        
        # Analyze Button
        self.analyzeButton = QPushButton("Analyze")
        self.analyzeButton.setObjectName("DataAnalysisRunButtonKEEP")
        self.analyzeButton.clicked.connect(self.run_analysis_callback)
        self.settingsGroupbox.layout().addWidget(self.analyzeButton, 99, 0, 1, 6)
        
        self.mainlayout.addWidget(self.settingsGroupbox)
        
        # --- Results Area ---
        # Splitter for visualization and text results
        self.splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.mainlayout.addWidget(self.splitter)
        
        # Plot Area
        self.plotWidget = QWidget()
        self.plotLayout = QVBoxLayout()
        self.plotWidget.setLayout(self.plotLayout)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.plotLayout.addWidget(self.toolbar)
        self.plotLayout.addWidget(self.canvas)
        self.splitter.addWidget(self.plotWidget)
        
        # Results Text Area
        self.resultsText = QTextEdit()
        self.resultsText.setReadOnly(True)
        self.splitter.addWidget(self.resultsText)
        
        # Save Button
        self.saveButton = QPushButton("Save Results")
        self.saveButton.clicked.connect(self.save_results_callback)
        self.mainlayout.addWidget(self.saveButton)
        
        # Data storage
        self.current_results = None
        self.current_figure = None

    def run_analysis_callback(self):
        logging.info("Data Analysis button pressed")

        # Check for Batch Mode
        if self.batchModeCheckbox.isChecked():
            # Validate directory
            dataLocation = self.parent.dataLocationInput.text()
            if not os.path.isdir(dataLocation):
                 logging.error(f"Batch mode requires a directory input. Got: {dataLocation}")
                 QMessageBox.critical(self, "Input Error", "Batch mode requires a directory input. Please select a folder.")
                 return
            
            self._run_batch_analysis(dataLocation)
            return

        
        # Construct function call string
        # We need to construct the args. Primary args: data, settings
        # The dynamic inputs are handled by utils.getEvalTextFromGUIFunction
        
        # We need to find the function name selected
        methodName = ""
        text = self.analysisDropdown.currentText()
        for i in range(len(self.functionNameToDisplayNameMapping)):
            if self.functionNameToDisplayNameMapping[i][0] == text:
                methodName = self.functionNameToDisplayNameMapping[i][1]
                break
        
        if not methodName:
            logging.error("No analysis method selected.")
            return

        # Prepare kwargs extraction similar to PostProcessing
        # We implementation a simplified version of getPostProcessingFunctionEvalText here
        # or use utils.getEvalTextFromGUIFunction directly if we can gather the inputs.
        
        methodKwargNames = []
        methodKwargValues = []
        
        # Helper to find inputs in the groupbox
        # We iterate over children of the groupbox layout
        layout = self.settingsGroupbox.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget():
                    widget = item.widget()
                    self._extract_widget_value(widget, methodName, methodKwargNames, methodKwargValues)
                elif item.layout(): # Sometimes inputs are in sub-layouts (like file inputs)
                    sublayout = item.layout()
                    for j in range(sublayout.count()):
                         subitem = sublayout.itemAt(j)
                         if subitem.widget():
                             self._extract_widget_value(subitem.widget(), methodName, methodKwargNames, methodKwargValues)

        # Construct the eval string
        
        # Load the data
        dataLocation = self.parent.dataLocationInput.text()
        eventsDict = self.parent.loadRawData(dataLocation)
        
        if eventsDict is None or len(eventsDict) == 0:
            logging.error("Could not load data for analysis.")
            return

        # Combine events if multiple chunks (e.g. split polarity) are returned
        if len(eventsDict) > 1:
             events = np.vstack(eventsDict)
             # Sort by timestamp (column 0)
             events = events[events[:, 0].argsort()]
        else:
             events = eventsDict[0]
        
        partialString = "ev=events"
        
        evalText = utils.getEvalTextFromGUIFunction(
            methodName, 
            methodKwargNames, 
            methodKwargValues, 
            partialStringStart=partialString
        )
        
        logging.debug(f"Data Analysis Eval: {evalText}")
        
        if evalText:
            try:
                # Execute
                result = eval(evalText)
                print(f"DEBUG: Analysis result type: {type(result)}")
                # Unpack results: (figure, results_dict)
                if isinstance(result, tuple) and len(result) == 2:
                    self.current_figure = result[0]
                    self.current_results = result[1]
                    
                    # Update Plot
                    self.plotLayout.removeWidget(self.canvas)
                    self.plotLayout.removeWidget(self.toolbar)
                    self.canvas.close()
                    self.toolbar.close()
                    
                    self.canvas = FigureCanvas(self.current_figure)
                    self.toolbar = NavigationToolbar(self.canvas, self)
                    self.plotLayout.addWidget(self.toolbar)
                    self.plotLayout.addWidget(self.canvas)
                    self.canvas.draw()
                    
                    # Update Results Text
                    self.resultsText.setText(json.dumps(self.current_results, indent=4))
                    
                else:
                    logging.error("Analysis script must return (figure, results_dict)")
                    self.resultsText.setText("Error: Analysis script must return (figure, results_dict)")

            except Exception as e:
                logging.error(f"Analysis failed: {e}")
                logging.error(traceback.format_exc())
                QMessageBox.critical(self, "Analysis Error", str(e))

    def _run_batch_analysis(self, data_path):
        """
        Helper to run analysis on all supported files in a directory recursively.
        Groups files by parent directory and averages results.
        """

        print(f"DEBUG: _run_batch_analysis called with {data_path}")

        # 1. Search for files
        extensions = ['*.raw']
        files = []
        for ext in extensions:
            found = glob.glob(os.path.join(data_path, '**', ext), recursive=True)
            print(f"DEBUG: Searching {ext} in {data_path} -> Found {len(found)}")
            files.extend(found)

        print(f"DEBUG: Total files found: {files}")

        if not files:
            QMessageBox.information(self, "Batch Analysis", "No supported data files found in the directory.")
            print("DEBUG: No files found, returning.")
            return

        # Determine Method and Args
        methodName = ""
        text = self.analysisDropdown.currentText()
        print(f"DEBUG: Dropdown text: {text}")
        for i in range(len(self.functionNameToDisplayNameMapping)):
            if self.functionNameToDisplayNameMapping[i][0] == text:
                methodName = self.functionNameToDisplayNameMapping[i][1]
                break
        print(f"DEBUG: Resolved methodName: {methodName}")

        if not methodName:
            logging.error("No analysis method selected.")
            return

        # 2. Setup Results Directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(data_path, f"Analysis_Results_{methodName}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"DEBUG: Created results dir: {results_dir}")

        methodKwargNames = []
        methodKwargValues = []
        layout = self.settingsGroupbox.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget():
                    self._extract_widget_value(item.widget(), methodName, methodKwargNames, methodKwargValues)
                elif item.layout():
                    sublayout = item.layout()
                    for j in range(sublayout.count()):
                        subitem = sublayout.itemAt(j)
                        if subitem.widget():
                            self._extract_widget_value(subitem.widget(), methodName, methodKwargNames,
                                                       methodKwargValues)

        # 3. Group files by directory (parent folder)
        files_by_directory = {}
        for file_path in files:
            directory = os.path.dirname(file_path)
            if directory not in files_by_directory:
                files_by_directory[directory] = []
            files_by_directory[directory].append(file_path)

        print(f"DEBUG: Grouped into {len(files_by_directory)} directories")

        # 4. Iterate over directories and analyze
        aggregated_results = []
        directory_overlay_curves = {}

        logging.info(f"Starting batch analysis on {len(files_by_directory)} directories...")
        self.resultsText.setText(f"Starting batch analysis on {len(files_by_directory)} directories...\n")

        for directory, files_in_dir in sorted(files_by_directory.items()):
            try:
                dir_name = os.path.basename(directory)
                logging.info(f"Processing directory: {dir_name} ({len(files_in_dir)} files)...")
                self.resultsText.append(f"Processing: {dir_name}\n")

                dir_results = []
                dir_curves = []

                # Analyze each file in the directory
                for file_path in files_in_dir:
                    try:
                        filename = os.path.basename(file_path)
                        logging.debug(f"  Analyzing {filename}...")

                        # Load Data
                        eventsDict = self.parent.loadRawData(file_path)

                        if eventsDict is None or len(eventsDict) == 0:
                            logging.warning(f"Empty or invalid data for {filename}, skipping.")
                            continue

                        if len(eventsDict) > 1:
                            events = np.vstack(eventsDict)
                            events = events[events[:, 0].argsort()]
                        else:
                            events = eventsDict[0]

                        # Prepare Eval
                        partialString = "ev=events"
                        evalText = utils.getEvalTextFromGUIFunction(
                            methodName,
                            methodKwargNames,
                            methodKwargValues,
                            partialStringStart=partialString
                        )

                        # Execute
                        result = eval(evalText)

                        if isinstance(result, tuple) and len(result) == 2:
                            current_fig = result[0]
                            current_res = result[1]

                            # Save individual Figure
                            fig_name = f"{os.path.splitext(filename)[0]}_analysis.png"
                            fig_path = os.path.join(results_dir, fig_name)
                            current_fig.savefig(fig_path)
                            current_fig.clf()
                            plt.close(current_fig)

                            # Collect Results
                            if isinstance(current_res, dict):
                                row = current_res.copy()
                                row['filename'] = filename
                                row['directory'] = dir_name
                                dir_results.append(row)

                                # Accumulate curves for averaging
                                if 'curve_x' in current_res and 'curve_y' in current_res:
                                    dir_curves.append({
                                        'filename': filename,
                                        'x': np.array(current_res['curve_x']),
                                        'y': np.array(current_res['curve_y'])
                                    })
                            else:
                                logging.warning(f"Unknown result format for {filename}: {type(current_res)}")

                        else:
                            logging.warning(f"Script returned invalid format for {filename}")

                    except Exception as e:
                        logging.error(f"Failed to analyze {filename}: {e}")
                        self.resultsText.append(f"  Failed: {filename} ({str(e)})")
                        continue

                # 5. Average results per directory
                if dir_results:
                    avg_result = self._average_directory_results(dir_results)
                    avg_result['directory'] = dir_name
                    avg_result['num_files'] = len(dir_results)
                    aggregated_results.append(avg_result)

                    # Store directory curves for overlay
                    if dir_curves:
                        directory_overlay_curves[dir_name] = dir_curves

                logging.info(f"Completed directory: {dir_name}")

            except Exception as e:
                logging.error(f"Failed to process directory {directory}: {e}")
                self.resultsText.append(f"Failed directory: {directory} ({str(e)})")
                continue

        # 6. Create overlay plot per directory AND Collect Global Averages
        global_averages = []  # <--- New list to store data for the global plot

        for dir_name, curves in directory_overlay_curves.items():
            try:
                logging.info(f"Generating overlay plot for {dir_name}...")
                fig_overlay = plt.figure(figsize=(10, 6))

                # Plot individual curves
                for curve in curves:
                    plt.plot(curve['x'], curve['y'],
                             label=curve['filename'],
                             marker='o', markersize=3, alpha=0.5)

                # Calculate and plot average
                if curves:
                    # Calculate mean for the Global Plot (even if there is only 1 file)
                    avg_x = curves[0]['x']
                    avg_y = np.mean([c['y'] for c in curves], axis=0)

                    # Store this directory's average for the next step
                    global_averages.append({
                        'directory': dir_name,
                        'x': avg_x,
                        'y': avg_y
                    })

                    # Existing logic: Only draw the black "Average" line on the directory plot
                    # if there is more than 1 file (to avoid clutter on single-file plots)
                    if len(curves) > 1:
                        plt.plot(avg_x, avg_y,
                                 label='Average',
                                 linewidth=2.5, color='black', marker='s', markersize=5)

                plt.title(f"Summary: {dir_name}")
                plt.xlabel("Interval (μs)")
                plt.ylabel("Contrast Value")
                plt.legend()
                plt.grid(True)

                overlay_path = os.path.join(results_dir, f"overlay_{dir_name}.png")
                fig_overlay.savefig(overlay_path, dpi=150)
                plt.close(fig_overlay)
                logging.info(f"Overlay saved to {overlay_path}")
            except Exception as e:
                logging.error(f"Failed to create overlay for {dir_name}: {e}")

        # 7. Create Global Directory Comparison Overlay (NEW)
        if global_averages:
            try:
                logging.info("Generating global directory comparison plot...")
                fig_global = plt.figure(figsize=(12, 8))

                for item in global_averages:
                    plt.plot(item['x'], item['y'],
                             label=item['directory'],
                             linewidth=2, marker='o', markersize=4, alpha=0.8)

                plt.title("Global Comparison: Average per Directory")
                plt.xlabel("Interval (μs)")
                plt.ylabel("Contrast Value")
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # Move legend outside if many dirs
                plt.grid(True)
                plt.tight_layout()

                global_path = os.path.join(results_dir, "global_directory_comparison.png")
                fig_global.savefig(global_path, dpi=150)
                plt.close(fig_global)
                logging.info(f"Global comparison saved to {global_path}")
            except Exception as e:
                logging.error(f"Failed to create global comparison plot: {e}")

        # 8. Save summary CSV
        if aggregated_results:
            df = pd.DataFrame(aggregated_results)
            cols = ['directory', 'num_files'] + [c for c in df.columns if c not in ['directory', 'num_files']]
            df = df[cols]

            csv_path = os.path.join(results_dir, "directory_summary.csv")
            df.to_csv(csv_path, index=False)

            summary_msg = f"Batch Analysis Complete.\nProcessed {len(aggregated_results)} directories.\nResults saved to: {results_dir}\n\nSummary:\n{df.to_markdown()}"
            self.resultsText.setText(summary_msg)
            logging.info(f"Batch analysis complete. Results saved at {results_dir}")
        else:
            self.resultsText.append("\nBatch analysis finished with no successful results.")

    def _average_directory_results(self, results_list):
        """
        Average numeric results across multiple files in a directory.
        Handles both scalar values and array-like curves.
        """
        if not results_list:
            return {}

        avg_result = {}

        # Get all numeric keys from first result
        first_result = results_list[0]

        for key in first_result.keys():
            if key in ['filename', 'directory', 'curve_x', 'curve_y']:
                continue

            values = []
            for result in results_list:
                if key in result:
                    val = result[key]
                    # Try to convert to float for averaging
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        continue

            # Calculate average and std
            if values and len(values) == len(results_list):
                avg_result[f"{key}_mean"] = np.mean(values)
                avg_result[f"{key}_std"] = np.std(values)

        return avg_result

    def _extract_widget_value(self, widget, methodName, names, values):
        if "LineEdit" in widget.objectName() and widget.isVisible():
            parts = widget.objectName().split('#')
            if len(parts) >= 3 and parts[1] == methodName:
                names.append(parts[2])
                values.append(widget.text().replace('\\', '/'))
        elif "ComboBox" in widget.objectName() and widget.isVisible() and "Dropdown" not in widget.objectName():
             parts = widget.objectName().split('#')
             if len(parts) >= 3 and parts[1] == methodName:
                names.append(parts[2])
                values.append(widget.currentText())

    def save_results_callback(self):
        if self.current_results is None:
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results")
        if folder:
            try:
                # Save JSON
                with open(os.path.join(folder, "analysis_results.json"), 'w') as f:
                    json.dump(self.current_results, f, indent=4)
                
                # Save Figure
                if self.current_figure:
                    self.current_figure.savefig(os.path.join(folder, "analysis_plot.png"))
                    self.current_figure.savefig(os.path.join(folder, "analysis_plot.pdf"))
                
                QMessageBox.information(self, "Success", "Results saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))