# from csbdeep.io import save_tiff_imagej_compatible
# from stardist import _draw_polygons, export_imagej_rois
import sys, os, logging, json, argparse, datetime, glob, csv, ast, platform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
# Add the folder 2 folders up to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Import all scripts in the custom script folders
from CandidateFitting import *
from CandidateFinding import *
#Obtain the helperfunctions
from Utils import utils, utilsHelper

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------

class MyGUI(QMainWindow):
    def __init__(self):
        #Create parser
        parser = argparse.ArgumentParser(description='EBS fitting - Endesfelder lab - Nov 2023')
        parser.add_argument('--debug', '-d', action='store_true', help='Enable debug')
        args=parser.parse_args()
        if args.debug:
            log_format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
            logging.basicConfig(format=log_format, level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Create a dictionary to store the entries
        self.entries = {}
        
        #Create a dictionary to store the global Settings
        self.globalSettings = self.initGlobalSettings()
        
        #Create a dictionary that stores info about each file being run (i.e. for metadata)
        self.currentFileInfo = {}
        
        #Create a dictionary that stores data and passes it between finding,fitting,saving, etc
        self.data = {}
        
        
        #Set some major settings on the UI
        super().__init__()
        self.setWindowTitle("EBS fitting - Endesfelder lab - Nov 2023")
        self.setMinimumSize(400, 300)  # Set minimum size for the GUI window
        
        #Set the central widget that contains everything
        #This is a group box that contains a grid layout. We fill everything inside this grid layout
        self.central_widget = QGroupBox()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        #Create a global settings group box
        self.globalSettingsGroupBox = QGroupBox("Global settings")
        self.globalSettingsGroupBox.setLayout(QGridLayout())
        
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

        #Create a tab widget and add this to the main group box
        self.mainTabWidget = QTabWidget()
        self.mainTabWidget.setTabPosition(QTabWidget.South)
        self.layout.addWidget(self.mainTabWidget, 2, 0)
        
        self.tab_processing = QWidget()
        self.mainTabWidget.addTab(self.tab_processing, "Processing")
        self.tab_postProcessing = QWidget()
        self.mainTabWidget.addTab(self.tab_postProcessing, "Post-processing")
        self.tab_locList = QWidget()
        self.mainTabWidget.addTab(self.tab_locList, "LocalizationList")
        self.tab_visualisation = QWidget()
        self.mainTabWidget.addTab(self.tab_visualisation, "Visualisation")
        
        #Set up the tabs
        self.setup_tab('Processing')
        self.setup_tab('Post-processing')
        self.setup_tab('LocalizationList')
        self.setup_tab('Visualisation')

        #Loop through all combobox states briefly to initialise them (and hide them)
        self.set_all_combobox_states()
        
        #Load the GUI settings from last time:
        self.load_entries_from_json()
    
    def open_advanced_settings(self):
        self.advancedSettingsWindow.show()
    
    def initGlobalSettings(self):
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
        
        globalSettings['JSONGUIstorePath'] = {}
        globalSettings['JSONGUIstorePath']['value'] = "GUI"+os.sep+"storage.json"
        globalSettings['JSONGUIstorePath']['input'] = str
        globalSettings['GlobalOptionsStorePath'] = {}
        globalSettings['GlobalOptionsStorePath']['value'] = "GUI" + os.sep + "GlobSettingStorage.json"
        globalSettings['GlobalOptionsStorePath']['input'] = str
        
        globalSettings['IgnoreInOptions'] = ('IgnoreInOptions','StoreFinalOutput', 'JSONGUIstorePath','GlobalOptionsStorePath') #Add options here that should NOT show up in the global settings window
        return globalSettings
    
    # Function to handle the button click event
    def datasetSearchButtonClicked(self):
        logging.debug('data lookup search button clicked')
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File",filter="EBS files (*.raw *.npy);;All Files (*)")
        if file_path:
            self.dataLocationInput.setText(file_path)
    
    def datasetFolderButtonClicked(self):
        logging.debug('data lookup Folder button clicked')
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.dataLocationInput.setText(folder_path)
    
    def setup_tab(self, tab_name):
        tab_mapping = {
            'Processing': self.setup_processingTab,
            'Post-processing': self.setup_postProcessingTab,
            'Save/Load': self.setup_saveloadTab,
            'Visualisation': self.setup_visualisationTab,
            'LocalizationList': self.setup_loclistTab
        }
        #Run the setup of this tab
        setup_func = tab_mapping.get(tab_name)
        if setup_func:
            setup_func()
            
    def setup_processingTab(self):
        #Create a grid layout and set it
        tab_layout = QGridLayout()
        self.tab_processing.setLayout(tab_layout)
        
        #add a box with multiple horizontally oriented entires:
        self.datasetLocation_layout = QGridLayout()
        tab_layout.addLayout(self.datasetLocation_layout, 0, 0)
        
        # Add a label:
        self.datasetLocation_label = QLabel("Dataset location:")
        self.datasetLocation_layout.addWidget(self.datasetLocation_label, 0, 0,2,1)
        # Create the input field
        self.dataLocationInput = QLineEdit()
        self.dataLocationInput.setObjectName("processing_dataLocationInput")
        self.datasetLocation_layout.layout().addWidget(self.dataLocationInput, 0, 1,2,1)
        # Create the search button
        self.datasetSearchButton = QPushButton("File...")
        self.datasetSearchButton.clicked.connect(self.datasetSearchButtonClicked)
        self.datasetLocation_layout.layout().addWidget(self.datasetSearchButton, 0, 2)
        self.datasetFolderButton = QPushButton("Folder...")
        self.datasetFolderButton.clicked.connect(self.datasetFolderButtonClicked)
        self.datasetLocation_layout.layout().addWidget(self.datasetFolderButton, 1, 2)
        #Add the global settings group box to the central widget
        # self.layout.addWidget(self.datasetsSettingsGroupBox, 1, 0)

        #Add a group box on candiddate finding
        self.groupboxFinding = QGroupBox("Candidate finding")
        self.groupboxFinding.setObjectName("groupboxFinding")
        self.groupboxFinding.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.groupboxFinding.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFinding, 1, 0)
        
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
        
        
        self.groupboxFitting = QGroupBox("Candidate fitting")
        self.groupboxFitting.setObjectName("groupboxFitting")
        self.groupboxFitting.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.groupboxFitting.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFitting, 2, 0)
        
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
        
        
        #Add a run button:
        self.buttonProcessingRun = QPushButton("Run")
        self.buttonProcessingRun.clicked.connect(lambda: self.run_processing())
        tab_layout.layout().addWidget(self.buttonProcessingRun,3,0)

        #Add spacing
        # Add an empty QWidget with stretch factor of 1
        # empty_widget = QWidget()
        # tab_layout.addWidget(empty_widget, 999,0)
        tab_layout.setRowStretch(tab_layout.rowCount(), 1)

    def setup_postProcessingTab(self):
        tab2_layout = QGridLayout()
        self.tab_postProcessing.setLayout(tab2_layout)
        
        self.label2 = QLabel("Hello from Tab 2!")
        tab2_layout.addWidget(self.label2, 0, 0)

    def setup_saveloadTab(self):
        tab3_layout = QGridLayout()
        self.tab_saveLoad.setLayout(tab3_layout)
        
        self.label3 = QLabel("Hello from Tab 3!")
        tab3_layout.addWidget(self.label3, 0, 0)
        
    def setup_loclistTab(self):
        tab4_layout = QGridLayout()
        self.tab_locList.setLayout(tab4_layout)
        
        #Create an empty table and add this:
        self.LocListTable = QTableWidget()
        tab4_layout.addWidget(self.LocListTable, 0, 0)
        
    def setup_visualisationTab(self):
        #Add a vertical layout, not a grid layout:
        visualisationTab_vertical_container = QVBoxLayout()
        self.tab_visualisation.setLayout(visualisationTab_vertical_container)
        
        #Add a horizontal layout to the first row of the vertical layout:
        visualisationTab_horizontal_container = QHBoxLayout()
        visualisationTab_vertical_container.addLayout(visualisationTab_horizontal_container)
        
        #Add a button that says scatter:
        self.buttonScatter = QPushButton("Scatter Plot")
        visualisationTab_horizontal_container.addWidget(self.buttonScatter)
        #Give it a function on click:
        self.buttonScatter.clicked.connect(lambda: self.plotScatter())
        
        self.data['figurePlot'], self.data['figureAx'] = plt.subplots(figsize=(5, 5))
        self.data['figureCanvas'] = FigureCanvas(self.data['figurePlot'])
        
        self.data['figurePlot'].tight_layout()
        
        visualisationTab_vertical_container.addWidget(NavigationToolbar(self.data['figureCanvas'], self))
        
        visualisationTab_vertical_container.addWidget(self.data['figureCanvas'])
    
    def plotScatter(self):
        #check if we have results stored:
        if 'FittingMethod' in self.data:
            logging.debug('Attempting to show scatter plot')
            #Delete any existing cbar - needs to be done before anything else:
            if hasattr(self, 'cbar'):
                self.cbar.remove()
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
            if self.checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                line_edit.setToolTip(utils.infoFromMetadata(current_selected_function,specificKwarg=optKwargs[k]))
                curr_layout.addWidget(line_edit,2+(k)+len(reqKwargs)+labelposoffset,1)
    
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
            return np.load(dataLocation)
        elif dataLocation.endswith('.raw'):
            events = self.RawToNpy(dataLocation)
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
            self.processSingleFile(self.dataLocationInput.text())
        #If it's nothing - only continue if we actually load a finding result:
        else:
            if 'ExistingFinding' in self.candidateFindingDropdown.currentText():
                logging.info('Skipping finding processing, going to fitting')
                
                #Ensure that we don't store the finding result
                origStoreFindingSetting=self.globalSettings['StoreFindingOutput']['value']
                self.globalSettings['StoreFindingOutput']['value'] = False
                
                self.processSingleFile(self.dataLocationInput.text(),onlyFitting=True)
                
                #Reset the global setting:
                self.globalSettings['StoreFindingOutput']['value']=origStoreFindingSetting
         
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
     
    def processSingleFile(self,FileName,onlyFitting=False):
        if not onlyFitting:
            #Run the analysis on a single file
            self.currentFileInfo['CurrentFileLoc'] = FileName
            npyData = self.loadRawData(FileName)
            if npyData is None:
                return
        #If we only fit, we still run more or less the same info, butwe don't care about the npyData in the CurrentFileLoc.
        elif onlyFitting:
            self.currentFileInfo['CurrentFileLoc'] = FileName
            logging.info('Candidate finding NOT performed')
            npyData = None
            
        #Run the finding function!
        FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings")
        if FindingEvalText is not None:
            self.data['FindingMethod'] = str(FindingEvalText)
            self.data['FindingResult'] = eval(str(FindingEvalText))
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
                self.data['FittingMethod'] = str(FittingEvalText)
                self.data['FittingResult'] = eval(str(FittingEvalText))
                logging.info('Candidate fitting done!')
                logging.debug(self.data['FittingResult'])
                
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
        import pickle
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

Custom output from finding function:
{self.data['FindingResult'][1]}

---- Fitting metadata output: ----
Methodology used:
{self.data['FittingMethod']}

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
                        
            if ("LineEdit" in widget.objectName()) and widget.isVisible():
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
                if isinstance(widget,QComboBox) and widget.isVisible() and className in widget.objectName():
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
                for child_widget in widget.children():
                    find_editable_fields(child_widget)

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