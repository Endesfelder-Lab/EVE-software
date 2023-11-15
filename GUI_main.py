# from csbdeep.io import save_tiff_imagej_compatible
# from stardist import _draw_polygons, export_imagej_rois
import sys, os, logging, json, argparse, datetime, glob
import numpy as np
# Add the folder 2 folders up to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Import all scripts in the custom script folders
from CandidateFitting import *
from CandidateFinding import *
#Obtain the helperfunctions
from Utils import utils, utilsHelper

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog

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
        #Create a dummy label and add this to the global settings group box
        self.label = QLabel("Hello, GUI!")
        self.globalSettingsGroupBox.layout().addWidget(self.label)
        #Add the global settings group box to the central widget
        self.layout.addWidget(self.globalSettingsGroupBox, 0, 0)

        #Create a new group box
        self.mainGroupBox = QGroupBox()
        self.mainGroupBox.setLayout(QGridLayout())
        self.layout.addWidget(self.mainGroupBox, 2, 0)

        #Create a tab widget and add this to the main group box
        self.mainTabWidget = QTabWidget()
        self.mainTabWidget.setTabPosition(QTabWidget.South)
        self.mainGroupBox.layout().addWidget(self.mainTabWidget, 0, 0)
        
        self.tab_processing = QWidget()
        self.mainTabWidget.addTab(self.tab_processing, "Processing")
        self.tab_postProcessing = QWidget()
        self.mainTabWidget.addTab(self.tab_postProcessing, "Post-processing")
        self.tab_saveLoad = QWidget()
        self.mainTabWidget.addTab(self.tab_saveLoad, "Save/Load")
        self.tab_locList = QWidget()
        self.mainTabWidget.addTab(self.tab_locList, "LocalizationList")
        
        #Set up the tabs
        self.setup_tab('Processing')
        self.setup_tab('Post-processing')
        self.setup_tab('Save/Load')
        self.setup_tab('LocalizationList')
        
         # Create a button to trigger saving the entries
        self.save_button = QPushButton("Save GUI contents", self)
        self.save_button.clicked.connect(self.save_entries_to_json)
        self.layout.addWidget(self.save_button, 4, 0)
         # Create a button to trigger loading the entries
        self.load_button = QPushButton("Load GUI contents", self)
        self.load_button.clicked.connect(self.load_entries_from_json)
        self.layout.addWidget(self.load_button, 5, 0)

        #Loop through all combobox states briefly to initialise them (and hide them)
        self.set_all_combobox_states()
    
    def initGlobalSettings(self):
        globalSettings = {}
        globalSettings['PixelSize_nm'] = 80
        globalSettings['StoreConvertedRawData'] = True
        globalSettings['StoreFileMetadata'] = True
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
            'Processing': self.setup_processing,
            'Post-processing': self.setup_postProcessing,
            'Save/Load': self.setup_saveload,
            'LocalizationList': self.setup_loclist
        }
        #Run the setup of this tab
        setup_func = tab_mapping.get(tab_name)
        if setup_func:
            setup_func()
            
    def setup_processing(self):
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

        #Add a group box on candiddate fitting
        self.groupboxFinding = QGroupBox("Candidate finding")
        self.groupboxFinding.setObjectName("groupboxFinding")
        self.groupboxFinding.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.groupboxFinding.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFinding, 1, 0)
        
        # Create a QComboBox and add options - this is the FINDING dropdown
        self.candidateFindingDropdown = QComboBox(self)
        options = utils.functionNamesFromDir('CandidateFinding')
        self.candidateFindingDropdown.setObjectName("CandidateFinding_candidateFindingDropdown")
        self.candidateFindingDropdown.addItems(options)
        #Add the candidateFindingDropdown to the layout
        self.groupboxFinding.layout().addWidget(self.candidateFindingDropdown,1,0,1,2)
        #Activation for candidateFindingDropdown.activated
        self.candidateFindingDropdown.activated.connect(lambda: self.changeLayout_choice(self.groupboxFinding.layout(),"CandidateFinding_candidateFindingDropdown"))
        
        
        #On startup/initiatlisation: also do changeLayout_choice
        self.changeLayout_choice(self.groupboxFinding.layout(),"CandidateFinding_candidateFindingDropdown")
        
        
        self.groupboxFitting = QGroupBox("Candidate fitting")
        self.groupboxFitting.setObjectName("groupboxFitting")
        self.groupboxFitting.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.groupboxFitting.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFitting, 2, 0)
        
        # Create a QComboBox and add options - this is the FITTING dropdown
        self.candidateFittingDropdown = QComboBox(self)
        options = utils.functionNamesFromDir('CandidateFitting')
        self.candidateFittingDropdown.setObjectName("CandidateFitting_candidateFittingDropdown")
        self.candidateFittingDropdown.addItems(options)
        #Add the candidateFindingDropdown to the layout
        self.groupboxFitting.layout().addWidget(self.candidateFittingDropdown,1,0,1,2)
        #Activation for candidateFindingDropdown.activated
        self.candidateFittingDropdown.activated.connect(lambda: self.changeLayout_choice(self.groupboxFitting.layout(),"CandidateFitting_candidateFittingDropdown"))
        
        #On startup/initiatlisation: also do changeLayout_choice
        self.changeLayout_choice(self.groupboxFitting.layout(),"CandidateFitting_candidateFittingDropdown")
        
        
        #Add a run button:
        self.buttonProcessingRun = QPushButton("Run")
        self.buttonProcessingRun.clicked.connect(lambda: self.run_processing())
        tab_layout.layout().addWidget(self.buttonProcessingRun,3,0)

        #Add spacing
        # Add an empty QWidget with stretch factor of 1
        # empty_widget = QWidget()
        # tab_layout.addWidget(empty_widget, 999,0)
        tab_layout.setRowStretch(tab_layout.rowCount(), 1)

    def setup_postProcessing(self):
        tab2_layout = QGridLayout()
        self.tab_postProcessing.setLayout(tab2_layout)
        
        self.label2 = QLabel("Hello from Tab 2!")
        tab2_layout.addWidget(self.label2, 0, 0)

    def setup_saveload(self):
        tab3_layout = QGridLayout()
        self.tab_saveLoad.setLayout(tab3_layout)
        
        self.label3 = QLabel("Hello from Tab 3!")
        tab3_layout.addWidget(self.label3, 0, 0)
        
    def setup_loclist(self):
        tab4_layout = QGridLayout()
        self.tab_locList.setLayout(tab4_layout)
        
        self.label4 = QLabel("Hello from Tab 4!")
        tab4_layout.addWidget(self.label4, 0, 0)

    #Main def that interacts with a new layout based on whatever entries we have!
    #We assume a X-by-4 (4 columns) size, where 1/2 are used for Operation, and 3/4 are used for Value-->Score conversion
    def changeLayout_choice(self,curr_layout,className):
        logging.debug('Changing layout'+curr_layout.parent().objectName())
        #This removes everything except the first entry (i.e. the drop-down menu)
        self.resetLayout(curr_layout,className)
        #Get the dropdown info
        curr_dropdown = self.getMethodDropdownInfo(curr_layout,className)
        #Get the kw-arguments from the current dropdown.
        reqKwargs = utils.reqKwargsFromFunction(curr_dropdown.currentText())
        #Add a widget-pair for every kwarg
        labelposoffset = 0
        for k in range(len(reqKwargs)):
            #Value is used for scoring, and takes the output of the method
            if reqKwargs[k] != 'methodValue':
                label = QLabel(f"<b>{reqKwargs[k]}</b>")
                label.setObjectName(f"Label#{curr_dropdown.currentText()}#{reqKwargs[k]}")
                if self.checkAndShowWidget(curr_layout,label.objectName()) == False:
                    label.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=reqKwargs[k]))
                    curr_layout.addWidget(label,2+(k)+labelposoffset,0)
                line_edit = QLineEdit()
                line_edit.setObjectName(f"LineEdit#{curr_dropdown.currentText()}#{reqKwargs[k]}")
                if self.checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                    line_edit.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=reqKwargs[k]))
                    curr_layout.addWidget(line_edit,2+k+labelposoffset,1)
            else:
                labelposoffset -= 1
            
        #Get the optional kw-arguments from the current dropdown.
        optKwargs = utils.optKwargsFromFunction(curr_dropdown.currentText())
        #Add a widget-pair for every kwarg
        for k in range(len(optKwargs)):
            label = QLabel(f"<i>{optKwargs[k]}</i>")
            label.setObjectName(f"Label#{curr_dropdown.currentText()}#{optKwargs[k]}")
            if self.checkAndShowWidget(curr_layout,label.objectName()) == False:
                label.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=optKwargs[k]))
                curr_layout.addWidget(label,2+(k)+len(reqKwargs)+labelposoffset,0)
            line_edit = QLineEdit()
            line_edit.setObjectName(f"LineEdit#{curr_dropdown.currentText()}#{optKwargs[k]}")
            if self.checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                line_edit.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=optKwargs[k]))
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
            # Add /usr/lib/python3/dist-packages/ to PYTHONPATH to include Metavision libraries
            sys.path.append("C:\Program Files\Prophesee\lib\python3\site-packages") 
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
            if self.globalSettings['StoreConvertedRawData']:
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
        elif os.path.isfile(self.dataLocationInput.text()):
            self.processSingleFile(self.dataLocationInput.text())
    
    def processSingleFile(self,FileName):
        #Run the analysis on a single file
        self.currentFileInfo['CurrentFileLoc'] = FileName
        npyData = self.loadRawData(FileName)
        if npyData is not None:
            #Run the finding function!
            FindingEvalText = self.getFunctionEvalText('Finding',"npyData","self.globalSettings")
            if FindingEvalText is not None:
                self.data['FindingMethod'] = str(FindingEvalText)
                self.data['FindingResult'] = eval(str(FindingEvalText))
                logging.info('Candidate finding done!')
                logging.debug(self.data['FindingResult'])
                #Run the finding function!
                FittingEvalText = self.getFunctionEvalText('Fitting',"self.data['FindingResult'][0]","self.globalSettings")
                if FittingEvalText is not None:
                    self.data['FittingMethod'] = str(FittingEvalText)
                    self.data['FittingResult'] = eval(str(FittingEvalText))
                    logging.info('Candidate fitting done!')
                    logging.debug(self.data['FittingResult'])
                    
                    #Create and store the metadata
                    if self.globalSettings['StoreFileMetadata']:
                        self.createAndStoreFileMetadata()
                    
                else:
                    logging.error('Candidate fitting NOT performed')
            else:
                logging.error('Candidate finding NOT performed')
    
    def createAndStoreFileMetadata(self):
        logging.debug('Attempting to create and store file metadata')
        
        #Get the current chosen finding function and its parameters:
        
        
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
            with open(self.currentFileInfo['CurrentFileLoc'][:-4]+'_RunInfo_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.txt', 'w') as f:
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
                methodKwargValues_method.append(widget.text())
        
        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = self.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method,partialStringStart=str(p1)+','+str(p2))
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)
        if moduleMethodEvalTexts is not None:
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
        json_file_path = "GUI"+os.sep+"storage.json"

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
        json_file_path = "GUI"+os.sep+"storage.json"

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
                                    self.changeLayout_choice(self.groupboxFinding.layout(),field_widget.objectName())
                                elif 'Fitting' in field_widget.objectName():
                                    self.changeLayout_choice(self.groupboxFitting.layout(),field_widget.objectName())
        

        except FileNotFoundError:
            # Handle the case when the JSON file doesn't exist yet
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
                        self.changeLayout_choice(self.groupboxFinding.layout(),widget.objectName())
                    elif 'Fitting' in widget.objectName():
                        self.changeLayout_choice(self.groupboxFitting.layout(),widget.objectName())
            elif isinstance(widget, QWidget):
                for child_widget in widget.children():
                    set_combobox_states(child_widget)

        set_combobox_states(self)
        # Reset to orig states
        for combobox, original_state in original_states.items():
            combobox.setCurrentIndex(original_state)
            if 'Finding' in combobox.objectName():
                self.changeLayout_choice(self.groupboxFinding.layout(),combobox.objectName())
            elif 'Fitting' in combobox.objectName():
                self.changeLayout_choice(self.groupboxFitting.layout(),combobox.objectName())
        # except:
        #     pass






