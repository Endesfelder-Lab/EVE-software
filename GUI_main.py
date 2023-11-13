# from csbdeep.io import save_tiff_imagej_compatible
# from stardist import _draw_polygons, export_imagej_rois
import sys, os, logging
# Add the folder 2 folders up to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Import all scripts in the custom script folders
# from CandidateFitting import *
from CandidateFinding import *
#Obtain the helperfunctions
from Utils import utils, utilsHelper

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------

class MyGUI(QMainWindow):
    def __init__(self):
        self.unique_id = 0
        logging.basicConfig(level=logging.DEBUG)
        
        
        
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
        self.layout.addWidget(self.mainGroupBox, 1, 0)

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

        #Add a group box on candiddate fitting
        self.groupboxFinding = QGroupBox("Candidate finding")
        self.groupboxFinding.setObjectName("groupboxFinding")
        self.groupboxFinding.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.groupboxFinding.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFinding, 0, 0)
        
        
        self.groupboxFitting = QGroupBox("Candidate fitting")
        self.groupboxFitting.setObjectName("groupboxFitting")
        self.groupboxFinding.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.groupboxFitting.setLayout(QGridLayout())
        tab_layout.addWidget(self.groupboxFitting, 1, 0)

        self.label2 = QLabel("Hello from Tab 1!")
        self.groupboxFinding.layout().addWidget(self.label2)

        # self.button = QPushButton("Click me")
        # self.button.clicked.connect(self.on_button_click)
        # self.groupboxFitting.layout().addWidget(self.button, 1, 0)
        
        
        # Create a QComboBox and add options - this is the METHOD dropdown
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

    def on_button_click(self):
        self.label.setText("Button clicked!")
        print(utils.reqKwargsFromFunction("ShowCaseFinding.lowerUpperBound"))


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
                curr_layout.addWidget(label,2+(k)+labelposoffset,0)
                label.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=reqKwargs[k]))
                line_edit = QLineEdit()
                line_edit.setObjectName(f"LineEdit_{self.unique_id}#{curr_dropdown.currentText()}#{reqKwargs[k]}")
                line_edit.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=reqKwargs[k]))
                self.unique_id+=1
                curr_layout.addWidget(line_edit,2+k+labelposoffset,1)
            else:
                labelposoffset -= 1
            
        #Get the optional kw-arguments from the current dropdown.
        optKwargs = utils.optKwargsFromFunction(curr_dropdown.currentText())
        #Add a widget-pair for every kwarg
        for k in range(len(optKwargs)):
            label = QLabel(f"<i>{optKwargs[k]}</i>")
            label.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=optKwargs[k]))
            curr_layout.addWidget(label,2+(k)+len(reqKwargs)+labelposoffset,0)
            line_edit = QLineEdit()
            line_edit.setObjectName(f"LineEdit_{self.unique_id}#{curr_dropdown.currentText()}#{optKwargs[k]}")
            line_edit.setToolTip(utils.infoFromMetadata(curr_dropdown.currentText(),specificKwarg=optKwargs[k]))
            self.unique_id+=1
            curr_layout.addWidget(line_edit,2+(k)+len(reqKwargs)+labelposoffset,1)
        

    #Remove everythign in this layout except className_dropdown
    def resetLayout(self,curr_layout,className):
        for index in range(curr_layout.count()):
            widget_item = curr_layout.itemAt(index)
            # Check if the item is a widget (as opposed to a layout)
            if widget_item.widget() is not None:
                widget = widget_item.widget()
                #If it's the dropdown segment, label it as such
                if not ("candidateFindingDropdown" in widget.objectName()) and not ("scoringDropdown" in widget.objectName()) and widget.objectName() != f"titleLabel_{className}" and not ("KEEP" in widget.objectName()):
                    logging.debug(f"Deleting {widget.objectName()}")
                    widget.deleteLater()
    
    
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
    
    def run_processing(self):
        #First runf for finding:
        #Get the dropdown info
        moduleMethodEvalTexts = []
        methodName_finding = self.candidateFindingDropdown.currentText()
        all_layouts = self.findChild(QWidget, "groupboxFinding").findChildren(QLayout)[0]
        
        methodKwargNames_method = []
        methodKwargValues_method = []
        methodName_method = ''
        # Iterate over the items in the layout
        for index in range(all_layouts.count()):
            item = all_layouts.itemAt(index)
            widget = item.widget()
            
            if ("LineEdit" in widget.objectName()):
                # The objectName will be along the lines of foo#bar#str
                #Check if the objectname is part of a method or part of a scoring
                split_list = widget.objectName().split('#')
                methodName_method = split_list[1]
                methodKwargNames_method.append(split_list[2])
                methodKwargValues_method.append(widget.text())
        
        #Function call: get the to-be-evaluated text out, giving the methodName, method KwargNames, methodKwargValues, and 'function Type (i.e. cellSegmentScripts, etc)' - do the same with scoring as with method
        if methodName_method != '':
            EvalTextMethod = self.getEvalTextFromGUIFunction(methodName_method, methodKwargNames_method, methodKwargValues_method)
            #append this to moduleEvalTexts
            moduleMethodEvalTexts.append(EvalTextMethod)
            # item = all_layouts.itemAt(index)
            # # Check if the item is a QWidget
            # if isinstance(item, QtWidgets.QWidgetItem):
            #     widget = item.widget()
            #     # Check if the widget is a PyQt5 module
            #     if isinstance(widget, QtWidgets.QWidget):
            #         # Do something with the PyQt5 module
            #         print(widget)
        
        print(eval(str(moduleMethodEvalTexts[0])))
        pass
        # #Get the kw-arguments from the current dropdown.
        # reqKwargs = utils.reqKwargsFromFunction(curr_dropdown.currentText())
        # #Get the optional kw-arguments from the current dropdown.
        # optKwargs = utils.optKwargsFromFunction(curr_dropdown.currentText())
        # #Get the values from the line edits
        

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
                        print(f'EMPTY VALUE of {kwargvalue}, NOT CONTINUING')
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
                    print('NOT ALL KWARGS PROVIDED!')
            else:
                print('SOMETHING VERY STUPID HAPPENED')
                return None
# testImageLoc = "./AutonomousMicroscopy/ExampleData/BF_test_avg.tiff"

# # Open the TIFF file
# with tifffile.TiffFile(testImageLoc) as tiff:
#     # Read the image data
#     ImageData = tiff.asarray()

# #Non-preloaded stardistsegmentation
# coords = eval(HelperFunctions.createFunctionWithKwargs("StarDist.StarDistSegment",image_data="ImageData",modelStorageLoc="\"./AutonomousMicroscopy/ExampleData/StarDistModel\"",prob_thresh="0.35",nms_thresh="0.2"))

# coordim = outlineCoordinatesToImage(coords)

# #Three examples
# cellCrowdedness = eval(HelperFunctions.createFunctionWithKwargs("SimpleCellOperants.nrneighbours_basedonCellWidth",outline_coords="coords",multiple_cellWidth_lookup="1"))
# cellCrowdedness_gauss=eval(HelperFunctions.createFunctionWithKwargs("DefaultScoringMetrics.gaussScore",methodValue="cellCrowdedness",meanVal="3",sigmaVal="2"))

# cellArea = eval(HelperFunctions.createFunctionWithKwargs("SimpleCellOperants.cellArea",outline_coords="coords"))
# cellArea_bounds=eval(HelperFunctions.createFunctionWithKwargs("DefaultScoringMetrics.lowerUpperBound",methodValue="cellArea",lower_bound="300",upper_bound="500"))

# cellAssym = eval(HelperFunctions.createFunctionWithKwargs("SimpleCellOperants.lengthWidthRatio",outline_coords="coords"))
# cellAssym_relativeToMax = eval(HelperFunctions.createFunctionWithKwargs("DefaultScoringMetrics.relativeToMaxScore",methodValue="cellAssym"))
