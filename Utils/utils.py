import os
import warnings
import inspect
import importlib
import re
import warnings, logging
import numpy as np
import itertools
from Utils import utilsHelper

#Import all scripts in the custom script folders
from CandidateFinding import *
from CandidateFitting import *

#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider, QSpacerItem
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile, QThread, pyqtSignal, QObject

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Function declarations
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------

#Returns whether a function exists and is callable
def function_exists(obj):
    return callable(obj) and inspect.isfunction(obj)

#Returns whether a subfunction exists specifically in module_name and is callable
def subfunction_exists(module_name, subfunction_name):
    try:
        if module_name.endswith('.py'):
            # Module path is provided
            loader = importlib.machinery.SourceFileLoader('', module_name) #type:ignore
            module = loader.load_module()
        else:
            module = importlib.import_module(module_name)
        a = hasattr(module, subfunction_name)
        b = callable(getattr(module, subfunction_name))
        return hasattr(module, subfunction_name) and callable(getattr(module, subfunction_name))
    except (ImportError, AttributeError):
        return False
    
# Return all functions that are found in a specific directory
def functionNamesFromDir(dirname):
    #Get the absolute path, assuming that this file will stay in the sister-folder
    absolute_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),dirname)
    #initialise empty array
    functionnamearr = []
    #Loop over all files
    for file in os.listdir(absolute_path):
        #Check if they're .py files
        if file.endswith(".py"):
            #Check that they're not init files or similar
            if not file.startswith("_"):
                #Get the function name
                functionName = file[:-3]
                #Get the metadata from this function and from there obtain
                try:
                    functionMetadata = eval(f'{str(functionName)}.__function_metadata__()')
                    for singlefunctiondata in functionMetadata:
                        #Also check this against the actual sub-routines and raise an error (this should also be present in the __init__ of the folders)
                        subroutineName = f"{functionName}.{singlefunctiondata}"
                        if subfunction_exists(f'{absolute_path}{os.sep}{functionName}.py',singlefunctiondata): #type:ignore
                            functionnamearr.append(subroutineName)
                        else:
                            warnings.warn(f"Warning: {subroutineName} is present in __function_metadata__ but not in the actual file!")
                #Error handling if __function_metadata__ doesn't exist
                except AttributeError:
                    #Get all callable subroutines and store those
                    subroutines = []
                    for subroutineName, obj in inspect.getmembers(eval(f'{functionName}')):
                        if function_exists(obj):
                            subroutines.append(subroutineName)
                            functionnamearr.append(subroutineName)
                    #Give the user the warning and the solution
                    warnings.warn(f"Warning: {str(functionName)} does not have the required __function_metadata__ ! All functions that are found in this module are added! They are {subroutines}")
    #return all functions
    return functionnamearr

#Returns the 'names' of the required kwargs of a function
def reqKwargsFromFunction(functionname):
    #Get all kwarg info
    allkwarginfo = kwargsFromFunction(functionname)
    #Perform a regex match on 'name'
    name_pattern = r"name:\s*(\S+)"
    #Get the names of the req_kwargs (allkwarginfo[0])
    names = re.findall(name_pattern, allkwarginfo[0][0])
    return names

#Returns the 'names' of the optional kwargs of a function
def optKwargsFromFunction(functionname):
    #Get all kwarg info
    allkwarginfo = kwargsFromFunction(functionname)
    #Perform a regex match on 'name'
    name_pattern = r"name:\s*(\S+)"
    #Get the names of the optional kwargs (allkwarginfo[1])
    names = re.findall(name_pattern, allkwarginfo[1][0])
    return names

def distKwargValuesFromFittingFunction(functionname):
    #Get all kwarg info
    allkwarginfo = kwargsFromFunction(functionname)
    derivedClasses = []
    if allkwarginfo[2] != []:
        base_pattern = r"base:\s*(\S+)"
        base_name = re.findall(base_pattern, allkwarginfo[2][0])[0]
        # get base class
        baseClass = getattr(utilsHelper, base_name, None)
        if not baseClass == None:
            # get all derived classes that share common base
            for name, obj in inspect.getmembers(utilsHelper):
                if inspect.isclass(obj) and issubclass(obj, baseClass) and obj != baseClass:
                    derivedClasses.append(name)
    return derivedClasses

def defaultOptionFromDistKwarg(functionname):
    #Check if the function has a 'default' option for the distribution kwarg. 
    defaultOption=None
    functionparent = functionname.split('.')[0]
    #Get the full function metadata
    functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')[functionname.split('.')[1]]
    if "dist_kwarg" in functionMetadata:
        if "default_option" in functionMetadata["dist_kwarg"]:
            defaultOption = functionMetadata["dist_kwarg"]["default_option"]
            # check if defaultOption is a valid option
            defaultOption = getattr(utilsHelper, defaultOption, None)
            if defaultOption != None:
                defaultOption = defaultOption.__name__
    return defaultOption

def getInfoFromDistribution(distribution):
    description = None
    distClass = getattr(utilsHelper, distribution, None)
    if not distClass == None:
        try:
            description = distClass.description
        except AttributeError:
            pass
    return description

#Obtain the kwargs from a function. Results in an array with entries
def kwargsFromFunction(functionname):
    try:
        #Check if parent function
        if not '.' in functionname:
            functionMetadata = eval(f'{str(functionname)}.__function_metadata__()')
            #Loop over all entries
            looprange = range(0,len(functionMetadata))
        else: #or specific sub-function
            #get the parent info
            functionparent = functionname.split('.')[0]
            functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')
            #sub-select the looprange
            loopv = next((index for index in range(0,len(functionMetadata)) if list(functionMetadata.keys())[index] == functionname.split('.')[1]), None)
            looprange = range(loopv,loopv+1) #type:ignore
        name_arr = []
        help_arr = []
        rkwarr_arr = []
        okwarr_arr = []
        dist_kwarg = []
        loopindex = 0
        for i in looprange:
            #Get name text for all entries
            name_arr.append([list(functionMetadata.keys())[i]])
            #Get help text for all entries
            help_arr.append(functionMetadata[list(functionMetadata.keys())[i]]["help_string"])
            #Get text for all the required kwarrs
            txt = ""
            #Loop over the number or rkwarrs
            for k in range(0,len(functionMetadata[list(functionMetadata.keys())[i]]["required_kwargs"])):
                zz = functionMetadata[list(functionMetadata.keys())[i]]["required_kwargs"][k]
                for key, value in functionMetadata[list(functionMetadata.keys())[i]]["required_kwargs"][k].items():
                    txt += f"{key}: {value}\n"
            rkwarr_arr.append(txt)
            #Get text for all the optional kwarrs
            txt = ""
            #Loop over the number of okwarrs
            for k in range(0,len(functionMetadata[list(functionMetadata.keys())[i]]["optional_kwargs"])):
                zz = functionMetadata[list(functionMetadata.keys())[i]]["optional_kwargs"][k]
                for key, value in functionMetadata[list(functionMetadata.keys())[i]]["optional_kwargs"][k].items():
                    txt += f"{key}: {value}\n"
            okwarr_arr.append(txt)
            #Get text for distribution kwargs
            txt = ""
            if "dist_kwarg" in functionMetadata[list(functionMetadata.keys())[i]]:
                for key, value in functionMetadata[list(functionMetadata.keys())[i]]["dist_kwarg"].items():
                    txt += f"{key}: {value}\n"
                dist_kwarg.append(txt)
    #Error handling if __function_metadata__ doesn't exist
    except AttributeError:
        rkwarr_arr = []
        okwarr_arr = []
        dist_kwarg = []
        return f"No __function_metadata__ in {functionname}"
            
    return [rkwarr_arr, okwarr_arr, dist_kwarg]

#Obtain the help-file and info on kwargs on a specific function
#Optional: Boolean kwarg showKwargs & Boolean kwarg showHelp
def infoFromMetadata(functionname,**kwargs):
    showKwargs = kwargs.get('showKwargs', True)
    showHelp = kwargs.get('showHelp', True)
    specificKwarg = kwargs.get('specificKwarg', False)
    try:
        skipfinalline = False
        #Check if parent function
        if not '.' in functionname:
            functionMetadata = eval(f'{str(functionname)}.__function_metadata__()')
            finaltext = f"""\
            --------------------------------------------------------------------------------------
            {functionname} contains {len(functionMetadata)} callable functions: {", ".join(str(singlefunctiondata) for singlefunctiondata in functionMetadata)}
            --------------------------------------------------------------------------------------
            """
            #Loop over all entries
            looprange = range(0,len(functionMetadata))
        else: #or specific sub-function
            if specificKwarg == False:
                #get the parent info
                functionparent = functionname.split('.')[0]
                functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')
                #sub-select the looprange
                loopv = next((index for index in range(0,len(functionMetadata)) if list(functionMetadata.keys())[index] == functionname.split('.')[1]), None)
                looprange = range(loopv,loopv+1) #type:ignore
                finaltext = ""
            else:
                #Get information on a single kwarg
                #get the parent info
                functionparent = functionname.split('.')[0]
                #Get the full function metadata
                functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')
                #Get the help string of a single kwarg
                
                #Find the help text of a single kwarg
                helptext = 'No help text set'
                #Look over optional kwargs
                for k in range(0,len(functionMetadata[functionname.split('.')[1]]["optional_kwargs"])):
                    if functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]['name'] == specificKwarg:
                        helptext = functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]['description']
                #look over required kwargs
                for k in range(0,len(functionMetadata[functionname.split('.')[1]]["required_kwargs"])):
                    if functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]['name'] == specificKwarg:
                        helptext = functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]['description']
                # for distribution kwarg
                if specificKwarg == 'dist_kwarg':
                    helptext = functionMetadata[functionname.split('.')[1]]["dist_kwarg"]['description']
                finaltext = helptext
                skipfinalline = True
                looprange = range(0,0)
        name_arr = []
        help_arr = []
        rkwarr_arr = []
        okwarr_arr = []
        loopindex = 0
        for i in looprange:
            #Get name text for all entries
            name_arr.append([list(functionMetadata.keys())[i]])
            #Get help text for all entries
            help_arr.append(functionMetadata[list(functionMetadata.keys())[i]]["help_string"])
            #Get text for all the required kwarrs
            txt = ""
            #Loop over the number or rkwarrs
            for k in range(0,len(functionMetadata[list(functionMetadata.keys())[i]]["required_kwargs"])):
                zz = functionMetadata[list(functionMetadata.keys())[i]]["required_kwargs"][k]
                for key, value in functionMetadata[list(functionMetadata.keys())[i]]["required_kwargs"][k].items():
                    txt += f"{key}: {value}\n"
            rkwarr_arr.append(txt)
            #Get text for all the optional kwarrs
            txt = ""
            #Loop over the number of okwarrs
            for k in range(0,len(functionMetadata[list(functionMetadata.keys())[i]]["optional_kwargs"])):
                zz = functionMetadata[list(functionMetadata.keys())[i]]["optional_kwargs"][k]
                for key, value in functionMetadata[list(functionMetadata.keys())[i]]["optional_kwargs"][k].items():
                    txt += f"{key}: {value}\n"
            okwarr_arr.append(txt)
        
            #Fuse all the texts together
            if showHelp or showKwargs:
                finaltext += f"""
                -------------------------------------------
                {name_arr[loopindex][0]} information:
                -------------------------------------------"""
            if showHelp:
                finaltext += f"""
                {help_arr[loopindex]}"""
            if showKwargs:
                finaltext += f"""
                ----------
                Required keyword arguments (kwargs):
                {rkwarr_arr[loopindex]}----------
                Optional keyword arguments (kwargs):
                {okwarr_arr[loopindex]}"""
            finaltext += "\n"
            loopindex+=1
        
        if not skipfinalline:
            finaltext += "--------------------------------------------------------------------------------------\n"
        #Remove left-leading spaces
        finaltext = "\n".join(line.lstrip() for line in finaltext.splitlines())

        return finaltext
    #Error handling if __function_metadata__ doesn't exist
    except AttributeError:
        return f"No __function_metadata__ in {functionname}"

#Run a function with unknown number of parameters via the eval() method
#Please note that the arg values need to be the string variants of the variable, not the variable itself!
def createFunctionWithArgs(functionname,*args):
    #Start string with functionname.functionname - probably changing later for safety/proper usages
    fullstring = functionname+"."+functionname+"("
    #Add all arguments to the function
    idloop = 0
    for arg in args:
        if idloop>0:
            fullstring = fullstring+","
        fullstring = fullstring+str(arg)
        idloop+=1
    #Finish the function string
    fullstring = fullstring+")"
    #run the function
    return fullstring

#Run a function with unknown number of kwargs via the eval() method
#Please note that the kwarg values need to be the string variants of the variable, not the variable itself!
def createFunctionWithKwargs(functionname,**kwargs):
    #Start string with functionname.functionname - probably changing later for safety/proper usages
    fullstring = functionname+"("
    #Add all arguments to the function
    idloop = 0
    for key, value in kwargs.items():
        if idloop>0:
            fullstring = fullstring+","
        fullstring = fullstring+str(key)+"="+str(value)
        idloop+=1
    #Finish the function string
    fullstring = fullstring+")"
    #run the function
    return fullstring

def defaultValueFromKwarg(functionname,kwargname):
    #Check if the function has a 'default' entry for the specific kwarg. If not, return None. Otherwise, return the default value.
    
    defaultEntry=None
    functionparent = functionname.split('.')[0]
    #Get the full function metadata
    functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')
    for k in range(0,len(functionMetadata[functionname.split('.')[1]]["optional_kwargs"])):
        if functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]['name'] == kwargname:
            #check if this has a default value:
            if 'default' in functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]:
                defaultEntry = functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]['default']
    #look over required kwargs
    for k in range(0,len(functionMetadata[functionname.split('.')[1]]["required_kwargs"])):
        if functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]['name'] == kwargname:
            #check if this has a default value:
            if 'default' in functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]:
                defaultEntry = functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]['default']
    
    return defaultEntry


def displayNamesFromFunctionNames(functionName, polval):
    displaynames = []
    functionName_to_displayName_map = []
    for function in functionName:
        #Extract the mother function name - before the period:
        subroutineName = function.split('.')[0]
        singlefunctiondata = function.split('.')[1]
        #Check if the subroutine has a display name - if so, use that, otherwise use the subroutineName
        functionMetadata = eval(f'{str(subroutineName)}.__function_metadata__()')
        if 'display_name' in functionMetadata[singlefunctiondata]:
            displayName = functionMetadata[singlefunctiondata]['display_name']+" ("+polval+")"
        else:
            displayName = subroutineName+': '+singlefunctiondata+" ("+polval+")"
        displaynames.append(displayName)
        functionName_to_displayName_map.append((displayName,function))
    #Check for ambiguity in both columns:
    
    # if not len(np.unique(list(set(functionName_to_displayName_map)))) == len(list(itertools.chain.from_iterable(functionName_to_displayName_map))):
    #     raise Exception('Ambiguous display names in functions!! Please check all function names and display names for uniqueness!')
        
    return displaynames, functionName_to_displayName_map

def polaritySelectedFromDisplayName(displayname):
    if '(pos)' in displayname:
        return 'pos'
    elif '(neg)' in displayname:
        return 'neg'
    elif '(mix)' in displayname:
        return 'mix'


def functionNameFromDisplayName(displayname,map):
    for pair in map:
        if pair[0] == displayname:
            return pair[1]
        
def typeFromKwarg(functionname,kwargname):
    #Check if the function has a 'type' entry for the specific kwarg. If not, return None. Otherwise, return the type value.
    typing=None
    functionparent = functionname.split('.')[0]
    #Get the full function metadata
    functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')
    for k in range(0,len(functionMetadata[functionname.split('.')[1]]["optional_kwargs"])):
        if functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]['name'] == kwargname:
            #check if this has a default value:
            if 'type' in functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]:
                typing = functionMetadata[functionname.split('.')[1]]["optional_kwargs"][k]['type']
    #look over required kwargs
    for k in range(0,len(functionMetadata[functionname.split('.')[1]]["required_kwargs"])):
        if functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]['name'] == kwargname:
            #check if this has a default value:
            if 'type' in functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]:
                typing = functionMetadata[functionname.split('.')[1]]["required_kwargs"][k]['type']
    
    return typing




def changeLayout_choice(curr_layout,className,displayNameToFunctionNameMap,parent=None):
    logging.debug('Changing layout '+curr_layout.parent().objectName())
    #This removes everything except the first entry (i.e. the drop-down menu)
    resetLayout(curr_layout,className)
    #Get the dropdown info
    curr_dropdown = getMethodDropdownInfo(curr_layout,className)
    if len(curr_dropdown) == 0:
        pass
    #Get the kw-arguments from the current dropdown.
    current_selected_function = functionNameFromDisplayName(curr_dropdown.currentText(),displayNameToFunctionNameMap)
    logging.debug('current selected function: '+current_selected_function)
    current_selected_polarity = polaritySelectedFromDisplayName(curr_dropdown.currentText())
    
    #Classname should always end in pos/neg/mix!
    wantedPolarity = className[-3:].lower()
    
    #Hide dropdown entries that are not part of the current_selected property
    model = curr_dropdown.model()
    totalNrRows = model.rowCount()
    for rowId in range(totalNrRows):
        #First show all rows:
        curr_dropdown.view().setRowHidden(rowId, False)
        item = model.item(rowId)
        item.setFlags(item.flags() | Qt.ItemIsEnabled)
        
        #Then hide based on the row name
        rowName = model.item(rowId,0).text()
        if polaritySelectedFromDisplayName(rowName) != wantedPolarity:
            item = model.item(rowId)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            curr_dropdown.view().setRowHidden(rowId, True)
    
    #Visual max number of rows before a 2nd column is started.
    maxNrRows = 4
    labelposoffset = 0

    #Add a widget-pair for the distribution
    distKwargValues = distKwargValuesFromFittingFunction(current_selected_function)
    if len(distKwargValues) != 0:
        # Add a combobox containing all the possible kw-args
        label = QLabel("<b>distribution</b>")
        label.setObjectName(f"Label#{current_selected_function}#dist_kwarg#{current_selected_polarity}")
        if checkAndShowWidget(curr_layout,label.objectName()) == False:
            label.setToolTip(infoFromMetadata(current_selected_function,specificKwarg='dist_kwarg'))
            curr_layout.addWidget(label,2,0)
        combobox = QComboBox()
        combobox.addItems(distKwargValues)
        combobox.setObjectName(f"ComboBox#{current_selected_function}#dist_kwarg#{current_selected_polarity}")
        defaultOption = defaultOptionFromDistKwarg(current_selected_function)
        if defaultOption != None:
            combobox.setCurrentText(defaultOption)
        test = combobox.currentText()
        combobox.setToolTip(getInfoFromDistribution(combobox.currentText()))
        combobox.currentTextChanged.connect(lambda text: combobox.setToolTip(getInfoFromDistribution(text)))
        if checkAndShowWidget(curr_layout,combobox.objectName()) == False:
            curr_layout.addWidget(combobox,2,1)
        labelposoffset += 1
        
    reqKwargs = reqKwargsFromFunction(current_selected_function)
    #Add a widget-pair for every kw-arg
    
    for k in range(len(reqKwargs)):
        #Value is used for scoring, and takes the output of the method
        if reqKwargs[k] != 'methodValue':
            label = QLabel(f"<b>{reqKwargs[k]}</b>")
            label.setObjectName(f"Label#{current_selected_function}#{reqKwargs[k]}#{current_selected_polarity}")
            if checkAndShowWidget(curr_layout,label.objectName()) == False:
                label.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=reqKwargs[k]))
                curr_layout.addWidget(label,2+((k+labelposoffset))%maxNrRows,(((k+labelposoffset))//maxNrRows)*2+0)
            #Check if we want to add a fileLoc-input:
            if typeFromKwarg(current_selected_function,reqKwargs[k]) == 'fileLoc':
                #Create a new qhboxlayout:
                hor_boxLayout = QHBoxLayout()
                #Add a line_edit to this:
                line_edit = QLineEdit()
                line_edit.setObjectName(f"LineEdit#{current_selected_function}#{reqKwargs[k]}#{current_selected_polarity}")
                defaultValue = defaultValueFromKwarg(current_selected_function,reqKwargs[k])
                hor_boxLayout.addWidget(line_edit)
                #Also add a QButton with ...:
                line_edit_lookup = QPushButton()
                line_edit_lookup.setText('...')
                line_edit_lookup.setObjectName(f"PushButton#{current_selected_function}#{reqKwargs[k]}#{current_selected_polarity}")
                hor_boxLayout.addWidget(line_edit_lookup)
                
                #Actually placing it in the layout
                checkAndShowWidget(curr_layout,line_edit.objectName())
                checkAndShowWidget(curr_layout,line_edit_lookup.objectName())
                if checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                    line_edit.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=reqKwargs[k]))
                    if defaultValue is not None:
                        line_edit.setText(str(defaultValue))
                    curr_layout.addLayout(hor_boxLayout,2+((k+labelposoffset))%maxNrRows,(((k+labelposoffset))//maxNrRows)*2+1)
                    #Add a on-change listener:
                    line_edit.textChanged.connect(lambda text,line_edit=line_edit: kwargValueInputChanged(line_edit))
                    
                    #Add an listener when the pushButton is pressed
                    line_edit_lookup.clicked.connect(lambda text2,line_edit_change_objName = line_edit,text="Select file",filter="*.*": lineEditFileLookup(line_edit_change_objName, text, filter,parent=parent))
                    
            else: #'normal' type - int, float, string, whatever
                #Creating a line-edit...
                line_edit = QLineEdit()
                line_edit.setObjectName(f"LineEdit#{current_selected_function}#{reqKwargs[k]}#{current_selected_polarity}")
                defaultValue = defaultValueFromKwarg(current_selected_function,reqKwargs[k])
                #Actually placing it in the layout
                if checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                    line_edit.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=reqKwargs[k]))
                    if defaultValue is not None:
                        line_edit.setText(str(defaultValue))
                    curr_layout.addWidget(line_edit,2+((k+labelposoffset))%maxNrRows,(((k+labelposoffset))//maxNrRows)*2+1)
                    #Add a on-change listener:
                    line_edit.textChanged.connect(lambda text,line_edit=line_edit: kwargValueInputChanged(line_edit))
        else:
            labelposoffset -= 1
        
    #Get the optional kw-arguments from the current dropdown.
    optKwargs = optKwargsFromFunction(current_selected_function)
    #Add a widget-pair for every kwarg
    for k in range(len(optKwargs)):
        label = QLabel(f"<i>{optKwargs[k]}</i>")
        label.setObjectName(f"Label#{current_selected_function}#{optKwargs[k]}#{current_selected_polarity}")
        if checkAndShowWidget(curr_layout,label.objectName()) == False:
            label.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=optKwargs[k]))
            curr_layout.addWidget(label,2+((k+labelposoffset+len(reqKwargs)))%maxNrRows,(((k+labelposoffset+len(reqKwargs)))//maxNrRows)*2+0)
        #Check if we want to add a fileLoc-input:
        if typeFromKwarg(current_selected_function,optKwargs[k]) == 'fileLoc':
            #Create a new qhboxlayout:
            hor_boxLayout = QHBoxLayout()
            #Add a line_edit to this:
            line_edit = QLineEdit()
            line_edit.setObjectName(f"LineEdit#{current_selected_function}#{optKwargs[k]}#{current_selected_polarity}")
            defaultValue = defaultValueFromKwarg(current_selected_function,optKwargs[k])
            hor_boxLayout.addWidget(line_edit)
            #Also add a QButton with ...:
            line_edit_lookup = QPushButton()
            line_edit_lookup.setText('...')
            line_edit_lookup.setObjectName(f"PushButton#{current_selected_function}#{optKwargs[k]}#{current_selected_polarity}")
            hor_boxLayout.addWidget(line_edit_lookup)
            
            #Actually placing it in the layout
            checkAndShowWidget(curr_layout,line_edit.objectName())
            checkAndShowWidget(curr_layout,line_edit_lookup.objectName())
            if checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                line_edit.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=optKwargs[k]))
                if defaultValue is not None:
                    line_edit.setText(str(defaultValue))
                curr_layout.addLayout(hor_boxLayout,2+((k+labelposoffset))%maxNrRows,(((k+labelposoffset))//maxNrRows)*2+1)
                #Add a on-change listener:
                line_edit.textChanged.connect(lambda text,line_edit=line_edit: kwargValueInputChanged(line_edit))
                
                #Add an listener when the pushButton is pressed
                line_edit_lookup.clicked.connect(lambda text2,line_edit_change_objName = line_edit,text="Select file",filter="*.*": lineEditFileLookup(line_edit_change_objName, text, filter,parent=parent))
                    
        else:
            line_edit = QLineEdit()
            line_edit.setObjectName(f"LineEdit#{current_selected_function}#{optKwargs[k]}#{current_selected_polarity}")
            defaultValue = defaultValueFromKwarg(current_selected_function,optKwargs[k])
            if checkAndShowWidget(curr_layout,line_edit.objectName()) == False:
                line_edit.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=optKwargs[k]))
                if defaultValue is not None:
                    line_edit.setText(str(defaultValue))
                curr_layout.addWidget(line_edit,2+((k+labelposoffset+len(reqKwargs)))%maxNrRows,(((k+labelposoffset+len(reqKwargs)))//maxNrRows)*2+1)
                #Add a on-change listener:
                line_edit.textChanged.connect(lambda text,line_edit=line_edit: kwargValueInputChanged(line_edit))

def kwargValueInputChanged(line_edit):
    #Get the function name
    function = line_edit.objectName().split("#")[1]
    #Get the kwarg
    kwarg = line_edit.objectName().split("#")[2]
    #Get the value
    value = line_edit.text()
    expectedType = typeFromKwarg(function,kwarg)
    if expectedType == 'fileLoc':
        expectedType=str
    if expectedType is not None:
        if expectedType is str:
            try:
                value = str(line_edit.text())
                setLineEditStyle(line_edit,type='Normal')
            except:
                #Show as warning
                setLineEditStyle(line_edit,type='Warning')
        elif expectedType is not str:
            try:
                value = eval(line_edit.text())
                if expectedType == float:
                    if isinstance(value,int) or isinstance(value,float):
                        setLineEditStyle(line_edit,type='Normal')
                    else:
                        setLineEditStyle(line_edit,type='Warning')
                else:
                    if isinstance(value,expectedType):
                        setLineEditStyle(line_edit,type='Normal')
                    else:
                        setLineEditStyle(line_edit,type='Warning')
            except:
                #Show as warning
                setLineEditStyle(line_edit,type='Warning')
    else:
        setLineEditStyle(line_edit,type='Normal')
    pass

def setLineEditStyle(line_edit,type='Normal'):
    if type == 'Normal':
        line_edit.setStyleSheet("border: 1px  solid #D5D5E5;")
    elif type == 'Warning':
        line_edit.setStyleSheet("border: 1px solid red;")

def checkAndShowWidget(layout, widgetName):
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
        else:
            for index2 in range(item.count()):
                item_sub = item.itemAt(index2)
                # Check if the item is a widget
                if item_sub.widget() is not None:
                    widget = item_sub.widget()
                    # Check if the widget has the desired name
                    if widget.objectName() == widgetName:
                        # Widget already exists, unhide it
                        widget.show()
                        return
    return False

#Remove everythign in this layout except className_dropdown
def resetLayout(curr_layout,className):
    for index in range(curr_layout.count()):
        widget_item = curr_layout.itemAt(index)
        # Check if the item is a widget (as opposed to a layout)
        if widget_item.widget() is not None:
            widget = widget_item.widget()
            #If it's the dropdown segment, label it as such
            if not ("CandidateFindingDropdown" in widget.objectName()) and not ("CandidateFittingDropdown" in widget.objectName()) and widget.objectName() != f"titleLabel_{className}" and not ("KEEP" in widget.objectName()):
                logging.debug(f"Hiding {widget.objectName()}")
                widget.hide()
        else:
            for index2 in range(widget_item.count()):
                widget_sub_item = widget_item.itemAt(index2)
                # Check if the item is a widget (as opposed to a layout)
                if widget_sub_item.widget() is not None:
                    widget = widget_sub_item.widget()
                    #If it's the dropdown segment, label it as such
                    if not ("CandidateFindingDropdown" in widget.objectName()) and not ("CandidateFittingDropdown" in widget.objectName()) and widget.objectName() != f"titleLabel_{className}" and not ("KEEP" in widget.objectName()):
                        logging.debug(f"Hiding {widget.objectName()}")
                        widget.hide()

def getMethodDropdownInfo(curr_layout,className):
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


def lineEditFileLookup(line_edit_objName, text, filter,parent=None):
    file_path = generalFileSearchButtonAction(parent=parent,text=text,filter=filter)
    line_edit_objName.setText(file_path)
        
def generalFileSearchButtonAction(parent=None,text='Select File',filter='*.txt'):
    file_path, _ = QFileDialog.getOpenFileName(parent,text,filter=filter)
    return file_path
    