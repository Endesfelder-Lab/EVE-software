import os
import warnings
import inspect
import importlib
import re
import warnings, logging
import numpy as np
import itertools
import time
import h5py

try:
    #Import all scripts in the custom script folders
    # List all files in the CandidateFitting directory
    from eve_smlm.CandidateFinding import *
    from eve_smlm.CandidateFitting import *
    from eve_smlm.TemporalFitting import timeFitting
    from eve_smlm.Visualisation import *
    from eve_smlm.PostProcessing import *
    from eve_smlm.CandidatePreview import *
    from eve_smlm.DataAnalysis import *
    from eve_smlm.Utils import *
    from eve_smlm.EventDistributions import eventDistributions

except ImportError:
    #Import all scripts in the custom script folders
    # List all files in the CandidateFitting directory
    from CandidateFinding import *
    from CandidateFitting import *
    from TemporalFitting import timeFitting
    from Visualisation import *
    from PostProcessing import *
    from CandidatePreview import *
    from DataAnalysis import *
    from Utils import *
    from EventDistributions import eventDistributions


#Imports for PyQt5 (GUI)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor, QTextCursor, QIntValidator
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QLayout, QMainWindow, QLabel, QPushButton, QSizePolicy, QGroupBox, QTabWidget, QGridLayout, QWidget, QComboBox, QLineEdit, QFileDialog, QToolBar, QCheckBox,QDesktopWidget, QMessageBox, QTextEdit, QSlider, QSpacerItem
from PyQt5.QtCore import Qt, QPoint, QProcess, QCoreApplication, QTimer, QFileSystemWatcher, QFile, QThread, pyqtSignal, QObject

from PyQt5.QtCore import QUrl
import markdown
from PyQt5.QtWebEngineWidgets import QWebEngineView
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
    """ 
    Return all functions that are found in a specific directory which have the correct function metadata
    """
    #initialise empty array
    functionnamearr = []
    def addFilesToAbsolutePath(functionnamearr,absolute_path):
        #Loop over all files
        for file in os.listdir(absolute_path):
            #Check if they're .py files
            if file.endswith(".py"):
                #Check that they're not init files or similar
                if not file.startswith("_") and not file == "utils.py" and not file == "utilsHelper.py":
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
        return functionnamearr
    
    #Get the absolute path, assuming that this file will stay in the sister-folder
    absolute_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),dirname)
    functionnamearr = addFilesToAbsolutePath(functionnamearr,absolute_path)
    
    #Also do this on the app-data folder
    #Try-except clause just if the folder isn't in appdata it shouldn't be an issue
    try:
        additional_folder_name = os.path.join("C:\\Users\\Koen Martens\\AppData\\Local\\UniBonn\\Eve",dirname)
        functionnamearr = addFilesToAbsolutePath(functionnamearr,additional_folder_name)
    except:
        pass
    
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

#Returns a display name (if available) of an individual kwarg name, from a specific function:
def displayNameFromKwarg(functionname,name):
    #Get all kwarg info
    allkwarginfo = kwargsFromFunction(functionname)
    displayName = name
    #Look through optional args first, then req. kwargs (so that req. kwargs have priority in case something weirdi s happening):
    for optOrReq in range(1,-1,-1):
    
        #Perform a regex match on 'name'
        name_pattern = r"name:\s*(\S+)"
        
        names = re.findall(name_pattern, allkwarginfo[optOrReq][0])
        instances = re.split(r'(?=name: )', allkwarginfo[optOrReq][0])[1:]

        #Find which instance this name belongs to:
        name_id = -1
        for i,namef in enumerate(names):
            if namef == name:
                name_id = i
        
        if name_id > -1:
            curr_instance = instances[name_id]
            displayText_pattern = r"display_text: (.*?)\n"
            displaytext = re.findall(displayText_pattern, curr_instance)
            if len(displaytext) > 0:
                displayName = displaytext[0]
            else:
                displayName = name
    
    return displayName
    

#Returns the 'names' of the optional kwargs of a function
def optKwargsFromFunction(functionname):
    #Get all kwarg info
    allkwarginfo = kwargsFromFunction(functionname)
    #Perform a regex match on 'name'
    name_pattern = r"name:\s*(\S+)"
    #Get the names of the optional kwargs (allkwarginfo[1])
    names = re.findall(name_pattern, allkwarginfo[1][0])
    return names

def classKwargValuesFromFittingFunction(functionname, class_type):
    #Get all kwarg info
    allkwarginfo = kwargsFromFunction(functionname)
    derivedClasses = []
    derivedClasses_display_name=[]
    derivedClasses_description=[]
    if allkwarginfo[2] != [] and class_type=="dist":
        base_pattern = r"base:\s*(\S+)"
        base_name = re.findall(base_pattern, allkwarginfo[2][0])[0]
        # get base class
        baseClass = getattr(eventDistributions, base_name, None)
        if not baseClass == None:
            # get all derived classes that share common base
            for name, obj in inspect.getmembers(eventDistributions):
                if inspect.isclass(obj) and issubclass(obj, baseClass) and obj != baseClass:
                    derivedClasses.append(name)
                    try:
                        derivedClasses_display_name.append(obj.display_name)
                    except:
                        derivedClasses_display_name.append(name)
                    try: 
                        derivedClasses_description.append(obj.description)
                    except:
                        derivedClasses_description.append("")
    elif allkwarginfo[3] != [] and class_type=="time":
        base_pattern = r"base:\s*(\S+)"
        base_name_matches = re.findall(base_pattern, allkwarginfo[3][0])
        if len(base_name_matches) > 0:
            base_name = base_name_matches[0]
            # get base class
            baseClass = getattr(timeFitting, base_name, None)
            if not baseClass == None:
                # get all derived classes that share common base
                for name, obj in inspect.getmembers(timeFitting):
                    if inspect.isclass(obj) and issubclass(obj, baseClass) and obj != baseClass:
                        derivedClasses.append(name)
                        try:
                            derivedClasses_display_name.append(obj.display_name)
                        except:
                            derivedClasses_display_name.append(name)
                        try:
                            derivedClasses_description.append(obj.description)
                        except:
                            derivedClasses_description.append("")
        else:
             logging.debug("No base pattern found for time fitting kwarg.")
                    
    name_to_displayName_map = dict(zip(derivedClasses, derivedClasses_display_name))
    
    return [derivedClasses, derivedClasses_display_name, name_to_displayName_map, derivedClasses_description]

def defaultOptionFromClassKwarg(functionname, classtype):
    #Check if the function has a 'default' option for the distribution kwarg. 
    defaultOption=None
    functionparent = functionname.split('.')[0]
    #Get the full function metadata
    functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')[functionname.split('.')[1]]
    if "dist_kwarg" in functionMetadata and classtype == "dist":
        if "default_option" in functionMetadata["dist_kwarg"]:
            defaultOption = functionMetadata["dist_kwarg"]["default_option"]
            # check if defaultOption is a valid option
            defaultOption = getattr(eventDistributions, defaultOption, None)
            if defaultOption != None:
                defaultOption = defaultOption.__name__
    if "time_kwarg" in functionMetadata and classtype == "time":
        if "default_option" in functionMetadata["time_kwarg"]:
            defaultOption = functionMetadata["time_kwarg"]["default_option"]
            # check if defaultOption is a valid option
            defaultOption = getattr(timeFitting, defaultOption, None)
            if defaultOption != None:
                defaultOption = defaultOption.__name__
    return defaultOption

def getInfoFromClass(class_name, class_type):
    description = None
    display_name = None
    if class_type == "dist":
        distClass = getattr(eventDistributions, class_name, None)
        if not distClass == None:
            try:
                description = distClass.description
                display_name = distClass.display_name
            except AttributeError:
                pass
    elif class_type == "time":
        timeClass = getattr(timeFitting, class_name, None)
        if not timeClass == None:
            try:
                description = timeClass.description
                display_name = timeClass.display_name
            except AttributeError:
                pass
    return description, display_name

def helpStringFromFunction(functionname):
    try:
        #Check if parent function
        if not '.' in functionname:
            functionMetadata = eval(f'{str(functionname)}.__function_metadata__()')
            return functionMetadata['help_string']
        else: #or specific sub-function
            #get the parent info
            functionparent = functionname.split('.')[0]
            functionMetadata = eval(f'{str(functionparent)}.__function_metadata__()')
            return functionMetadata[functionname.split('.')[1]]['help_string']
    except:
        return ""

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
        time_kwarg = []
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
            #Get text for time fitting kwargs
            txt = ""
            if "time_kwarg" in functionMetadata[list(functionMetadata.keys())[i]]:
                for key, value in functionMetadata[list(functionMetadata.keys())[i]]["time_kwarg"].items():
                    txt += f"{key}: {value}\n"
                time_kwarg.append(txt)
    #Error handling if __function_metadata__ doesn't exist
    except AttributeError:
        rkwarr_arr = []
        okwarr_arr = []
        dist_kwarg = []
        time_kwarg = []
        return f"No __function_metadata__ in {functionname}"
            
    return [rkwarr_arr, okwarr_arr, dist_kwarg, time_kwarg]

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
                # for time fitting kwarg
                if specificKwarg == 'time_kwarg':
                    helptext = functionMetadata[functionname.split('.')[1]]["time_kwarg"]['description']
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

def changeTab(parent,text="Processing"):
    """
    Change the tab in the mainTabWidget to the one with the specified text.

    Args:
        parent: The parent widget containing the mainTabWidget.
        text (str): The text of the tab to change to. Defaults to "Processing".
    """
    for i in range(parent.mainTabWidget.count()):
        if parent.mainTabWidget.tabText(i) == text:
            parent.mainTabWidget.setCurrentIndex(i)
            break
    import time
    time.sleep(0.1)
        
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


def displayNamesFromFunctionNames(functionName, polval=''):
    displaynames = []
    functionName_to_displayName_map = []
    for function in functionName:
        #Extract the mother function name - before the period:
        subroutineName = function.split('.')[0]
        singlefunctiondata = function.split('.')[1]
        #Check if the subroutine has a display name - if so, use that, otherwise use the subroutineName
        functionMetadata = eval(f'{str(subroutineName)}.__function_metadata__()')
        if 'display_name' in functionMetadata[singlefunctiondata]:
            displayName = functionMetadata[singlefunctiondata]['display_name']
            #Add the polarity info between brackets if required
            if polval != '':
                displayName += " ("+polval+")"
        else:
            displayName = subroutineName+': '+singlefunctiondata
            #Add the polarity info between brackets if required
            if polval != '':
                displayName += " ("+polval+")"
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


def functionNameFromDisplayName(displayname,map,typev='Normal'):
    if typev == 'Normal':
        for pair in map:
            if pair[0] == displayname:
                return pair[1]
    elif typev == 'distOrTime':
        for name in map:
            if map[name] == displayname:
                return name
        
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




def changeLayout_choice(curr_layout,className,displayNameToFunctionNameMap,parent=None,ignorePolarity=False,maxNrRows=4):
    logging.debug('Changing layout '+curr_layout.parent().objectName())
    #This removes everything except the first entry (i.e. the drop-down menu)
    resetLayout(curr_layout,className)
    #Get the dropdown info
    curr_dropdown = getMethodDropdownInfo(curr_layout,className)
    if len(curr_dropdown) > 0:
        #Get the kw-arguments from the current dropdown.
        current_selected_function = functionNameFromDisplayName(curr_dropdown.currentText(),displayNameToFunctionNameMap)
        logging.debug('current selected function: '+current_selected_function)
        if not ignorePolarity:
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
        else:
            #Unhide everything
            model = curr_dropdown.model()
            totalNrRows = model.rowCount()
            for rowId in range(totalNrRows):
                #First show all rows:
                curr_dropdown.view().setRowHidden(rowId, False)
                item = model.item(rowId)
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
            current_selected_polarity = 'None'
    
        #Visual max number of rows before a 2nd column is started.
        labelposoffset = 0


        #Find the time fit distributions
        [timeFitValues, timeFit_displayNames, timeFit_name_to_displayName_map, timeFit_descriptions] = classKwargValuesFromFittingFunction(current_selected_function, 'time')
        if len(timeFitValues) != 0:
            parent.timeFit_name_to_displayName_map = timeFit_name_to_displayName_map
            parent.timeFit_descriptions = timeFit_descriptions
            # Add a combobox containing all the possible kw-args
            label = QLabel("<b>Time fit routine</b>")
            label.setObjectName(f"Label#{current_selected_function}#time_kwarg#{current_selected_polarity}")
            if checkAndShowWidget(curr_layout,label.objectName()) == False:
                label.setToolTip(infoFromMetadata(current_selected_function,specificKwarg='time_kwarg'))
                curr_layout.addWidget(label,labelposoffset+2,0)
            combobox = QComboBox()
            combobox.addItems(timeFit_displayNames)
            combobox.setObjectName(f"ComboBox#{current_selected_function}#time_kwarg#{current_selected_polarity}")
            
            defaultOption = defaultOptionFromClassKwarg(current_selected_function, 'time')
            if defaultOption != None:
                defaultOption_displayName = timeFit_name_to_displayName_map[defaultOption]
                combobox.setCurrentText(defaultOption_displayName)
            #update tooltip
            updateTimeFitTooltip(combobox,parent)
            combobox.currentTextChanged.connect(lambda: updateTimeFitTooltip(combobox,parent))
            if checkAndShowWidget(curr_layout,combobox.objectName()) == False:
                curr_layout.addWidget(combobox,labelposoffset+2,1)
            labelposoffset += 1

        #Add a widget-pair for the distribution
        [distKwargValues, distKwarg_displayNames, distKwarg_name_to_displayName_map, distKwarg_descriptions] = classKwargValuesFromFittingFunction(current_selected_function, 'dist')
        if len(distKwargValues) != 0:
            parent.distKwarg_name_to_displayName_map = distKwarg_name_to_displayName_map
            parent.distKwarg_descriptions = distKwarg_descriptions
            # Add a combobox containing all the possible kw-args
            label = QLabel("<b>distribution</b>")
            label.setObjectName(f"Label#{current_selected_function}#dist_kwarg#{current_selected_polarity}")
            if checkAndShowWidget(curr_layout,label.objectName()) == False:
                label.setToolTip(infoFromMetadata(current_selected_function,specificKwarg='dist_kwarg'))
                curr_layout.addWidget(label,labelposoffset+2,0)
            combobox = QComboBox()
            combobox.addItems(distKwarg_displayNames)
            combobox.setObjectName(f"ComboBox#{current_selected_function}#dist_kwarg#{current_selected_polarity}")
            
            defaultOption = defaultOptionFromClassKwarg(current_selected_function, 'dist')
            if defaultOption != None:
                defaultOption_displayName = distKwarg_name_to_displayName_map[defaultOption]
                combobox.setCurrentText(defaultOption_displayName)
            #update tooltip
            updateDistKwargTooltip(combobox,parent)
            combobox.currentTextChanged.connect(lambda: updateDistKwargTooltip(combobox,parent))
            if checkAndShowWidget(curr_layout,combobox.objectName()) == False:
                curr_layout.addWidget(combobox,labelposoffset+2,1)
            labelposoffset += 1
            
        reqKwargs = reqKwargsFromFunction(current_selected_function)
        
        #Add a widget-pair for every kw-arg
        
        for k in range(len(reqKwargs)):
            #Value is used for scoring, and takes the output of the method
            if reqKwargs[k] != 'methodValue':
                label = QLabel(f"<b>{displayNameFromKwarg(current_selected_function,reqKwargs[k])}</b>")
                label.setObjectName(f"Label#{current_selected_function}#{reqKwargs[k]}#{current_selected_polarity}")
                if checkAndShowWidget(curr_layout,label.objectName()) == False:
                    label.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=reqKwargs[k]))
                    curr_layout.addWidget(label,2+((k+labelposoffset))%maxNrRows,(((k+labelposoffset))//maxNrRows)*2+0)
                #Check if we want to add a fileLoc-input:
                if typeFromKwarg(current_selected_function,reqKwargs[k]) == 'fileLoc' or typeFromKwarg(current_selected_function,reqKwargs[k]) == 'fileLocSave':
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
                        
                        if typeFromKwarg(current_selected_function,reqKwargs[k]) == 'fileLocSave':
                            #Add an listener when the pushButton is pressed
                            line_edit_lookup.clicked.connect(lambda text2,line_edit_change_objName = line_edit,text="Select file",filter="*.*": lineEditFileSaveLookup(line_edit_change_objName, text, filter,parent=parent))
                        else:
                            #Add an listener when the pushButton is pressed
                            line_edit_lookup.clicked.connect(lambda text2,line_edit_change_objName = line_edit,text="Select file",filter="*.*": lineEditFileLookup(line_edit_change_objName, text, filter,parent=parent))
                elif type(typeFromKwarg(current_selected_function,reqKwargs[k])) == str and typeFromKwarg(current_selected_function,reqKwargs[k])[0:9] == 'dropDown(':
                    centerText = typeFromKwarg(current_selected_function,reqKwargs[k])[9:-1]
                    if centerText[0:2] == '__' and centerText[-2:] == '__':
                        if centerText == '__locListHeaders__':
                            try:
                                dropDownOptions = parent.parent.data['FittingResult'][0].columns.tolist()
                            except:
                                dropDownOptions = ['Error in setting dropdown values (locListHeaders)']
                    else:
                        dropDownOptions = centerText.split(',')
                    
                    
                    #Delete old dropdown:
                    removeWidget(curr_layout,f"ComboBox#{current_selected_function}#{reqKwargs[k]}#{current_selected_polarity}")
                    #allow the layout to actually process a deletelater event:
                    QApplication.processEvents()
                    
                    #We want to add a dropdown!
                    #Create a new qhboxlayout:
                    hor_boxLayout = QHBoxLayout()
                    #Add a line_edit to this:
                    dropDown = QComboBox()
                    dropDown.setObjectName(f"ComboBox#{current_selected_function}#{reqKwargs[k]}#{current_selected_polarity}")
                    
                    
                    #Add the options to the dropdown:
                    for option in dropDownOptions:
                        dropDown.addItem(option)
                    defaultValue = defaultValueFromKwarg(current_selected_function,reqKwargs[k])
                    hor_boxLayout.addWidget(dropDown)
                    
                    #Actually placing it in the layout - this is different than other methods, in that it will be removed + re-added if it already exists.
                    checkAndShowWidget(curr_layout,dropDown.objectName())
                    if checkAndShowWidget(curr_layout,dropDown.objectName()) == False:
                        dropDown.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=reqKwargs[k]))
                        if defaultValue is not None:
                            index = dropDown.findText(str(defaultValue))
                            if index >= 0:
                                dropDown.setCurrentIndex(index)
                        curr_layout.addLayout(hor_boxLayout,2+((k+labelposoffset))%maxNrRows,(((k+labelposoffset))//maxNrRows)*2+1)
                        
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
            label = QLabel(f"<i>{displayNameFromKwarg(current_selected_function,optKwargs[k])}</i>")
            label.setObjectName(f"Label#{current_selected_function}#{optKwargs[k]}#{current_selected_polarity}")
            if checkAndShowWidget(curr_layout,label.objectName()) == False:
                label.setToolTip(infoFromMetadata(current_selected_function,specificKwarg=optKwargs[k]))
                curr_layout.addWidget(label,2+((k+labelposoffset+len(reqKwargs)))%maxNrRows,(((k+labelposoffset+len(reqKwargs)))//maxNrRows)*2+0)
            #Check if we want to add a fileLoc-input:
            if typeFromKwarg(current_selected_function,optKwargs[k]) == 'fileLoc' or typeFromKwarg(current_selected_function,optKwargs[k]) == 'fileLocSave':
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
                    curr_layout.addLayout(hor_boxLayout,2+((k+labelposoffset+len(reqKwargs)))%maxNrRows,(((k+labelposoffset+len(reqKwargs)))//maxNrRows)*2+1)
                    #Add a on-change listener:
                    line_edit.textChanged.connect(lambda text,line_edit=line_edit: kwargValueInputChanged(line_edit))
                    
                    if typeFromKwarg(current_selected_function,optKwargs[k]) == 'fileLocSave':
                        #Add an listener when the pushButton is pressed
                        line_edit_lookup.clicked.connect(lambda text2,line_edit_change_objName = line_edit,text="Select file",filter="*.*": lineEditFileSaveLookup(line_edit_change_objName, text, filter,parent=parent))
                    else:
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
    
    #Attempt a dropdown tooltip update:
    if hasattr(parent,className):
        if len(curr_dropdown) > 0:
            current_selected_function = functionNameFromDisplayName(curr_dropdown.currentText(),displayNameToFunctionNameMap)
            curr_dropdown.setToolTip(helpStringFromFunction(current_selected_function))

def updateTimeFitTooltip(line_edit,parent):
    """
    Updates the tooltip of a line_edit of a time fit routine
    """ 
    index = -1
    #Get the index of the current Text in the description list:
    for index, displayName in enumerate(parent.timeFit_name_to_displayName_map.values()):
        if displayName == line_edit.currentText():
            break
    
    #Set the tooltip if found
    if index > -1:
        line_edit.setToolTip(parent.timeFit_descriptions[index])

def updateDistKwargTooltip(line_edit,parent):
    """
    Updates the tooltip of a line_edit of a distribution kwargs routine
    """
    index = -1
    #Get the index of the current Text in the description list:
    for index, displayName in enumerate(parent.distKwarg_name_to_displayName_map.values()):
        if displayName == line_edit.currentText():
            break
    #Set the tooltip if found
    if index > -1:
        line_edit.setToolTip(parent.distKwarg_descriptions[index])

def kwargValueInputChanged(line_edit):
    #Get the function name
    function = line_edit.objectName().split("#")[1]
    #Get the kwarg
    kwarg = line_edit.objectName().split("#")[2]
    #Get the value
    value = line_edit.text()
    expectedType = typeFromKwarg(function,kwarg)
    if expectedType == 'fileLoc' or expectedType == 'fileLocSave':
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

def removeWidget(layout,widgetName):
    # Iterate over the layout's items
    for index in range(layout.count()):
        item = layout.itemAt(index)
        # Check if the item is a widget
        if item.widget() is not None:
            widget = item.widget()
            # Check if the widget has the desired name
            if widget.objectName() == widgetName:
                # Widget already exists, delete it
                widget.setParent(None)
                widget.setObjectName(None)
                widget.deleteLater()
                return
        else:
            for index2 in range(item.count()):
                item_sub = item.itemAt(index2)
                # Check if the item is a widget
                if item_sub.widget() is not None:
                    widget = item_sub.widget()
                    # Check if the widget has the desired name
                    if widget.objectName() == widgetName:
                        # Widget already exists, delete it
                        widget.setParent(None)
                        widget.setObjectName(None)
                        widget.deleteLater()
                        return

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
    parentFolder = line_edit_objName.text()
    if parentFolder != "":
        parentFolder = os.path.dirname(parentFolder)
    
    file_path = generalFileSearchButtonAction(parent=parent,text=text,filter=filter,parentFolder=parentFolder)
    line_edit_objName.setText(file_path)
        

def lineEditFileSaveLookup(line_edit_objName, text, filter,parent=None):
    parentFolder = line_edit_objName.text()
    if parentFolder != "":
        parentFolder = os.path.dirname(parentFolder)
    
    file_path = generalFileSaveButtonAction(parent=parent,text=text,filter=filter,parentFolder=parentFolder)
    line_edit_objName.setText(file_path)
        
def generalFileSearchButtonAction(parent=None,text='Select File',filter='*.txt',parentFolder=""):
    file_path, _ = QFileDialog.getOpenFileName(parent,text,parentFolder,filter=filter)
    return file_path

def generalFileSaveButtonAction(parent=None,text='Select File',filter='*.txt',parentFolder=""):
    file_path, _ = QFileDialog.getSaveFileName(parent,text,parentFolder,filter=filter)
    return file_path

def generalSaveButtonAction(parent=None,text='Select File',filter='*.txt',parentFolder=""):
    file_path, _ = QFileDialog.getOpenFileName(parent,text,parentFolder,filter=filter)
    return file_path


    
def getEvalTextFromGUIFunction(methodName, methodKwargNames, methodKwargValues, partialStringStart=None, removeKwargs=None):
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
        reqKwargs = reqKwargsFromFunction(methodName)
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
                #First find the index of the function-based reqKwargs in the GUI-based methodKwargNames. 
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
                    #First find the index of the function-based reqKwargs in the GUI-based methodKwargNames. 
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
                optKwargs = optKwargsFromFunction(methodName)
                for id in range(0,len(optKwargs)):
                    if methodKwargValues[id+len(reqKwargs)] != '':
                        if partialString != '':
                            partialString+=","
                        partialString+=optKwargs[id]+"=\""+methodKwargValues[methodKwargNames.index(optKwargs[id])]+"\""
                #Add the distribution kwarg if it exists
                if 'dist_kwarg' in methodKwargNames:
                    partialString += ",dist_kwarg=\""+methodKwargValues[methodKwargNames.index('dist_kwarg')]+"\""
                #Add the time fit if it exists
                if 'time_kwarg' in methodKwargNames:
                    partialString += ",time_kwarg=\""+methodKwargValues[methodKwargNames.index('time_kwarg')]+"\""
                segmentEval = methodName+"("+partialString+")"
                return segmentEval
            else:
                logging.error('NOT ALL KWARGS PROVIDED!')
                return None
        else:
            logging.error('SOMETHING VERY STUPID HAPPENED')
            return None
        


""" 
Functions to obtain events or filter events:
"""
def determineAllStartStopTimesHDF(dataLocation,timeChunkMs=10000,timeChunkOverlapMs=500,chunkStartStopTime=[0,np.inf]):
    """
    Determine all start/stop times of all chunks that are to be found in a HDF5 file
    """
    #Find the last time point from the hdf5 file:
    with h5py.File(dataLocation, mode='r') as file:
        #Events are here in this file:
        events_hdf5 = file['CD']['events']
        hdf5_maxtime = events_hdf5[events_hdf5.size-1]['t']/1000
    
    if chunkStartStopTime[1]-chunkStartStopTime[0] < timeChunkMs: #a very short time
        #We just need a single chunk:
        start_time_ms_arr = [np.maximum(chunkStartStopTime[0]-timeChunkOverlapMs,0)]
        end_time_ms_arr = [np.minimum(chunkStartStopTime[1]+timeChunkOverlapMs,hdf5_maxtime)]
    else: #We're looking at a long time
        start_time_ms_arr = np.maximum(np.arange(chunkStartStopTime[0],min(hdf5_maxtime,chunkStartStopTime[1]),timeChunkMs)-timeChunkOverlapMs,0)
        try:
            end_time_ms_arr = np.minimum(np.arange(chunkStartStopTime[0]+timeChunkMs,min(hdf5_maxtime,chunkStartStopTime[1])+timeChunkMs,timeChunkMs)+timeChunkOverlapMs,hdf5_maxtime)
        except:
            if chunkStartStopTime[1] == np.inf:
                end_time_ms_arr = np.array([np.inf])
            else:
                logging.error('Not sure how to do the chunking here...')
    
    return start_time_ms_arr,end_time_ms_arr

def findIndexFromTimeSliceHDF(dataLocation,requested_start_time_ms_arr = [0],requested_end_time_ms_arr=[1000],n_course_chunks = 5000):
    # Returns:
    #N-by-2 array of start/end indeces from hdf5 file:
    startEndIndecesHdf5File = np.zeros((len(requested_start_time_ms_arr),2))
    
    #Check if start/end time arrays are same size:
    if len(requested_start_time_ms_arr) != len(requested_end_time_ms_arr):
        logging.error('Start and end time arrays must be same size!')
        return None

    #Load the hdf5 file
    with h5py.File(dataLocation, mode='r') as file:
        #Events are here in this file:
        events_hdf5 = file['CD']['events']
        
        howOftenCheckHdfTime= events_hdf5.size//n_course_chunks
        
        #Create a 'chunk' array that links chunk nrs to times:
        chunk_arr = np.zeros(int(np.ceil(events_hdf5.size/howOftenCheckHdfTime)),)
        
        curr_chunk = 0
        allChunksFound=False
        #Loop over the hdf5 to get course info:
        while (curr_chunk < len(chunk_arr)) and (not allChunksFound):
            index = curr_chunk*howOftenCheckHdfTime
            if index <= events_hdf5.size:
                foundtime = events_hdf5[index]['t']/1000
                chunk_arr[curr_chunk] = foundtime
            curr_chunk+=1
            if curr_chunk > len(chunk_arr):
                allChunksFound = True

        for index,start_time in enumerate(requested_start_time_ms_arr):
            #Get corresponding end_time:
            end_time = requested_end_time_ms_arr[index]
            #First get the coarse/wide start/end index:
            wide_start_index = np.max((0,np.where(chunk_arr > start_time)[0][0]-1))
            wide_end_index = np.min((events_hdf5.size,np.where(chunk_arr <= end_time)[0][-1]))
            
            #If wide_start_index is e.g. at index 10, we know for sure our exact start time is somewhere between 10 and 11.
            #Thus, we exract the data between wide_start_index and wide_start_index+1:
            precise_start_lookupdata = events_hdf5[wide_start_index*howOftenCheckHdfTime:(wide_start_index+1)*howOftenCheckHdfTime]['t']/1000
            precise_end_lookupdata = events_hdf5[wide_end_index*howOftenCheckHdfTime:(wide_end_index+1)*howOftenCheckHdfTime]['t']/1000
            #Find the index where precise_start_lookup data is the first time that is higher than the start time:
            lookup_start_index = int(np.where(precise_start_lookupdata >= start_time)[0][0]+wide_start_index*howOftenCheckHdfTime)
            #Check if we're requesting an end_time that is before the end of the file:
            if end_time < precise_end_lookupdata[-1]:
                lookup_end_index = int(np.where(precise_end_lookupdata > end_time)[0][0]+wide_end_index*howOftenCheckHdfTime-1)
            else: #Set to teh hdf5 size
                lookup_end_index = int(events_hdf5.size-1)
            #Add to big array
            startEndIndecesHdf5File[index] = (lookup_start_index,lookup_end_index) 
        
        return startEndIndecesHdf5File

def timeSliceFromHDFFromIndeces(dataLocation,startEndIndecesHdf5File,index=0):
    #Get the time slice from a hdf5 file for index, after running findIndexFromTimeSliceHDF
    with h5py.File(dataLocation, mode='r') as file:
        events_hdf5 = file['CD']['events']
        events =  events_hdf5[int(startEndIndecesHdf5File[index][0]):int(startEndIndecesHdf5File[index][1])]
    return events


def filter_finding_on_chunking(candidate,chunking_limits):
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

def filterEvents_xy(events,xyStretch=(-np.Inf,-np.Inf,np.Inf,np.Inf)):
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

def filterEvents_p(events,pValue=0):
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

def RawToNpy(filepath,metaVisionPath,storeConvertedData=False,buffer_size = 5e7, n_batches=5e7):
    import sys
    if(os.path.exists(filepath[:-4]+'.npy')):
        events = np.load(filepath[:-4]+'.npy')
        logging.info('NPY file from RAW was already present, loading this instead of RAW!')
    else:
        logging.info('Starting to load RAW...')
        sys.path.append(metaVisionPath)
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
        if storeConvertedData:
            np.save(filepath[:-4]+'.npy',events)
            logging.debug('NPY file created')
        logging.info('Raw data loaded')
    return events

def readRawTimeStretch(filepath,metaVisionPath,buffer_size = 5e7, n_batches=5e7, timeStretchMs=[0,1000]):
    import sys
    #Function to read only part of the raw file, between time stretch [0] and [1]
    logging.info('Starting to load RAW...')
    sys.path.append(metaVisionPath)
    from metavision_core.event_io.raw_reader import RawReader
    record_raw = RawReader(filepath,max_events=int(buffer_size))
    #First seek to the start-time:
    record_raw.seek_time(timeStretchMs[0]*1000)
    #Then load the time in a single batch:
    events = record_raw.load_delta_t(timeStretchMs[1]*1000-timeStretchMs[0]*1000)
    record_raw.reset()
    return events

def removeHotPixelEvents(events,hotPixelArray=None):
    origEventLen = len(events)
    #Remove all events that are on a hot pixel:
    if hotPixelArray is not None:
        for id in range(len(hotPixelArray)):
            events = events[~((events['x'] == hotPixelArray[id][0]) & (events['y'] == hotPixelArray[id][1]))]
    finalEventLen = len(events)
    if origEventLen-finalEventLen > 0:
        print("Removed "+str(origEventLen-finalEventLen)+" hot pixel events, which is "+str(round(100*(origEventLen-finalEventLen)/origEventLen,2))+"pct of all events.")
    return events


class SmallWindow(QMainWindow):
    """ 
    General class that creates a small popup window to have some data. Mostly used for utility functions.
    """
    
    #Create a small window that pops up
    def __init__(self, parent, windowTitle="Small Window"):
        super().__init__(parent)
        self.setWindowTitle(windowTitle)
        self.resize(300, 200)

        self.parent = parent
        # Set the window icon to the parent's icon
        self.setWindowIcon(self.parent.windowIcon())
        
        #Add a layout
        layout = QVBoxLayout()
        self.setCentralWidget(QWidget())  # Create a central widget for the window
        self.centralWidget().setLayout(layout)  # Set the layout for the central widget

    #Function to find/select a file and add it to the lineedit
    def openFileDialog(self,fileArgs = "All Files (*)"):
        options = QFileDialog.Options()
        #Try to get current folder from self.fileLocationLineEdit:
        try:
            #Split the filelocationtext on slash:
            filefolder = self.fileLocationLineEdit.text().split('/')
            #Get all but the last element of this:
            filefolder = '/'.join(filefolder[:-1])
            folderName = filefolder
        except:
            folderName = ""
        
        file_name, _ = QFileDialog.getOpenFileName(None, "Open File", folderName, fileArgs, options=options)
        if file_name:
            self.fileLocationLineEdit.setText(file_name)
        
        return file_name
    
    #Add extra text before the period
    def addTextPrePriod(self,lineedit,LineEditText,textAddPrePeriod = ""):
        #Add the textAddPrePriod directly before the last found period in the LineEditText:
        if textAddPrePeriod != "":
            try:
                LineEditText = LineEditText.split('.')
                LineEditText[-2] = LineEditText[-2]+textAddPrePeriod
                LineEditText = '.'.join(LineEditText)
            except:
                pass
        lineedit.setText(LineEditText)
        
    #change the text after the period
    def changeTextPostPeriod(self,lineedit,LineEditText,textPostPeriod = ""):
        #Add the textAddPrePriod directly before the last found period in the LineEditText:
        if textPostPeriod != "":
            try:
                LineEditText = LineEditText.split('.')
                LineEditText[-1] = textPostPeriod
                LineEditText = '.'.join(LineEditText)
            except:
                pass
        lineedit.setText(LineEditText)
    
    def addMarkdown(self,mdfile,width=700,height=800):
        newlayout = QVBoxLayout()
        markdownViewer = QWebEngineView()
        markdownViewer.setFixedHeight(height)
        markdownViewer.setFixedWidth(width)
        md_file = mdfile
        with open(md_file, 'r', encoding='utf-8') as file:
            md_content = file.read()
        # Convert Markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['markdown_captions','fenced_code', 'codehilite', 'toc', 'attr_list', 'meta'])
        # Get the directory of the Markdown file
        base_dir = os.path.dirname(os.path.abspath(md_file))
        # Create a complete HTML document with MathJax support
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="text/javascript" async
                src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
            </script>
            <script type="text/x-mathjax-config">
                MathJax.Hub.Config({{
                    tex2jax: {{
                        inlineMath: [['$','$']],
                        processEscapes: true
                    }} 
                }});
            </script>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Load the HTML content into the web view
        markdownViewer.setHtml(full_html, QUrl.fromLocalFile(base_dir + "/"))
        newlayout.addWidget(markdownViewer)
        self.centralWidget().layout().addLayout(newlayout)
    
    def addDescription(self,description):
        #Create a horizontal box layout:
        layout = QHBoxLayout()
        #add the description as text, allowing for multi-line text:
        self.descriptionLabel = QLabel(description)
        self.descriptionLabel.setWordWrap(True)
        #Add the label to the layout:
        layout.addWidget(self.descriptionLabel)
        #Add the layout to the central widget:
        self.centralWidget().layout().addLayout(layout)
        return self.descriptionLabel
    
    def addButton(self,buttonText="Button"):
        #Create a horizontal box layout:
        layout = QHBoxLayout()
        #add a button:
        self.button = QPushButton(buttonText)
        #Add the button to the layout:
        layout.addWidget(self.button)
        #Add the layout to the central widget:
        self.centralWidget().layout().addLayout(layout)
        return self.button
    
    def addTextEdit(self,labelText = "Text edit:", preFilledText = ""):
        #Create a horizontal box layout:
        layout = QHBoxLayout()
        #add a label and text edit:
        self.textEdit = QLineEdit()
        self.textEdit.setText(preFilledText)
        #Add the label and text edit to the layout:
        layout.addWidget(QLabel(labelText))
        layout.addWidget(self.textEdit)
        #Add the layout to the central widget:
        self.centralWidget().layout().addLayout(layout)
        return self.textEdit
    
    def addMultiLineTextEdit(self,labelText = "Text edit:", preFilledText = ""):
        #Create a horizontal box layout:
        layout = QHBoxLayout()
        #add a label and text edit:
        self.textEdit = QTextEdit()
        self.textEdit.setText(preFilledText)
        #Add the label and text edit to the layout:
        layout.addWidget(QLabel(labelText))
        layout.addWidget(self.textEdit)
        #Add the layout to the central widget:
        self.centralWidget().layout().addLayout(layout)
        return self.textEdit
    
    #Add a file information label/text/button:
    def addFileLocation(self, labelText="File location:", textAddPrePeriod = "", textPostPeriod = ""):
        #Create a horizontal box layout:
        layout = QHBoxLayout()
        #add a label, line edit and button:
        self.fileLocationLabel = QLabel(labelText)
        self.fileLocationLineEdit = QLineEdit()
        LineEditText = self.parent.dataLocationInput.text()
        
        self.addTextPrePriod(self.fileLocationLineEdit,LineEditText,textAddPrePeriod)
        LineEditText = self.fileLocationLineEdit.text()
        self.changeTextPostPeriod(self.fileLocationLineEdit,LineEditText,textPostPeriod)
        
        self.fileLocationButton = QPushButton("...")
        self.fileLocationButton.clicked.connect(lambda: self.openFileDialog(fileArgs = "All Files (*)"))
        
        #Add the label, line edit and button to the layout:
        layout.addWidget(self.fileLocationLabel)
        layout.addWidget(self.fileLocationLineEdit)
        layout.addWidget(self.fileLocationButton)
        #Add the layout to the central widget:
        self.centralWidget().layout().addLayout(layout)
        return self.fileLocationLineEdit






#For CLI
def timeSliceFromHDF(dataLocation,requested_start_time_ms = 0,requested_end_time_ms=1000,howOftenCheckHdfTime = 100000,loggingBool=False,curr_chunk = 0):
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
                    print('Loading HDF, currently on chunk '+str(curr_chunk)+', at time: '+str(foundtime))

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
                print('End of file reached while chunking HDF5')
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

def printInformationFromFunction(functionName):
    """ 
    Provide information about a function in the terminal.
    """
    helpString = (helpStringFromFunction(functionName)
    )
    kwargsInfo = (kwargsFromFunction(functionName))
    reqKwargs = reqKwargsFromFunction(functionName)
    optKwargs = optKwargsFromFunction(functionName)
    displayNamev = displayNamesFromFunctionNames([functionName],'')

    InfoString = ''
    InfoString += f"\033[1m{displayNamev[0][0]}\033[0m\n"
    InfoString += f"\033[31mHelp info\033[0m \n\033[33m{helpString}\033[0m\n"
    if len(reqKwargs) > 0:
        InfoString += f"\033[31mRequired parameters\033[0m\n"
        for i in range(len(reqKwargs)):
            InfoString+=f"\033[33m{displayNameFromKwarg(functionName,reqKwargs[i])} [{reqKwargs[i]}]: \n\t{infoFromMetadata(functionName,specificKwarg=reqKwargs[i])} \n\tDefault value: {defaultValueFromKwarg(functionName,kwargname=reqKwargs[i])}\n\033[0m"
    if len(optKwargs) > 0:
        InfoString += f"\033[31mOptional parameters\033[0m\n"
        for i in range(len(optKwargs)):
            InfoString+=f"\033[33m{displayNameFromKwarg(functionName,optKwargs[i])} [{optKwargs[i]}]: \n\t{infoFromMetadata(functionName,specificKwarg=optKwargs[i])} \n\tDefault value: {defaultValueFromKwarg(functionName,kwargname=optKwargs[i])}\n\033[0m"
    return InfoString