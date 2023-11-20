# Eve: Single-molecule localization fitting software for event-based sensors

## Overview
Eve is a software package that provides a plethora of options to localize emitters from single molecule localization microscopy (SMLM) experiments performed on a MetaVision event-based sensor (eveSMLM).


## For users
Eve has multiple analysis parts:

1. Processing: This is the core of Eve, and allows for a multitude of options of single-molecule finding and fitting. After processing has finished, Eve creates a .csv file containing the localization data. More details on processing below.
2. Post-processing: This contains general functions that can be run on the obtained list of localizations, such as filtering, drift correction, localization accuracy prediction, etc.
3. Visualisation: Contains options to visualise the obtained localizations.

### General usage instructions:
1. Ensure that the global settings are properly set. In the section 'global settings', press 'advanced settings' and ensure that these are correct. The MetaVision SDK path is the folder that contains all metavision python .pyd files and metavision_core, _hal, _sdk folders.
2. In the 'Processing' tab, choose your dataset location. This could be a MetaVision .raw file, or a .npy file containing x, y, polarity, time columns. If a .npy file is not found, it will be created by converting the .raw file. Alternatively, a folder can be chosen, and the analysis will be run on all .raw and/or .npy files in this folder.
3. Under 'Candidate finding', choose candidate finding options and set required (bold) and optional (italics) parameters. Hover over the title for brief information on the settings and method. The output of the candidate finding routine is a collection of sets of events, where each set of events make up a single localization.
4. Under 'Candidate fitting', choose candidate fitting options and set required (bold) and optional (italics) parameters. Hover over the title for brief information on the settings and method. The output of the candidate fitting routine is a list of localizations.
5. Press 'Run'. Observe running documentary of the fitting process in the 'Run info' tab or Python command line. After fitting is complete, a .csv containing the localizations will be created, and the 'Localization List' tab will be filled with the found localizations.
6. Observe the found localizations in the 'Localization List' tab, visualise these via the 'Visualisation tab' and/or correct/analyse these via the 'Post-processing' tab.

## For developers
Eve is created from the ground up to be flexible and allowing for easy implementation of new candidate finding and fitting routines. To add new routines, a new .py file can be created in the CandidateFinding or CandidateFitting folder, respectively, and these will be automatically found and run by Eve. Here are instructions on how to create these files. For exemplary finding and fitting routines, see CandidateFinding/DummyCandidateFinding.py and CandidateFitting/SimpleOperants.py.

### Function structure
Eve will recognise any and all functions present in the CandidateFinding/CandidateFitting folders, and thus can be expanded at will. Every .py file in these folders is **required** to have the following function as the first function:

~~~python
# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "FunctionOne": {
            "required_kwargs": [
                {"name": "req_kwarg_1", "description": "The first required keyword-argument","default":10},
                {"name": "req_kwarg_2", "description": "The second required keyword-argument","default":"A nice string"},
            ],
            "optional_kwargs": [
                {"name": "opt_kwarg", "description": "An optional keyword-argument"},
            ],
            "help_string": "The help string of the first function.",
            "display_name": "Function One Display Name"
        },
        "FunctionTwo": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "The help string of the second function. This function has no required or optional arguments",
            "display_name": "Function Two Display Name"
        },
    }
~~~

This function metadata provides information and pointers towards all functions found in this .py file. By implementing this .py file, Eve will be expanded by two functions: "Function One Display Name" which calls *FunctionOne*, and "Function Two Display Name", which calls *FunctionTwo*. 
*FunctionOne* will be called with two required keyword-arguments and one optional keyword-arguments. This means that *FunctionOne* will be called with argument *kwarg*, containing *kwarg['req_kwarg_1']* and *kwarg['req_kwarg_2']*. Optionally, *kwarg['opt_kwarg']* will also be present. The *default* values will be the pre-filled values in the Eve software, but Eve does not handle any type-checking. 

In 'def FunctionOne()' (and all other defined functions), it is recommended to run the following line:

~~~python
[provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
~~~

This checks that all required arguments are present, and provides the provided and missing optional arguments.

### Candidate finding
Candidate finding functions should be structured as follows:

~~~python
def FunctionOne(npy_array,settings,**kwargs):
    #Perform function here
    #Create candidates and metadata
    return candidates, metadata
~~~

*FunctionOne* should be a function name defined as such in the file's __function_metadata__. It is allowed to call other functions that are not necessarily defined in __function_metadata__. 

#### Candidate finding input
As seen above, the candidate finding functions are ran with 3 parameters: npy_array, settings, and kwargs (key-word-arguments). 

* npy_array: An N-sized array, where N is the number of events in the dataset. Every event (i.e. npy_array[0]) is a (4-by-1) array containing x, y, p, t (x-pixel, y-pixel, polarisation, time in microsecond), and can be called by e.g. npy_array[0]['x'].
* settings: Contains the global settings. Most importantly, settings['PixelSize_nm'] contains the pixel size in nm.
* kwargs: All keyword arguments as defined above. Please note that **all entries are strings**. I.e. kwarg['req_kwarg_1'] will output **"10"**. If a number is required instead, use 'float(kwarg['req_kwarg_1'])'.

#### Candidate finding output
Candidate fitting should output a *candidates* dictionary and a *metadata* string. 

* The *candidates* dictionary should have an entry for each cluster, and should minimally contain the following entries: **'events'**, which contains all events ([x, y, polarity, time]-headered) belonging to this cluster; **'N_events'**, an integer value of the number of events belonging to this cluster; and **'cluster_size'**, a (3,1)-sized array which contains the x-dimension, y-dimension, and time-dimension, respectively (can be used to filter on).
* The *metadata* string should be a single- or multi-line string, and can store information about the ran candidate finding routine.

#### Pseudo-code to explain the structure of candidates finding output
~~~python
candidates = {}
for cluster in all_clusters:
    clusterEvents = all_cluster_events(cluster_id==cluster)
    candidates[cluster] = {}
    candidates[cluster]['events'] = clusterEvents
    candidates[cluster]['N_events'] = len(clusterEvents)
    candidates[cluster]['cluster_size'] =...
    [np.max(clusterEvents['y'])-np.min(clusterEvents['y']),...
    np.max(clusterEvents['x'])-np.min(clusterEvents['x']),...
    np.max(clusterEvents['t'])-np.min(clusterEvents['t'])]

metadata = 'The file ran as expected!'
~~~
### Candidate fitting
Candidate finding functions should be structured as follows:

~~~python
def FunctionOne_Fitting(candidate_dic,settings,**kwargs):
    #Perform function here
    #Create localizations and metadata
    return localizations, metadata
~~~

*FunctionOne_Fitting* should be a function name defined as such in the file's __function_metadata__. It is allowed to call other functions that are not necessarily defined in __function_metadata__. 

#### Candidate fitting input

As seen above, the candidate fitting functions are ran with 3 parameters: candidate_dic, settings, and kwargs (key-word-arguments). 

* candidate_dic: The output of a candidate finding routine: a dictionary with an entry for each found cluster. See the paragraph above for information on this.
* settings: Contains the global settings. Most importantly, settings['PixelSize_nm'] contains the pixel size in nm.
* kwargs: All keyword arguments as defined above. Please note that **all entries are strings**. I.e. kwarg['req_kwarg_1'] will output **"10"**. If a number is required instead, use 'float(kwarg['req_kwarg_1'])'.

#### Candidate finding output

Candidate fitting should output a *localizations* pandas dataframe and a *metadata* string. 

* The *localizations* pandas dataframe should have a separate entry for each localization, and each localization should at minimum contain the following inputs: **'x'**: x-position in nm (float); **'y'**: y-position in nm (float); **'p'**: polarity of the localization (0 for negative, 1 for positive); **'t'**: time in ms (milliseconds)
* The *metadata* string should be a single- or multi-line string, and can store information about the ran candidate finding routine.

#### Pseudo-code to explain the structure of candidate fitting output

~~~python
localizations = {}
for i in np.unique(list(candidate_dic)):
    localizations[i]={}
    localizations[i]['x'] = np.mean(candidate_dic[i]['events']['x'])*float(settings['PixelSize_nm']['value']) #X position in nm
    localizations[i]['y'] = np.mean(candidate_dic[i]['events']['y'])*float(settings['PixelSize_nm']['value']) #Y position in nm
    localizations[i]['p'] = 1 #Polarisation: 0 or 1
    localizations[i]['t'] = np.mean(candidate_dic[i]['events']['t'])/1000 #time in ms

#Make a pd dataframe out of it - needs to be transposed
localizations = pd.DataFrame(localizations).T

metadata = 'The file ran as expected!'
~~~