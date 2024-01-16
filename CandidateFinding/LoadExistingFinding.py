import inspect, logging
from Utils import utilsHelper
import pickle

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "LoadExistingFinding": {
            "required_kwargs": [
                {"name": "File_Location", "description": "Location of a FindingResults .pickle file"},
            ],
            "optional_kwargs": [
             ],
            "help_string": "Loads a previously created FindingResults and only runs fitting routine on this.",
            "display_name": "Load an existing Finding Result"
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def LoadExistingFinding(npy_array,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    try:
        #Check if ends with .pickle:
        if kwargs['File_Location'][-7:] != '.pickle':
            kwargs['File_Location'] = kwargs['File_Location']+'.pickle'

        with open(kwargs['File_Location'], 'rb') as file:
            candidates = pickle.load(file)
        performance_metadata = f"Loaded file {kwargs['File_Location']}."
        logging.info('Existing Finding result correctly loaded')
    except:
        logging.error('Issue with loading an existing finding!')
        candidates = None
        performance_metadata = None
    return candidates, performance_metadata
