import inspect, logging
from Utils import utilsHelper
import pickle

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "LoadExistingFitting": {
            "required_kwargs": [
                {"name": "File_Location", "display_text":"file location", "description": "Location of a FittingResults .pickle file","type":"fileLoc"},
            ],
            "optional_kwargs": [
             ],
            "help_string": "Loads a previously created FittingResults and doesn't run finding and fitting.",
            "display_name": "Load an existing Fitting Result"
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def LoadExistingFitting(candidate_dic,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    try:
        #Check if ends with .pickle:
        if kwargs['File_Location'][-7:] != '.pickle':
            kwargs['File_Location'] = kwargs['File_Location']+'.pickle'

        with open(kwargs['File_Location'], 'rb') as file:
            candidateRef = pickle.load(file)
            localizations = pickle.load(file)
        performance_metadata = f"Loaded file {kwargs['File_Location']}."
        logging.info('Existing Fitting result correctly loaded')
    except:
        logging.error('Issue with loading an existing fitting!')
        localizations = None
        performance_metadata = None
    return localizations, performance_metadata
