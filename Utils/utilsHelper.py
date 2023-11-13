import warnings

def argumentChecking(dictionaryInfo, function_name, given_kwargs):
    #Checks whether named required and optional arguments are given or not, and gives errors when things are missing.

    #Get required kwargs and throw error if any are missing
    required_args = [req_kwarg["name"] for req_kwarg in dictionaryInfo[function_name]["required_kwargs"]]
    missing_req_args = [arg for arg in required_args if arg not in given_kwargs]
    if missing_req_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_req_args)} in function {str(function_name)}")
    
    #Get optional kwargs and throw warning if any are missing
    optional_args = [req_kwarg["name"] for req_kwarg in dictionaryInfo[function_name]["optional_kwargs"]]
    missing_opt_args = [arg for arg in optional_args if arg not in given_kwargs]
    provided_opt_args = [arg for arg in optional_args if arg in given_kwargs]
    if missing_opt_args:
        warnings.warn(f"Unused optional arguments: {', '.join(missing_opt_args)} in function {str(function_name)}")
    
    #Return the provided and missing optional arguments
    return [provided_opt_args, missing_opt_args]