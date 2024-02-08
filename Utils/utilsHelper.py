import warnings
import pandas as pd
import numpy as np
import logging

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

# Convert a string representation of truth to true (1) or false (0), or raise an exception
def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true','on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

def removeCandidatesWithLargeBoundingBox(candidates,xymax,tmax):
    npopped = 0
    for candidate in sorted(candidates, reverse=True):
        if candidates[candidate]['cluster_size'][0] > float(xymax) or candidates[candidate]['cluster_size'][1] > float(xymax) or candidates[candidate]['cluster_size'][2] > float(tmax):
            candidates.pop(candidate)
            #set correct numbering of remaining candidates
            candidates = {index: value for index, value in enumerate(candidates.values(), start=0)}
            npopped += 1
    return candidates, npopped

# determin candidate fitting performance information
def info(candidates, fails):
    nb_candidates = len(candidates)
    fit_info = ''
    grouped_fails = fails.groupby('fit_info')
    for fail_reason, count in grouped_fails.size().items():
        fail_text = f'Removed {count}/{nb_candidates} ({count/(nb_candidates)*100:.2f}%) candidates due to {fail_reason}.'
        logging.warning(fail_text)
        fit_info += fail_text + '\n'
    fit_info += '\n'
    for fail_reason, candidate_group in grouped_fails:
        candidate_list = candidate_group['candidate_id'].tolist()
        if len(candidate_list)>20:
            fit_info += f'Candidates discarded by {fail_reason}: [{", ".join(map(str, candidate_list[:10]))}, ..., {", ".join(map(str, candidate_list[-10:]))}]\n'
        else:
            fit_info += f'Candidates discarded by {fail_reason}: {candidate_list}\n'
    return fit_info

# calculate number of jobs on CPU for fitting routine
def nb_jobs(candidate_dic, num_cores):
    nb_candidates = len(candidate_dic)
    if nb_candidates < num_cores or num_cores == 1:
        njobs = 1
        num_cores = 1
    elif nb_candidates/num_cores > 100:
        njobs = np.int64(np.ceil(nb_candidates/100.))
    else:
        njobs = num_cores
    return njobs, num_cores

def get_subdictionary(main_dict, start_key, end_key):
    sub_dict = {k: main_dict[k] for k in range(start_key, end_key + 1) if k in main_dict}
    return sub_dict

# Slice data to distribute the computation on several cores for fitting routine
def slice_data(candidate_dic, nb_slices):
    slice_size=1.*len(candidate_dic)/nb_slices
    slice_size=np.int64(np.ceil(slice_size))
    data_split=[]
    last_key = list(candidate_dic.keys())[-1]
    for k in np.arange(nb_slices):
        keys = [k*slice_size, min((k+1)*slice_size-1,last_key)]
        data_split.append(get_subdictionary(candidate_dic, keys[0], keys[1]))
    return data_split