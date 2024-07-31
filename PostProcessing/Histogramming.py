import inspect
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
    from Utils import utilsHelper
import pandas as pd
import numpy as np
import time
import logging

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "histogramming": {
            "required_kwargs": [
                {"name": "Value", "description": "Filter text. E.g. \"x < 50 & y > 100\". ","default":"x [nm]","display_text":"Key","type":"dropDown(__locListHeaders__)"},
                {"name": "n_bins", "description": "Number of histogram bins ","default":100,"display_text":"Number of bins","type":int},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Histograms result data",
            "display_name": "Histogramming"
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def histogramming(localizations,findingResult,settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    orig_len_localizations = len(localizations)
    
    # Start the timer
    start_time = time.time()
    
    key = kwargs['Value']
    
    data_to_hist = localizations[key]
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Calculate statistics
    mean_val = np.mean(data_to_hist)
    median_val = np.median(data_to_hist)
    std_val = np.std(data_to_hist)
    min_val = np.min(data_to_hist)
    max_val = np.max(data_to_hist)
    skew_val = stats.skew(data_to_hist)
    kurtosis_val = stats.kurtosis(data_to_hist)
    q75, q25 = np.percentile(data_to_hist, [75, 25])
    iqr = q75 - q25


    # Create a new figure with extra space for the text
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [4, 1]})

    # Plot the histogram on the left subplot
    ax1.hist(data_to_hist, bins=int(kwargs['n_bins']))
    ax1.set_xlabel(key)
    ax1.set_ylabel('Count')
    ax1.set_title(f'Histogram of {key}')

    # Add statistical information to the right subplot
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nSkew: {skew_val:.2f}\nKurtosis: {kurtosis_val:.2f}\nIQR: {iqr:.2f}'
    ax2.text(0.05, 0.95, stats_text, verticalalignment='top', fontsize=10)
    ax2.axis('off')  # Hide the axis of the right subplot

    plt.tight_layout()
    plt.show()
    
    #Required output: localizations
    metadata = '-'
    return localizations,metadata
