import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# create histogram	
def hist(data, bin_width, **kwargs):
    # Calculate the number of bins based on the bin width
    data_range = np.max(data) - np.min(data)
    num_bins = int(data_range / bin_width)
    # Plot the histogram using numpy.hist
    hist_values, bins = np.histogram(data, bins=num_bins, **kwargs)
    return hist_values, bins

def open_pickle_file(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        
filepath_gauss = "/home/laura/PhD/Event_Based_Sensor_Project/RAW_Analysis/data/DNA_nanoruler/20240222/20240222_DNAPAINT_xyCut._FitResults_20240312_101101.csv"
filepath_loggauss = "/home/laura/PhD/Event_Based_Sensor_Project/RAW_Analysis/data/DNA_nanoruler/20240222/20240222_DNAPAINT_xyCut._FitResults_20240312_102237.csv"
localization_df_gauss = pd.read_csv(filepath_gauss)
localization_df_gauss_grouped = localization_df_gauss.groupby('p')
localization_df_log_gauss = pd.read_csv(filepath_loggauss)
localization_df_log_gauss_grouped = localization_df_log_gauss.groupby('p')

print(localization_df_gauss)
print(localization_df_log_gauss)

plt.figure(1)
plt.hist(localization_df_gauss_grouped.get_group(0)['del_x'],bins=100, alpha=0.5, label="x-error Gauss")
plt.hist(localization_df_log_gauss_grouped.get_group(0)['del_x'],bins=100, alpha=0.5, label="x-error LogGauss")
plt.legend()

plt.figure(2)
plt.hist(localization_df_gauss_grouped.get_group(0)['del_y'],bins=100, alpha=0.5, label="y-error Gauss")
plt.hist(localization_df_log_gauss_grouped.get_group(0)['del_y'],bins=100, alpha=0.5, label="y-error LogGauss")
plt.legend()

plt.figure(3)
del_t_gauss = localization_df_gauss_grouped.get_group(0)['del_t']
del_t_log_gauss = localization_df_log_gauss_grouped.get_group(0)['del_t']
del_t_gauss = del_t_gauss[del_t_gauss<200]
del_t_log_gauss = del_t_log_gauss[del_t_log_gauss<200]
plt.hist(del_t_gauss,bins=100, alpha=0.5, label="t-error Gauss")
plt.hist( del_t_log_gauss,bins=100, alpha=0.5, label="t-error LogGauss")
plt.legend()

plt.figure(4)
plt.hist(localization_df_gauss_grouped.get_group(1)['del_x'],bins=100, alpha=0.5, label="x-error Gauss")
plt.hist(localization_df_log_gauss_grouped.get_group(1)['del_x'],bins=100, alpha=0.5, label="x-error LogGauss")
plt.legend()

plt.figure(5)
plt.hist(localization_df_gauss_grouped.get_group(1)['del_y'],bins=100, alpha=0.5, label="y-error Gauss")
plt.hist(localization_df_log_gauss_grouped.get_group(1)['del_y'],bins=100, alpha=0.5, label="y-error LogGauss")
plt.legend()

plt.figure(6)
del_t_gauss = localization_df_gauss_grouped.get_group(1)['del_t']
del_t_log_gauss = localization_df_log_gauss_grouped.get_group(1)['del_t']
del_t_gauss = del_t_gauss[del_t_gauss<200]
del_t_log_gauss = del_t_log_gauss[del_t_log_gauss<200]
plt.hist(del_t_gauss,bins=100, alpha=0.5, label="t-error Gauss")
plt.hist( del_t_log_gauss,bins=100, alpha=0.5, label="t-error LogGauss")
plt.legend()

plt.figure(7)
plt.hist(localization_df_gauss_grouped.get_group(0)['del_x'],bins=100, alpha=0.5, label="x-error Gauss (neg)")
plt.hist(localization_df_gauss_grouped.get_group(1)['del_x'],bins=100, alpha=0.5, label="x-error Gauss (pos)")
plt.legend()

plt.show()