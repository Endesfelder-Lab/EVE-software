import pickle
import numpy as np
import matplotlib.pyplot as plt

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
        
        
# Spectral clustering coverslip: /home/laura/PhD/Event_Based_Sensor_Project/RAW_Analysis/data/coverslip/AF647_coverslip_FindingResults_20240115_095139.pickle
# DBSCAN coverslip: /home/laura/PhD/Event_Based_Sensor_Project/RAW_Analysis/data/coverslip/AF647_coverslip_FindingResults_20240115_095045.pickle
# Frame based coverslip: /home/laura/PhD/Event_Based_Sensor_Project/RAW_Analysis/data/coverslip/AF647_coverslip_FindingResults_20240112_144137.pickle

filepath = "/home/laura/PhD/Event_Based_Sensor_Project/RAW_Analysis/data/coverslip/AF647_coverslip_FindingResults_20240115_095045.pickle"
candidate_dic = open_pickle_file(filepath)
nr_candidates = len(candidate_dic)
method = "/home/laura/PhD/Event_Based_Sensor_Project/RAW_Analysis/CandidateFinding_Stats/DBSCAN"

candidate_dims = np.ndarray(nr_candidates, dtype={'names':['x','y','t'], 'formats':['int','int', 'int']})
print(nr_candidates)
candidate_Nevs = np.ndarray(nr_candidates)
candidate_COM = np.ndarray(nr_candidates, dtype={'names':['x','y'], 'formats':['float','float']})
candidate_COM_centered = np.ndarray(nr_candidates, dtype={'names':['x','y'], 'formats':['float','float']})
i=0

for candidate in candidate_dic.values():
    candidate_dims[i]['x'] = candidate['cluster_size'][0]
    candidate_dims[i]['y'] = candidate['cluster_size'][1]
    candidate_dims[i]['t'] = candidate['cluster_size'][2]
    candidate_Nevs[i] = candidate['N_events']
    candidate_COM[i]['x'] = np.mean(candidate['events']['x']-np.min(candidate['events']['x']))
    candidate_COM[i]['y'] = np.mean(candidate['events']['y']-np.min(candidate['events']['y']))
    candidate_COM_centered[i]['x'] = candidate_COM[i]['x']-float(candidate['cluster_size'][0]-1.0)/2.0 # I'm pretty sure we need the -1 here, as max-min=dim-1
    candidate_COM_centered[i]['y'] = candidate_COM[i]['y']-float(candidate['cluster_size'][1]-1.0)/2.0
    i+=1

# spatial dimension distribution
plt.figure(1)
counts, bin_edges = hist(candidate_dims['x'], 1)
plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='x-dim', alpha=0.5)
counts, bin_edges = hist(candidate_dims['y'], 1)
plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='y-dim', alpha=0.5)
plt.xlabel("spatial cluster dimensions in [px]")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_spatial.png")

# temporal dimension distribution
plt.figure(2)
# counts, bin_edges = hist(candidate_dims['t']*1e-6, 0.5)
# plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='t-dim', alpha=0.1)
plt.hist(candidate_dims['t']*1e-6, bins='auto', label='t-dim')
plt.xlabel("temporal cluster dimensions in [s]")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_temporal.png")

# N_events per cluster
plt.figure(3)
plt.hist(candidate_Nevs, bins='auto', label="events per cluster")
plt.xlabel("number of events per cluster")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_events.png")

# Candidate area
plt.figure(4)
# counts, bin_edges = hist(candidate_dims['x']*candidate_dims['y'], 10)
# plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='xy-area', alpha=0.5)
plt.hist(candidate_dims['x']*candidate_dims['y'], bins='auto', label='xy-area')
plt.xlabel(r"cluster area in [px$^2$]")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_area.png")

# Candidate volume
plt.figure(5)
# counts, bin_edges = hist(candidate_dims['x']*candidate_dims['y']*candidate_dims['t']*1e-6, 3.0)
# plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='xyt-volume', alpha=0.5)
plt.hist(candidate_dims['x']*candidate_dims['y']*candidate_dims['t']*1e-6, bins='auto', label='xyt-volume')
plt.xlabel(r"cluster volume in [px$^2$ s]")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_volume.png")

# Candidate density
plt.figure(6)
mask0 = (candidate_dims['x']>0) & (candidate_dims['y']>0) & (candidate_dims['t']>0)
n0 = np.count_nonzero(mask0 == False)
print(f"A dimension of 0 was detected for {n0} times.")
N_events = candidate_Nevs[mask0]
filtered_dims = candidate_dims[mask0]
plt.hist(N_events/(filtered_dims['x']*filtered_dims['y']*filtered_dims['t'])*1e6, bins='auto', label='Event density')
plt.xlabel(r"cluster density in [px$^{-2}$ s$^{-1}$]")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_density.png")

# Cluster symmetry
plt.figure(7)
counts, bin_edges = hist(candidate_dims['x']/candidate_dims['y'], 0.1)
plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='x/y-symmetry', alpha=0.5)
counts, bin_edges = hist(candidate_dims['y']/candidate_dims['x'], 0.1)
plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='y/x-symmetry', alpha=0.5)
plt.xlabel("cluster symmetry")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_symmetry.png")

# Symmetry of COM
plt.figure(8)
counts, bin_edges = hist(candidate_COM_centered['x'], 0.5)
plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='x-COM centering', alpha=0.5)
counts, bin_edges = hist(candidate_COM_centered['y'], 0.5)
plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',label='y-COM centering', alpha=0.5)
# plt.hist(candidate_COM_centered['x'], bins='auto', label='x-COM centering', alpha=0.5)
# plt.hist(candidate_COM_centered['y'], bins='auto', label='y-COM centering', alpha=0.5)
plt.xlabel("COM centering")
plt.ylabel("probability")
plt.legend()
plt.savefig(method + "_COM.png")

plt.show()

