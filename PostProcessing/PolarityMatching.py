import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
import time
import logging
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "PolarityMatching": {
            "required_kwargs": [
                {"name": "Max_xyDistance", "description": "Maximum distance in x,y in nm units (nm)","default":"200","type":float,"display_text":"Max. XY distance (nm)"},
                {"name": "Max_tDistance", "description": "Maximum time distance in ms (between positive and negative)","default":"1000","type":float,"display_text":"Max time distance (ms)"},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Link and filter on positive and negative events."
        },
        "PolarityMatching_NeNA": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Perform NeNA on the matched polarities."
        },
        "PolarityMatching_NeNASpatial": {
            "required_kwargs": [
                {"name": "n_points_per_bin", "description": "Number of points per bin","default":"200","type":int,"display_text":"Number of points per bin"},
                
            ],
            "optional_kwargs": [
            ],
            "help_string": "Perform NeNA on the matched polarities."
        },
        "PolarityMatching_time": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Find the average lifetime of fluorophores."
        }
    }





def compute_area(r, y):
    areaF = abs(np.trapz(y, r))
    return areaF

def cFunc_2dCorr(x, dSMLM, xc, w, A1, A2, A3):
    #dSMLM is the value which you wanna get out of it
    y = (x / (2 * dSMLM * dSMLM)) * np.exp((-1) * x * x / (4 * dSMLM * dSMLM)) * A1 + (A2 / (w * np.sqrt(np.pi * 2))) * np.exp(-0.5 * ((x - xc) / w) * ((x - xc) / w)) + A3 * x
    return y

def CFit_resultsCorr( x, y, initialValue, lowerBound, upperBound):
    A = compute_area(x, y)
    p0 = np.array([initialValue, 15, 100, (A / 2), (A / 2), ((y[98] / 200))])
    bounds = ([lowerBound,0,0,0,0,0],[upperBound,1000,1000,1,1,1])
    popt, pcov = curve_fit(cFunc_2dCorr, x, y, p0=p0, bounds=bounds)
    return popt, pcov
    

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def PolarityMatching(localizations,findingResult,settings,**kwargs):
    """
    Matches positive and negative PSFs with each other. Requires a maximum xy, t distance.
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore
    
    start_time = time.time()
    #Error message and early exit if there isn't both pos and neg events
    pols = np.unique(localizations['p'])
    if not np.array_equal(pols[~np.isnan(pols)], [0, 1]) or np.array_equal(pols[~np.isnan(pols)], [1, 0]):
        logging.error('PolarityMatching requires both positive and negative events!')
        return localizations, 'PolarityMatching requires both positive and negative events!'
    else:
        #add empty columns to localizations:
        localizations['pol_link_id'] = -1
        localizations['pol_link_time'] = 0
        localizations['pol_link_xy'] = 0
        
        #Get the pos and neg events
        posEvents = localizations[localizations['p']==1]
        negEvents = localizations[localizations['p']==0]
        
        
        mininAll = np.searchsorted(negEvents['t'], posEvents['t'])
        maxinAll = np.searchsorted(negEvents['t'], posEvents['t'] + float(kwargs['Max_tDistance']))

        #We loop over the positive events:
        for posEventId,posEvent in posEvents.iterrows():
            if posEventId < len(posEvents):
                if np.mod(posEventId,500) == 0:
                    logging.info('PolarityMatching progress: ' + str(posEventId) + ' of ' + str(len(posEvents)))
                minin = mininAll[posEventId]
                maxin = maxinAll[posEventId]
            
                negEventsInTime = negEvents[minin:maxin]
                
                x_diff = negEventsInTime['x'].values - posEvent['x']
                y_diff = negEventsInTime['y'].values - posEvent['y']
                distance = np.sqrt(x_diff**2 + y_diff**2)

                foundNegEventId = distance < float(kwargs['Max_xyDistance'])

                #If we found at least one:
                if sum(foundNegEventId) > 0:
                    #Find the first id of True (the closest in time)
                    foundNegEventId = np.argmax(foundNegEventId)
                    #Find the corresponding event
                    negEventsWithinDistance = negEventsInTime.iloc[foundNegEventId]
                    #And find the event distance belonging to it
                    eventDistance = distance[foundNegEventId]
                    
                    #Renaming
                    negEventFound = negEventsWithinDistance
                    
                    negEventId = negEventFound._name
                    
                    #Update the positive candidate
                    posEvents.loc[posEventId,'pol_link_id'] = (negEventFound.candidate_id)
                    posEvents.loc[posEventId,'pol_link_time'] = (negEventFound.t - posEvent.t)
                    posEvents.loc[posEventId,'pol_link_xy'] = eventDistance
                    
                    #And update the negative candidate
                    negEvents.loc[negEventId,'pol_link_id'] = (posEvent.candidate_id)
                    negEvents.loc[negEventId,'pol_link_time'] = (posEvent.t-negEventFound.t)
                    negEvents.loc[negEventId,'pol_link_xy'] = eventDistance
                    
                
        #re-create localizations by adding these below one another again:
        localizations = pd.concat([posEvents, negEvents])
                
    end_time = time.time()
    logging.info(f'Polarity Matching took {end_time-start_time} seconds')
    
    #Required output: localizations
    metadata = 'Information or so'
    return localizations,metadata

def runNeNA(sublocs,n_bins_nena=99,loggingShow=True,visualisation=True):
    #Create a histogram for the fitting
    ahist=np.histogram(sublocs['pol_link_xy'],bins=n_bins_nena,density=True)
    #Get the x-values (relative distance)
    ar=ahist[1][1:len(ahist[1])]-1
    #Get the y-values
    ay=ahist[0]
    #Perform the fit
    nena_start = 20;
    nena_lower = 2
    nena_upper = 100
    #Results will be stored in aF, error in aFerr
    aF,aFerr=CFit_resultsCorr(ar,ay,nena_start,nena_lower,nena_upper)
    
    if loggingShow is True:
        #Output on the logging
        logging.info(f'NeNA precision: {np.round(aF[0],2)} +- {np.round(np.sqrt(aFerr[0,0]),2)}nm')
        logging.info(f'All NeNA parameters: {aF.tolist()}')
    
    if visualisation is True:
        #Create the visual fit
        ayf=cFunc_2dCorr(ar,aF[0],aF[1],aF[2],aF[3],aF[4],aF[5])
        
        #Create a plot
        fig, ax = plt.subplots()
        #Show the NeNA histogram
        ax.hist(sublocs['pol_link_xy'],bins=n_bins_nena,density=True)
        #Add the fit
        ax.plot(ar,ayf,'r-')
        # Add a text box at position (0.5, 0.5) with the text "Your Text Here"
        ax.text(0.95, 0.95, "NeNA precision: "+str(np.round(aF[0],2))+ " +- " +str(np.round(np.sqrt(aFerr[0,0]),2)) + "nm", ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
        #Label
        plt.xlabel('Relative pairwise distance (nm)')
        plt.ylabel('Probability')
        #show
        plt.show()
    
    return aF,aFerr

def PolarityMatching_NeNA(localizations,findingResult,settings,**kwargs):
    """
    A function to perform NeNA (Nearest Neighbor Analysis) for polarity matching to assess localization precision.
    This function takes in localizations, findingResult, settings, and optional keyword arguments.
    It checks for pre-run polarity matching and creates a new figure to plot the polarity_linked_xy as a histogram.
    -- REQUIRES to have ran polarityMatching beforehand.
    """
    #Check if we have the pre-run polarity matching:
    if not ('pol_link_id' in localizations.columns and 'pol_link_time' in localizations.columns and 'pol_link_xy' in localizations.columns):
        logging.error('PolarityMatching-NeNA requires to first run Polarity Matching!')
    else:
        #Create a new figure and plot the polarity_linked_xy as a histogram:
        #Pre-filter to only positive events (to only have all links once)
        sublocs = localizations[localizations['p'] == 1]
        #Also remove all -1 pol link xy:
        sublocs = sublocs[sublocs['pol_link_xy']>0]
        
        #Define the nr of bins for nena
        n_bins_nena = 99
        runNeNA(sublocs,n_bins_nena=n_bins_nena,loggingShow=True,visualisation=True)
        
        
        
def PolarityMatching_time(localizations,findingResult,settings,**kwargs):

    #Create a new figure and plot the polarity_linked_xy as a histogram:
    #Pre-filter to only positive events (to only have all links once)
    sublocs = localizations[localizations['p'] == 1]
    #Also remove all -1 pol link xy:
    sublocs = sublocs[sublocs['pol_link_xy']>0]
    
    #Find the 95th percentile:
    perc1 = np.percentile(sublocs['pol_link_time'],1)
    perc95 = np.percentile(sublocs['pol_link_time'],95)
    
    #Create a plt figure with 2 subplots:
    fig = plt.figure()
    ax = fig.add_subplot(121)
    #Create the histogram where the bins are defined by the 95th percentile:
    ax.hist(sublocs['pol_link_time'], bins=np.linspace(0, perc95, 100), density=True)
    
    #Label
    plt.xlabel('Time between pos/neg events (ms)')
    plt.ylabel('Probability')
    
    ax2 = fig.add_subplot(122)
    #show the same data but on log x-scale:
    ax2.hist(sublocs['pol_link_time'], bins=np.logspace(np.log10(perc1), np.log10(perc95), 100), density=True)
    ax2.set_yscale('log')
    plt.xlabel('Time between pos/neg events (ms)')
    plt.show()



def PolarityMatching_NeNASpatial(localizations,findingResult,settings,**kwargs):
    
    
    #Pre-filter to only positive events (to only have all links once)
    sublocs = localizations[localizations['p'] == 1]
    #Also remove all -1 pol link xy:
    sublocs = sublocs[sublocs['pol_link_xy']>0]
    
    #We will cluster the points in sublocs into n_bins
    #But we base this on n_points_per bin
    n_points_per_bin = int(kwargs['n_points_per_bin'])
    n_bins = len(sublocs)//n_points_per_bin
    
    n_colsrows = int((np.sqrt(n_bins)))
    logging.info("n_colsrows: " + str(n_colsrows))
    
    #Split the sublocs data in these bins in x,y:
    #sort data on x:
    sublocs = sublocs.sort_values(by=['x'])
    #split
    sublocssplit = np.array_split(sublocs, n_colsrows)
    xcounter = 0
    ycounter = 0
    neNAval = np.zeros((n_colsrows,n_colsrows))
    
    for sublocssplit_i in sublocssplit:
        sublocssplit_i = sublocssplit_i.sort_values(by=['y'])
        sublocssplit_ij = np.array_split(sublocssplit_i, n_colsrows)
        for sublocssplit_ij_i in sublocssplit_ij:
            aF = runNeNA(sublocssplit_ij_i,n_bins_nena=99,loggingShow=False,visualisation=False)
            neNAval[xcounter,ycounter] = aF[0][0]
            ycounter+=1
        xcounter+=1
        ycounter = 0
    
    
    #Plot the figure
    plt.figure()
    #plot a 2d image:
    plt.imshow(neNAval, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()
    
    
    #Percentage error
    # pcterr = 0.2
    
    # subsamplerate = 50
    # #We do a contrained k_means clustering
    # from k_means_constrained import KMeansConstrained
    # clf = KMeansConstrained(
    #     n_clusters=n_bins,
    #     size_min=n_points_per_bin*(1-pcterr)//subsamplerate,
    #     size_max=int(np.ceil(n_points_per_bin*(1+pcterr)/subsamplerate)),
    #     random_state=0
    # )
    # #randomly subsample the data:
    # sublocspartial = sublocs.sample(frac=1/subsamplerate, random_state=0)
    # clf.fit_predict(sublocspartial[['x','y']].values)
        
    # #create a figure:
    # pxsizenm = 10;
    # figxsize = int(np.ceil(np.ceil(localizations['x'].max() - localizations['x'].min())//pxsizenm))
    # figysize = int(np.ceil(np.ceil(localizations['y'].max() - localizations['y'].min())//pxsizenm))
    
    # # Create grid of x and y coordinates
    # x_coords, y_coords = np.meshgrid(np.arange(figxsize), np.arange(figysize))

    # # Adjust cluster centers
    # adjusted_cluster_centers = clf.cluster_centers_ - np.array([localizations['x'].min(), localizations['y'].min()])

    # # Calculate distances
    # distances = np.sum((adjusted_cluster_centers[:, np.newaxis, np.newaxis, :] - 
    #                     np.stack([x_coords*pxsizenm, y_coords*pxsizenm], axis=-1))**2, axis=-1)

    # # Find index of minimum distance
    # closest_cluster = np.argmin(distances, axis=0)

    # #Calculate NeNA for entries in each cluster
    # neNAval = np.zeros(n_bins)
    # for l in range(0,n_bins):
    #     locs = sublocs[clf.labels_==l]
    #     #Calculate NeNA of this cluster:
    #     aF = runNeNA(locs,n_bins_nena=99,loggingShow=True,visualisation=False)
    #     neNAval[l] = aF[0][0]

    # # Assign values to the imagearray
    # imagearray = neNAval[closest_cluster]
    
    # #Plot the figure
    # plt.figure()
    # #plot a 2d image:
    # plt.imshow(imagearray, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    #Required output: localizations
    metadata = 'Information or so'
    return localizations,metadata