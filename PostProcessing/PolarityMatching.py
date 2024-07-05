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
            "help_string": "Link and filter on positive and negative events.",
            "display_name": "Polarity Matching - match on and off events"
        },
        "PolarityMatching_NeNA": {
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Perform NeNA on the matched polarities.",
            "display_name": "Nearest neighbour analysis (NeNA) precision on matched polarities"
        },
        "PolarityMatching_NeNASpatial": {
            "required_kwargs": [
                {"name": "n_points_per_bin", "description": "Number of points per bin","default":"200","type":int,"display_text":"Number of points per bin"},
                
            ],
            "optional_kwargs": [
            ],
            "help_string": "Perform NeNA on the matched polarities.",
            "display_name": "Spatial Nearest neighbour analysis (NeNA) precision on matched polarities"
        },
        "PolarityMatching_time": {
            "required_kwargs": [
                {"name": "timeOffsetPerc", "description": "Percentage time offset","default":"20","type":float,"display_text":"Time offset (%)"},
                {"name": "nPops", "description": "Number of populations to fit, max 3","default":"1","type":int,"display_text":"Number of populations (max 3)"},
                
            ],
            "optional_kwargs": [
            ],
            "help_string": "Find the average lifetime of fluorophores.",
            "display_name": "Lifetime analysis on matched polarities"
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
        #Need to copy for some reason
        localizations = localizations.copy()
        #remove the nans:
        localizations = localizations.dropna()
        #reset the index:
        localizations = localizations.reset_index()

        #add empty columns to localizations:
        localizations.loc[:,'pol_link_id'] = -1
        localizations.loc[:,'pol_link_time'] = 0
        localizations.loc[:,'pol_link_xy'] = 0
        
        #Get the pos and neg events
        posEvents = localizations[localizations['p']==1]
        negEvents = localizations[localizations['p']==0]
        
        #Sort the pos and neg events:
        posEvents = posEvents.sort_values(by=['t'])
        negEvents = negEvents.sort_values(by=['t'])
        
        #remove the nans:
        posEvents = posEvents.dropna()
        negEvents = negEvents.dropna()

        if len(posEvents)>len(negEvents): #I'm not completely sure if I need to split it in this if-statement, but it seems to work fine
            mininAll = np.searchsorted(negEvents['t'], posEvents['t'])
            maxinAll = np.searchsorted(negEvents['t'], posEvents['t'] + float(kwargs['Max_tDistance']))
            #We loop over the positive events:
            for posEventId,posEvent in posEvents.iterrows():
                if posEventId < len(posEvents):
                    if np.mod(posEventId,500) == 0:
                        logging.info('PolarityMatching progress: ' + str(posEventId) + ' of ' + str(len(posEvents)))
                    
                    minin = mininAll[posEventId]
                    maxin = maxinAll[posEventId]
                    
                    posEvent = posEvents.loc[posEventId]
                
                    negEventsInTime = negEvents[minin:maxin]
                    
                    # print(negEventsInTime['t']-posEvent['t'])
                    
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
        else:
            mininAll = np.searchsorted(posEvents['t'], negEvents['t'] - float(kwargs['Max_tDistance']))
            maxinAll = np.searchsorted(posEvents['t'], negEvents['t'])
            #We loop over the positive events:
            for negEventIda,negEvent in negEvents.iterrows():
                negEventId = negEventIda-len(posEvents)
                if negEventId < len(negEvents):
                    if np.mod(negEventId,500) == 0:
                        logging.info('PolarityMatching progress: ' + str(negEventId) + ' of ' + str(len(negEvents)))
                    
                    minin = mininAll[negEventId]
                    maxin = maxinAll[negEventId]
                    
                    negEvent = negEvents.loc[negEventIda]
                
                    posEventsInTime = posEvents[minin:maxin]
                    
                    # print(posEventsInTime['t']-negEvent['t'])
                    
                    x_diff = posEventsInTime['x'].values - negEvent['x']
                    y_diff = posEventsInTime['y'].values - negEvent['y']
                    distance = np.sqrt(x_diff**2 + y_diff**2)

                    foundPosEventId = distance < float(kwargs['Max_xyDistance'])

                    #If we found at least one:
                    if sum(foundPosEventId) > 0:
                        #Find the first id of True (the closest in time)
                        foundPosEventId = np.argmax(foundPosEventId)
                        #Find the corresponding event
                        posEventsWithinDistance = posEventsInTime.iloc[foundPosEventId]
                        #And find the event distance belonging to it
                        eventDistance = distance[foundPosEventId]
                        
                        #Renaming
                        posEventFound = posEventsWithinDistance
                        
                        posEventId = posEventFound._name
                        
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

def showWarningPolMatchingNotRan():
    """
    This function shows a warning dialog when e.g. PolarityMatching-NeNA is run before PolarityMatching..
    """
    #Show a dialog popup:
    from PyQt5.QtWidgets import QMessageBox
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("First run Polarity Matching itself!")
    msg.setWindowTitle("Polarity Matching error")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()
    logging.error('Polarity Matching is required!')

def PolarityMatching_NeNA(localizations,findingResult,settings,**kwargs):
    """
    A function to perform NeNA (Nearest Neighbor Analysis) for polarity matching to assess localization precision.
    This function takes in localizations, findingResult, settings, and optional keyword arguments.
    It checks for pre-run polarity matching and creates a new figure to plot the polarity_linked_xy as a histogram.
    -- REQUIRES to have ran polarityMatching beforehand.
    """
    #Check if we have the pre-run polarity matching:
    if not ('pol_link_id' in localizations.columns and 'pol_link_time' in localizations.columns and 'pol_link_xy' in localizations.columns):
        showWarningPolMatchingNotRan()
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

    #Check if we have the pre-run polarity matching:
    if not ('pol_link_id' in localizations.columns and 'pol_link_time' in localizations.columns and 'pol_link_xy' in localizations.columns):
        showWarningPolMatchingNotRan()
    else:
        #Create a new figure and plot the polarity_linked_xy as a histogram:
        #Pre-filter to only positive events (to only have all links once)
        sublocs = localizations[localizations['p'] == 1]
        #Also remove all -1 pol link xy:
        sublocs = sublocs[sublocs['pol_link_xy']>0]
        
        #Find the 95th percentile:
        perc1 = np.percentile(sublocs['pol_link_time'],1)
        perc95 = np.percentile(sublocs['pol_link_time'],95)
        
        
        histV,bin_edges = np.histogram(sublocs['pol_link_time'], bins=np.linspace(0, perc95, 100), density=True)
        
        #Perform a smoothing on the histogram:
        from scipy.signal import savgol_filter
        filterhistV = savgol_filter(histV, int(len(histV)/5), 4)
        #Find the maximum of the smoothed histogram:
        maxV = max(filterhistV)
        maxBin = np.argmax(filterhistV)
        peakTime = bin_edges[maxBin+1]
        minTimeFit = peakTime*(1+float(kwargs['timeOffsetPerc'])/100)
        minTimeFitBin = np.argmin(np.abs(bin_edges - minTimeFit))
        
        #List of values to fit, where valuesToFit[0] is times, valuesToFit[1] is prob
        valuesToFit = [bin_edges[minTimeFitBin:],histV[minTimeFitBin-1:]]
        
        n_curves = int(kwargs['nPops'])
        #Fit with n_curves exponential decays:
        from scipy.optimize import curve_fit
        def exponential_decay_1(x, a, b):
            return a * np.exp(-b * x)
        def exponential_decay_2(x, a, b, c, f1):
            return a * (f1 * np.exp(-b * x) + (1-f1) * np.exp(-c * x))
        def exponential_decay_3(x, a, b, c, d, f1, f2):
            return a * (f1 * np.exp(-b * x) + f2 * np.exp(-c * x) + (1-f1-f2) * np.exp(-d * x))
        if n_curves == 1:
            popt, pcov = curve_fit(exponential_decay_1, valuesToFit[0], valuesToFit[1], p0=[1, 0.01], bounds=(0, np.inf), maxfev=5000)
            #get RelUncertainty on popt[1]:
            popt1Unc = np.sqrt(pcov[1][1])/popt[1]
        elif n_curves == 2:
            popt, pcov = curve_fit(exponential_decay_2, valuesToFit[0], valuesToFit[1], p0=[1, 0.01, 0.02, 0.5], bounds=(0, np.inf), maxfev=5000)
            #get RelUncertainty on popt[1]:
            popt1Unc = np.sqrt(pcov[1][1])/popt[1]
            popt2Unc = np.sqrt(pcov[2][2])/popt[2]
        elif n_curves == 3:
            popt, pcov = curve_fit(exponential_decay_3, valuesToFit[0], valuesToFit[1], p0=[1, 0.01, 0.02, 0.03, 0.30, 0.33], bounds=(0, np.inf), maxfev=5000)
            #get RelUncertainty on popt[1]:
            popt1Unc = np.sqrt(pcov[1][1])/popt[1]
            popt2Unc = np.sqrt(pcov[2][2])/popt[2]
            popt3Unc = np.sqrt(pcov[3][3])/popt[3]
        
        fig = plt.figure()
        #4 subplots. 2 top ones: linear. 2 bottom ones: log (y).
        #            2 left ones: only in data range. 2 right ones: full exp decay
        colHist = '#606060'
        colHistFilt = 'k'
        linestyleHist = '-'
        linestyleHistFilt = '--'
        colPeak = 'k'
        linestylePeak = '-'
        colOffset = 'r'
        linestyleOffset = '-'
        colDecay = 'b'
        linestyleDecay1 = '-'
        linestyleDecay2 = '--'
        
        ax = fig.add_subplot(231)
        ax.plot(bin_edges[1:],histV,colHist,linestyle=linestyleHist)
        ax.plot(bin_edges[1:],filterhistV,colHistFilt,linestyle=linestyleHistFilt)
        ax.plot([peakTime,peakTime],[0,maxV*1.25],colPeak,linestyle=linestylePeak)
        ax.plot([minTimeFit,minTimeFit],[0,maxV*1.25],colOffset,linestyle=linestyleOffset)
        if n_curves == 1:
            ax.plot(valuesToFit[0],exponential_decay_1(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
            #let the ax title reflect the fit params:
            fig.suptitle('t1/2 = ' + str(round(1/popt[1],2)) + '+-' + str(round(1/popt[1]*popt1Unc,2)) + 'ms')
        elif n_curves == 2:
            ax.plot(valuesToFit[0],exponential_decay_2(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
            #let the ax title reflect the fit params:
            fig.suptitle('t1/2 = ' + str(round(1/popt[1],2)) + '+-' + str(round(1/popt[1]*popt1Unc,2)) + ' (' + str(round(popt[3]*100,1)) +'%) | ' + str(round(1/popt[2],2)) + '+-' + str(round(1/popt[2]*popt2Unc,2)) + '(' + str(round((1-popt[3])*100,1)) +'%)  (ms)')
            # ax.set_title('a = ' + str(round(popt[0],4)) + ', b = ' + str(round(popt[1],4)) + ', c = ' + str(round(popt[2],4)) + ', f1 = ' + str(round(popt[3],4)))
        elif n_curves == 3:
            ax.plot(valuesToFit[0],exponential_decay_3(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
            #let the ax title reflect the fit params:
            #set the ax title for 3 curves:
            fig.suptitle('t1/2 = ' + str(round(1/popt[1],2)) + '+-' + str(round(1/popt[1]*popt1Unc,2)) + ' (' + str(round(popt[4]*100,1)) +'%) | ' + str(round(1/popt[2],2)) + '+-' + str(round(1/popt[2]*popt2Unc,2)) + ' (' + str(round((popt[5])*100,1)) +'%) | ' + str(round(1/popt[3],2)) + '+-' + str(round(1/popt[3]*popt3Unc,2)) + ' (' + str(round((1-popt[4]-popt[5])*100,1)) +'%)  (ms)')
        ax.set_ylabel('Probability')
        
        ax = fig.add_subplot(232)
        ax.plot(bin_edges[1:],histV,colHist,linestyle=linestyleHist)
        ax.plot(bin_edges[1:],filterhistV,colHistFilt,linestyle=linestyleHistFilt)
        ax.plot([peakTime,peakTime],[0,maxV*1.25],colPeak,linestyle=linestylePeak)
        ax.plot([minTimeFit,minTimeFit],[0,maxV*1.25],colOffset,linestyle=linestyleOffset)
        if n_curves == 1:
            ax.plot(bin_edges,exponential_decay_1(bin_edges, *popt), colDecay, linestyle=linestyleDecay2)
            ax.plot(valuesToFit[0],exponential_decay_1(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        elif n_curves == 2:
            ax.plot(bin_edges,exponential_decay_2(bin_edges, *popt), colDecay, linestyle=linestyleDecay2)
            ax.plot(valuesToFit[0],exponential_decay_2(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        elif n_curves == 3:
            ax.plot(bin_edges,exponential_decay_3(bin_edges, *popt), colDecay, linestyle=linestyleDecay2)
            ax.plot(valuesToFit[0],exponential_decay_3(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        
        #add a legend:
        ax.legend(['Data','Smoothed','Peak','Offset','Exp. fit'])
        
        #Plot the difference between expected and data
        ax = fig.add_subplot(233)
        ax.set_title('Expected minus Data')
        
        if n_curves == 1:
            totVals = exponential_decay_1(bin_edges[1:], *popt)
        elif n_curves == 2:
            totVals = exponential_decay_2(bin_edges[1:], *popt)
        elif n_curves == 3:
            totVals = exponential_decay_3(bin_edges[1:], *popt)
            
        ax.plot(bin_edges[1:],totVals-histV,colHist,linestyle=linestyleHist)
        ax.plot(bin_edges[1:],totVals-filterhistV,colHistFilt,linestyle=linestyleHistFilt)
        ax.legend(['Difference','Smoothed difference'])
        
        
        ax = fig.add_subplot(234)
        ax.plot(bin_edges[1:],histV,colHist,linestyle=linestyleHist)
        ax.plot(bin_edges[1:],filterhistV,colHistFilt,linestyle=linestyleHistFilt)
        ax.plot([peakTime,peakTime],[0,maxV*1.25],colPeak,linestyle=linestylePeak)
        ax.plot([minTimeFit,minTimeFit],[0,maxV*1.25],colOffset,linestyle=linestyleOffset)
        if n_curves == 1:
            ax.plot(valuesToFit[0],exponential_decay_1(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        elif n_curves == 2:
            ax.plot(valuesToFit[0],exponential_decay_2(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        elif n_curves == 3:
            ax.plot(valuesToFit[0],exponential_decay_3(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        ax.set_yscale('log')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Time between pos/neg events (ms)')
            
        ax = fig.add_subplot(235)
        ax.plot(bin_edges[1:],histV,colHist,linestyle=linestyleHist)
        ax.plot(bin_edges[1:],filterhistV,colHistFilt,linestyle=linestyleHistFilt)
        ax.plot([peakTime,peakTime],[0,maxV*1.25],colPeak,linestyle=linestylePeak)
        ax.plot([minTimeFit,minTimeFit],[0,maxV*1.25],colOffset,linestyle=linestyleOffset)
        if n_curves == 1:
            ax.plot(bin_edges,exponential_decay_1(bin_edges, *popt), colDecay, linestyle=linestyleDecay2)
            ax.plot(valuesToFit[0],exponential_decay_1(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        elif n_curves == 2:
            ax.plot(bin_edges,exponential_decay_2(bin_edges, *popt), colDecay, linestyle=linestyleDecay2)
            ax.plot(valuesToFit[0],exponential_decay_2(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        elif n_curves == 3:
            ax.plot(bin_edges,exponential_decay_3(bin_edges, *popt), colDecay, linestyle=linestyleDecay2)
            ax.plot(valuesToFit[0],exponential_decay_3(valuesToFit[0], *popt), colDecay, linestyle=linestyleDecay1)
        ax.set_yscale('log')
        ax.set_xlabel('Time between pos/neg events (ms)')
        
        #Plot the difference between expected and data
        ax = fig.add_subplot(236)
        ax.set_title('Expected minus Data')
        
        if n_curves == 1:
            totVals = exponential_decay_1(bin_edges[1:], *popt)
        elif n_curves == 2:
            totVals = exponential_decay_2(bin_edges[1:], *popt)
        elif n_curves == 3:
            totVals = exponential_decay_3(bin_edges[1:], *popt)
            
        ax.plot(bin_edges[1:],totVals-histV,colHist,linestyle=linestyleHist)
        ax.plot(bin_edges[1:],totVals-filterhistV,colHistFilt,linestyle=linestyleHistFilt)
        ax.legend(['Difference','Smoothed difference'])
        ax.set_yscale('log')
        ax.set_xlabel('Time between pos/neg events (ms)')
        
        plt.show()
        



def PolarityMatching_NeNASpatial(localizations,findingResult,settings,**kwargs):
    
    
    #Check if we have the pre-run polarity matching:
    if not ('pol_link_id' in localizations.columns and 'pol_link_time' in localizations.columns and 'pol_link_xy' in localizations.columns):
        showWarningPolMatchingNotRan()
    else:
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
                #print(str(aF[0][0]) + ' at ' + str(xcounter) + ' ' + str(ycounter))
                neNAval[xcounter,ycounter] = aF[0][0]
                ycounter+=1
            xcounter+=1
            ycounter = 0
        

        
        print("Average NeNA value: " + str(np.mean(neNAval)))
        print("Median NeNA value: " + str(np.median(neNAval)))
        print("Standard Deviation of NeNA value: " + str(np.std(neNAval)))

        #Now we have the NeNA values in neNAval, we want to remove regions where precision is too low, we will classify this as regions 
        #Where the NeNA value is greater than the average NeNA value + 2.5 standard deviations

        erraniousRegions = np.where(neNAval > np.mean(neNAval) + 2.5*np.std(neNAval))
        #print("Erranious regions: " + str(erraniousRegions))

        #Now we will remove these regions from the data:
        
        #Without figuring out how to remove these from the original data (would require some arithmatic):
        xcounter = 0
        ycounter = 0

        sublocsNoOutliers = []
        for sublocssplit_i in sublocssplit:
            sublocssplit_i = sublocssplit_i.sort_values(by=['y'])
            sublocssplit_ij = np.array_split(sublocssplit_i, n_colsrows)
            for sublocssplit_ij_i in sublocssplit_ij:
                doAppend = True
                for index in range(len(erraniousRegions[0])):
                    if erraniousRegions[0][index] == xcounter and erraniousRegions[1][index] == ycounter:
                        doAppend = False
                        print("Dropping Region " + str(xcounter) + " " + str(ycounter))
                if doAppend:
                    #print("Appending: " + str(xcounter) + " " + str(ycounter))
                    sublocsNoOutliers.append(sublocssplit_ij_i)
                ycounter+=1
            xcounter+=1
            ycounter=0
                
        sublocsNoOutliers = pd.concat(sublocsNoOutliers)
        pointsremove = len(sublocs) - len(sublocsNoOutliers)
        pointsremovepercentage = pointsremove/len(sublocs)
        print("\nRemoved: " + str(pointsremove) + " points out of " + str(len(sublocs)) + " points")
        print("Percentage removed: " + str(pointsremovepercentage * 100) + "%" + "\n")  

        NeNANoOutliers = runNeNA(sublocsNoOutliers,n_bins_nena=99,loggingShow=False,visualisation=False)
        NeNAWithOutliers = runNeNA(sublocs,n_bins_nena=99,loggingShow=False,visualisation=False)
        print("Total NeNA value with outliers: " + str(NeNAWithOutliers[0][0]))
        print("Total NeNA value without outliers: " + str(NeNANoOutliers[0][0]))

     
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