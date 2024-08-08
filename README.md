---
bibliography: Markdown_info/citations.bib
---

# Supplementary Note 2: Software manual

## EVE - General-purpose software for eveSMLM localization
![](Quick_Start/EveSmall.png)

## About EVE
EVE is a user-interfaced software package that provides several methods to localize emitters from single molecule localization microscopy (SMLM) experiments performed on event-based sensors (eveSMLM).

Event-based data differs fundamentally from conventional camera images. Unlike traditional sensors, event-based sensors only capture intensity changes, registering them as either positive (when intensity surpasses a predefined threshold) or negative events (when intensity drops below a predefined threshold). As a result, only a list of x and y pixel coordinate pairs is stored together with the detected event polarities and timestamps.

EVE is designed to quickly and directly process and analyse event-based single molecule data. The event-based data analysis is divided into three main parts:<br>  
1. **Candidate Finding:** The complete event-list is searched for characteristic event clusters that are generated by blinking fluorophores. Potential candidate clusters are then extracted and returned for further processing.<br>  
2. **Candidate Fitting:** The x,y,(z),t-localization is determined for each candidate cluster.<br>  
3. **Postprocessing and Evaluation:** Various analytical routines to process and interpret the data.<br>  

EVE allows flexible combinations of different finding and fitting routines to optimize the localization results for the specific event-based dataset. Besides a variety of different finding and fitting algorithms, EVE also offers various preview options, visualisation tools and post-processing and evaluation functions. A detailed description of the algorithms can be found in **Supplementary Note 1: Analysis methods implemented in EVE**.

EVE is written in Python and structured in such a way, that is easy to implement and add new functionalties in form of new finding, fitting routines, etc. Details about this can be found in the **Supplementary Note 3: Developer Instructions**. EVE can also be run via the command line interface, see `EVE_CommandLine.ipynb` for detailed information.

<!-- TOC -->
# Contents
- [How to install and run EVE](#how-to-install-and-run-eve)
   - [Installation instructions](#installation-instructions)
      - [Optional](#optional)
   - [Running instructions](#running-instructions)
- [Quick Start Guide](#quick-start-guide)
   - [Set up the Processing Tab](#set-up-the-processing-tab)
   - [Perform a Preview Run](#perform-a-preview-run)
   - [Explore the Candidate Preview](#explore-the-candidate-preview)
   - [Notes of best finding/fitting algorithms and parameters](#notes-of-best-findingfitting-algorithms-and-parameters)
   - [Execute a full Run](#execute-a-full-run)
   - [Visualize the Localization Results](#visualize-the-localization-results)
   - [Apply a Drift Correction](#apply-a-drift-correction)
      - [Under Windows](#under-windows)
      - [Under Linux](#under-linux)
   - [The final Localization List](#the-final-localization-list)
   - [Estimate the Localization Precision](#estimate-the-localization-precision)
   - [Metadata, run and result files](#metadata-run-and-result-files)
   - [Command-line interface](#command-line-interface)
   - [Exemplary data](#exemplary-data)
<!-- /TOC -->

## How to install and run EVE
This software was tested under Linux (Ubuntu 20.04) and Windows 10/11. Besides the drift correction module `DriftCorr_DME: DriftCorr_entropyMin` and the Gaussian visualization methods (`Gaussian_display: GaussianKernal_fixedSigma`, `Gaussian_display: GaussianKernal_locPrec`) that use pre-compiled dll-files and are therefore only running on Windows, everthing is running under MacOS, Linux and Windows. The software requires Python 3.9.

### Installation instructions
First download EVE or clone it to your local repository using git:  

```bash
git clone https://github.com/Endesfelder-Lab/Eve.git
```

Create a virtual environment, and install EVE:

**With conda**  
Install conda from the website (www.conda.io)  
Run the following lines in the terminal:  
```bash
conda env create --name EVE python=3.9
conda activate EVE
pip install eve-SMLM"
```

**With virtualenv**  

First, install python 3.9 from python.org.

Replace `PYTHON_PATH` by your python path, e.g. `/usr/bin` and `ENVIRONMENT_PATH` by the path to the virtual environments on your machine and follow the instructions below:
```bash
virtualenv -p PYTHON_PATH/python3.9.18 ENVIRONMENT_PATH/Eve
pip install eve-SMLM"
```

#### Optional 
EVE can read and process event-based data in `.npy` and `.hdf5` format. Additionally the `.raw` format of [Prophesee](https://www.prophesee.ai/) can be used. If you have `.raw` data that you want to analyze you need to install the [Metavision SDK from Prophesee](https://docs.prophesee.ai/stable/installation/index.html) beforehand ([https://docs.prophesee.ai/stable/installation/index.html](https://docs.prophesee.ai/stable/installation/index.html)). We recommend using `.hdf5` whenever possible, as this is a hierarchical data format optimized for efficient saving and reading of large files.


### Running instructions
Start the software by running `eve-SMLM` in the terminal


## Quick Start Guide

### 1. Set up the Processing Tab
Running `GUI.py` will open EVE's graphical user interface which you can see on the right side of the figure.

![](Quick_Start/1_Setting_up_GUI_new.png)

The main window has 7 major parts that are marked with red boxes and are described in more detail in the following.  
1. **Menu bar:** By clicking on `Settings`, you can open the `Advanced settings`, save or load the current GUI configuration. Under `Utilities` you will find some additional functionalities to pre-process the raw event data files before processing them with Eve. Under `Help` you have access to all important information around EVE, such as the Users manual, the Developers manual and the Scientific information.<br>  
The folder `Data\Nanoruler` contains the GUI configuration (`GUI_settings_Nanoruler.json`) that we will use throughout this users manual. You can load the GUI configuration via `Settings -> Load specific GUI contents` and selecting the correct path. Now, open the `Advanced settings`, where you can adapt more advanced settings.<br>  
   <br>  
   ![](Quick_Start/3_advanced_settings.png)<br>  
2. **Data to analyse:** Here, you specify the data that will be analysed in the following. You can either select a single file (in `.npy`, `.hdf5` or `.raw` format) or a folder. If you select a folder all files in the folder will be analysed one after another.

   The folder `Data\Nanoruler` contains an exemplary event-based acquisition of a DNA nanoruler (`Nanoruler_Small.hdf5`) which we will use in this tutorial. Fill the path entry field `Dataset location:` with the corresponding path to the nanoruler dataset.

   In the `Data selection` box, you can now further specify which parts of the data should be analysed and how. You have different options for `Polarity`, `Time` and `Position`. Choose `Pos and neg seperately` as `Polarity` option while leaving the remaining settings unchanged. Thereby, you simply load the all events without temporal or spatial constraints. By selecting `Pos and neg seperately` all subsequent analysis steps will be run on the positive and negative events distinctly. As you can see, the GUI has adapted to your selection and you can now choose finding and fitting routines for positive and negative events seperatly.<br>  

![](Quick_Start/4_change_and_save_GUI_contents.png)<br>  
3. **Candidate Finding routine:** Here, you can select among different candidate finding routines. Choose `Eigen-feature analysis`, both for positive and negative events and change the settings as shown in the screenshot above.<br>  
4. **Candidate Fitting routine:** Here, you can specify which fitting routines you want to use to get localizations for each candidate, cluster. Choose `2D Gaussian` again for both polarities and modify all other parameters as shown in the screenshot above.<br>  
Everything is now ready for the first run. Before you start the first run, save the GUI settings (`Settings -> Save GUI contents`). When you open EVE again, the last saved GUI settings will be loaded automatically.<br>  
5. **Run box:** When you click run, a full run will be executed.<br>  
6. **Preview box:** To check whether the current selection of parameters for the candidate finding is suitable for your data or needs further fine tuning, you can perform a preview run. Doing so, will perform the analysis routines only on a smaller subset of the data that you can specify in the preview box. To view the event data, which in its raw form is just a list of events, it is converted into a format that is easier for humans to view and interpret (images). You must therefore specify a `display frame time` together with the data selection you would like to display. <br>  
7. **Tab menu:** Here, you can change between different tabs: `Processing` (current tab), `Post-processing` (follow up analysis after a full run), `Localization List` (view, import and export localization tables), `Visualization` (visualize the super-resolved event SMLM data), `Run Info` (info about current run), `Preview run` (preview of finding and fitting performance, useful for parameter tweaking) and `Candidate Preview` (view single candidate clusters and their x,y,t localization results)<br>  

### 2. Perform a Preview Run
Now, change the `Duration` in the preview box to 10000ms and then press `Preview`. This will immediately open the run info. 
By leaving the settings `min` and `max` for x,y empty, you will get a preview for the entire FOV without spatial restrictions. <br>  
Switch to the `Preview run` tab, as soon as the preview run is complete (`UpdateShowPreview ran!` is printed in `Run info` tab). 

![](Quick_Start/6_preview2.png)<br>

Each candidate cluster found by the candidate finding routine will be highlighted with an purple or green box and gets a unique candidate ID that is here displayed in blue. A red box indicates that a failed fit, meaning that no localization could be generated. All localizations are marked as red crosses in the frame where the x,y,t localization is found.
You can use the slider below the image to view all the frames that were created in the preview run.

### 3. Explore the Candidate Preview
By either double-clicking on a candidate in the preview image or switching to the last tab, you can open the `Candidate preview`.  
Change the plot options for the first plot to `3D point cloud of the candidate cluster` and for the second plot to `2D projections of candidate cluster`.   
In a preview run you can also display surrounding events to evaluate if the full candidate cluster is found. To do so, set `show surrounding` to `True` and add a custom x,y and t-padding in the first plot options.  

![](Quick_Start/7_candidate_preview_neg_surrounding.png)<br>  

By the `Previous` and `Next` buttons you can simply click through all the clusters found to evaluate finding and fitting results.   

![](Quick_Start/7_candidate_preview_pos.png)<br>

### 4. Notes of best finding/fitting algorithms and parameters
Knowing which finding/fitting algorithms are best is a currently unsolved problem, and part of why EVE has its expandability. The `Preview run` and `Candidate preview` are designed specifically to give a user the required tools to fully explore the finding and fitting routines. Nonetheless, here are some recommendations to find the best finding and fitting routines at the beginning of a new dataset:

For finding routines, it is recommended to first explore the `Eigen-feature analysis` method via `Preview`, which appears to work robustly. First, set `Maximum Eigenvalue cutoff` to `0`, and setting `Debug Boolean` to `True`: now, a debug figure will be shown. This figure contains a histogram of the maximum Eigenvalue for all events using these settings. Two populations should be discriminatable: low values correspond to single-molecule clusters, while high values correspond to noise. Close the figure, and change the `Number of neighbours` to higher and lower values (e.g. `10` and `40`), and explore whether this discrimination becomes more pronounced or less pronounced. Do the same with `Ratio ms to px`, and note the single value that most clearly separates the two populations. Next, use this value in the `Maximum Eigenvalue cutoff`, and set `Debug Boolean` to `False`. Re-run the `Preview` and inspect the found clusters. If clusters are missed, try increasing `DBSCAN epsilon` and/or decreasing `DBSCAN nr. neighbours`. If clusters are over-represented, try decreasing `DBSCAN epsilon` and/or increasing `DBSCAN nr. neighbours`.

For fitting routines, it is recommended to start with `2D LogGaussian` with `distribution` set to `Hist2d_xy` and `Time fit routine` set to `2D Gaussian (first events)`, if normal, 2-dimensional SMLM is performed.  The `expected width` value should be set to roughly the expected sigma of the Point Spread Function (~150 nm for 561 excitation). Then, in `Candidate preview`, loop through a few candidates to see if the found localization agrees with visual inspection, both in its spatial and temporal profile. If many clusters seem to provide a localization where visually not enough events are present, try either decreasing the `fitting tolerance` (which leads to more fits being discarded), or try changing the `Finding` parameters so noisy clusters are discarded. If visually very good clusters are not localized and show a `No localization generated due to ...`-error, try increasing `fitting tolerance`.<br>  

### 5. Execute a full Run
If you are satisfied with the results of the current selection of finding and fitting parameters, you can start a complete run. To do so, switch back to the `Processing` tab and click `Run`.<br>  
The `Run Info` tab will again open automatically and show additional info regarding the current run, e.g. number of candidates and valid localizations found as well as a number of candidates (absolute and percentage) that was removed during fitting.  

![](Quick_Start/8_full_run.png)<br>

### 6. Visualize the Localization Results
As soon as the full run is completed, you can visualize your results. Therefore, switch to the `Visualisation` tab, select `Histogram_Convolution: Histogram_convolution` and press `Visualise`.  

![](Quick_Start/9_visualization.png)<br>

As you can see the sample data is rather drifty and we can't see our DNA nanorulers yet.  

### 7. Apply a Drift Correction
To apply a drift correction, switch to the `Post-processing` tab.  

#### Under Windows
Windows users can choose between two different drift correction routines. Select `DriftCorr_DME: DriftCorr_RCC` and press `Post processing!`. This will open a pop-up window showing the estimated x,y-drift. Additionally, an entry was added to the `Post-processing history` at the bottom of the tab. By clicking `Restore to before this` you can undo the last post-processing step.  

![](Quick_Start/10_drift_correction_windows.png)<br>

#### Under Linux
For Linux users currently only one method (`Drift correction by entropy minimization`) is working. Select the method and set the `Use ConvHist(Linux)` flag to `True`. Now press `Post processing!` which will open a pop-up window showing the estimated x,y-drift. Additionally, an entry was added to the `Post-processing history` at the bottom of the tab. By clicking `Restore to before this` you can undo the last post-processing step. 

![](Quick_Start/10_drift_correction_linux.png)<br>

As you can see, the drift estimated by the two different drift correction methods is quite similar. You can now switch again to `Visualization` and press `Visualize!` to view the super-resolved event SMLM image.  

![](Quick_Start/11_updated_visualization.png)<br>

### 8. The final Localization List
The drift corrected localization list is not saved automatically. To export the list in `.csv` format, switch to the `Localization List` tab and press `Save CSV`.  

![](Quick_Start/12_localization_list.png)<br>

### 9. Estimate the Localization Precision
To get an estimate on the localzation precision we can make use of the fact, that for each fluorophore blink we measure the on and off-switching. This means, that we have one cluster of positive polarity for the on switching and one cluster of negative cluster polarity for the off switching. By comparing the distances between the corresponding positive and negative localizations we can get our localization precision via a Nearest neighbour analysis (NeNA).  
To do so, we first need to do the polarity matching. Therefore, switch again to the `Post-processing` tab and select `Polarity Matching - match on and off events` and press `PostProcessing!`.   

![](Quick_Start/14_polarity_matching.png)<br>

Now, you can perform a NeNA fit by selecting `Nearest neighbour analysis (NeNA) precision on matched polarities` and pressing `PostProcessing!`. This will open a pop-up window showing the distance distribution of all matched localizations along with the NeNA fit. For our DNA nanoruler and the candidate finding and fitting methods and parameters we get a localization precision of ~10 nm.  

![](Quick_Start/15_NeNa.png)<br>

### 10. Metadata, run and result files

If selected correctly in the `Advanced Settings`, EVE will store a lot of metadata, run and result files automatically. In our "Getting Started" example, three finding and three fitting `.pickle` files are stored containing all, only positive and only negative candidates/localizations. In addition, a `.csv` file with all localizations and a metadata `Runinfo` file are stored.  

![](Quick_Start/17_saved_files.png)<br>

### 11. Command-line interface

All of EVE can also be accessed via the command-line. For a detailed overview look at `EVE_CommandLine.ipynb`.

### 12. Exemplary data
The EVE software is packaged with three exemplary datasets, found in the **Data** folder: DNA-PAINT nanoruler, *E. coli* with endogenous RpoC-mEos3.2 and Nile Red membrane stain. The $\alpha$-tubulin labeled Cos7 dSTORM sample dataset, as well as full-sized versions of the other datasets, can be found at [doi:10.5281/zenodo.13269600](doi:10.5281/zenodo.13269600).  Here are succint instructions for how to fully analyse these datasets:

1. Nanoruler (DNA-PAINT):
   - In EVE, in the processing tab, set the Dataset location to the */Data/Nanoruler/Nanoruler_Small.hdf5* file (via the *File...* button).
   - By using *Settings-Load Specific GUI contents*, load the */Data/Nanoruler/GUI_Settings_Nanoruler.json* file.
   - Open the advanced settings via *Settings-Advanced Settings*, and use *Load Global Settings* to load the */Data/Nanoruler/GUI_Settings_Nanoruler_advancedSettings.json* file.
   - Run the analysis by pressing **Run** in the Processing tab, or alternatively, investigate the findin g/fitting options via the **Preview** tab.
   - After running, the results need to be drift-corrected - either via *Post-processing - Drift correction by entropy minimization [2D]* (default settings), or by loading (*Load stored drift correction DME/RCC*) the */Data/Nanoruler/DME_driftCorrection.npz* file.
   - The results can be visualized by opening the *Visualization* tab and running *Gaussian-blurred with locPrecision sigma*.
   - The results should be similar to the localizations found in */Data/Nanoruler/Nanoruler_Small_FitResults_DriftCorr.csv*.
2. *E. coli* datasets (PALM and PAINT):
   - Two datasets are provided: *Ecoli_RpoC_Small* and *Ecoli_NR_Large*, along with their corresponding GUI settings. Throughout these instructions, these can be interchanged to obtain the results for either dataset.
   - In EVE, in the processing tab, set the Dataset location to the */Data/Ecoli/Ecoli_NR_Small.hdf5* file (via the *File...* button).
   - By using *Settings-Load Specific GUI contents*, load the */Data/Ecoli/GUI_Settings_EColi_NR.json* file.
   - Open the advanced settings via *Settings-Advanced Settings*, and use *Load Global Settings* to load the */Data/Ecoli/GUI_Settings_EColi_NR_advancedSettings.json* file.
   - Run the analysis by pressing **Run** in the Processing tab, or alternatively, investigate the findin g/fitting options via the **Preview** tab.
   - The results can be visualized by opening the *Visualization* tab and running *Gaussian-blurred with locPrecision sigma*.
   - These results should be similar to the localizations found in */Data/EColi/NEcoli_NR_Small_FitResults.csv*.
2. *$\alpha$-tubulin* in Cos-7 cell (dSTORM):
   - Download the dataset from [doi:10.5281/zenodo.13269600](doi:10.5281/zenodo.13269600).
   - In EVE, in the processing tab, set the Dataset location to the */Data/aTubulin/aTubulin_Small.hdf5* file (via the *File...* button).
   - By using *Settings-Load Specific GUI contents*, load the */Data/aTubulin/GUI_Settings_aTubulin.json* file.
   - Open the advanced settings via *Settings-Advanced Settings*, and use *Load Global Settings* to load the */Data/aTubulin/GUI_Settings_aTubulin_advancedSettings.json* file.
   - Run the analysis by pressing **Run** in the Processing tab, or alternatively, investigate the findin g/fitting options via the **Preview** tab.
   - After running, the results need to be drift-corrected - either via *Post-processing - Drift correction by entropy minimization [2D]* (default settings), or by loading (*Load stored drift correction DME/RCC*) the */Data/aTubulin/DME_driftCorrection.npz* file.
   - The results can be visualized by opening the *Visualization* tab and running *Gaussian-blurred with locPrecision sigma*.
   - These results should be similar to the localizations found in */Data/aTubulin/aTubulin_Small_FitResults.csv*.