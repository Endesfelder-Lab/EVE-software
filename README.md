# Eve - General-purpose software for eve-SMLM localization
![](Eve.png)
## About Eve
Eve is a user-interfaced software package that provides a plethora of options to localize emitters from single molecule localization microscopy (SMLM) experiments performed on event-based sensor (eveSMLM).

Event-based data differs fundamentally from conventional camera images. Unlike traditional sensors, event-based sensors only capture intensity changes, registering them as either positive (when intensity surpasses a predefined threshold) or negative events (when intensity drops below a predefined threshold). As a result, only a list of x and y pixel coordinate pairs is stored together with the detected event polarities and timestamps.

Eve is designed to quickly and directly process and analyse event-based single molecule data without the need to convert event lists back to image frames, followed by traditional SMLM data analysis. 
The event-based data analysis is divided into two main parts:
1. **Candidate Finding:** The complete event-list is searched for characteristic event clusters that are generated by blinking fluorophores. Potential candidate clusters are then extracted and returned for further processing.
2. **Candidate Fitting:** The x,y,(z),t-localization is determined for each candidate cluster.

Eve allows flexible combinations of different finding and fitting routines to optimize the localization results for the specific event-based data. Besides a variety of different finding and fitting algorithms, Eve also offers various preview options, visualisation tools and post-processing functions.

Eve is written in Python and structured in such a way, that is easy to implement and add new functionalties in form of new finding, fitting routines, etc. 
## Version
The `main` branch contains the latest running version of the software. The latest developments are stored in the `develop` branch.
## How to install and run Eve
This software was tested under Linux (Ubuntu 20.04) and Windows 10. Besides the drift correction module `DriftCorr_DME: DriftCorr_entropyMin` and the Gaussian visualization methods (`Gaussian_display: GaussianKernal_fixedSigma`, `Gaussian_display: GaussianKernal_locPrec`) that use pre-compiled dll-files and are therefore only running on Windows, everthing is running under Linux and Windows.

The software requires Python 3.9.18.
### Installation instructions
1. First download Eve or clone it to your local repository using git:
    ```bash
    git clone https://github.com/Endesfelder-Lab/Eve.git
    ```
2. Install required python dependencies
    #### With `virtualenv`
    Replace `PYTHON_PATH` by your python path, e.g. `/usr/bin` and `ENVIRONMENT_PATH` by the path to the virtual environments on your machine and follow the instructions below:
    ```bash
    virtualenv -p PYTHON_PATH/python3.9.18 ENVIRONMENT_PATH/Eve
    source ENVIRONMENT_PATH/Eve/bin/activate
    pip install -r requirements.txt
    ```
    #### With `conda`
   Simply follow the instructions below, this will also install the correct python version on your system:
    ```bash
    cd Eve
    conda env create -f environment_eve.yml
    conda activate Eve
    ```
3. Optional:\
Eve can read and process event-based data in `.npy` and `.hdf5` format. Additionally the `.raw` format of [Prophesee](https://www.prophesee.ai/) can be used. If you have `.raw` data that you want to analyze you need to install the [Metavision SDK from Prophesee](https://docs.prophesee.ai/stable/installation/index.html) in addition.
### Running Eve
To open the graphical user interface and run Eve, first activate the python environment you created during the installation. Then run `GUI.py` with Python.
## Quick Start Guide
### Set up the Processing Tab
Running `GUI.py` will open Eve's graphical user interface which you can see on the right side of the figure.

![](Quick_Start/1_Setting_up_GUI_new.png)

The main window has 7 major parts that are marked with red boxes and are described in more detail in the following.
1. **Menu bar:** By clicking on `Settings`, you can open the `Advanced settings`, save the current GUI configuration and settings or load a specific GUI configuration. Under `Utilities` you will find some additional functionalities to pre-process the raw event data files before processing them with Eve. \
Now, open the advaced settings, and change the settings in accordance with the image below.\
   \
   ![](Quick_Start/3_advanced_settings.png)
2. **Data to analyse:** Here, you specify the data that will be analysed in the following. You can either select a single file (in `.npy`, `.hdf5` or `.raw` format) or a folder. If you select a folder all files in the folder will be analysed one after another.

   The folder `Data` contains an event-based acquisition of a DNA nanoruler (`DNAPAINT.hdf5`) which we will use in this tutorial. Fill the path entry field `Dataset localtion:` with the corresponding path to the nanoruler dataset.

   In the `Data selection` box, you can now further specify which parts of the data should be analysed and how. You have different options for `Polarity`, `Time` and `Position`. Choose `Pos and neg seperately` as `Polarity` option while leaving the remaining settings unchanged. Thereby, you simply load the all events without temporal or spatial constraints. By selecting `Pos and neg seperately` all subsequent analysis steps will be run on the positive and negative events distinctly. As you can see, the GUI has adapted to your selection and you can now choose finding and fitting routines for positive and negative events seperatly.\
   \
![](Quick_Start/4_change_and_save_GUI_contents.png)
3. **Candidate Finding routine:** Here, you can select among different candidate finding routines. Choose `Eigen-feature analysis`, both for positive and negative events and change the settings as shown in the screenshot above.
4. **Candidate Fitting routine:** Here, you can specify which fitting routines you want to use to get localizations for each candidate, cluster. Choose `2D Gaussian` again for both polarities and modify all other parameters as shown in the screenshot above. \
Everything is now ready for the first run. Before you start the first run, save the GUI settings (`Settings -> Save GUI contents`). When you open Eve again, the last saved GUI settings will be loaded automatically.
5. **Run box:** When you click run, a full run will be executed.
6. **Preview box:** To check whether the current selection of parameters for the candidate finding is suitable for your data or needs further fine tuning, you can perform a preview run. Doing so, will perform the analysis routines only on a smaller subset of the data that you can specify in the preview box. To view the event data, which in its raw form is just a list of events, it is converted into a format that is easier for humans to view and interpret (images). You must therefore specify a `display frame time` together with the data selection you would like to display. 
7. **Tab menu:** Here, you can change between different tabs: `Processing` (current tab), `Post-processing` (follow up analysis after a full run), `Localization List` (view, import and export localization tables), `Visualization` (visualize the super-resolved event SMLM data), `Run Info` (info about current run), `Preview run` (preview of finding and fitting performance, useful for parameter tweaking) and `Candidate Preview` (view single candidate clusters and their x,y,t localization results)
### Perform a Preview Run
Now, change the `Duration` in the preview box to 10000ms and then press `Preview`. This will immediately open the run info. 
By leaving the settings `min` and `max` for x,y empty, you will get a preview for the entire FOV without spatial restrictions. \
Switch to the `Preview run` tab, as soon as the preview run is complete (`UpdateShowPreview ran!` is printed in `Run info` tab). 

![](Quick_Start/6_preview2.png)

Each candidate cluster found by the candidate finding routine will be highlighted with an orange or red box and gets a unique candidate ID that is here displayed in blue. A red box indicates that a failed fit, meaning that no localization could be generated. All localizations are marked as red crosses in the frame where the x,y,t localization is found.
You can use the slider below the image to view all the frames that were created in the preview run.
### Explore the Candidate Preview
By either double-clicking on a candidate in the preview image or switching to the last tab, you can open the `Candidate preview`.
Change the plot options for the first plot to `3D point cloud of the candidate cluster` and for the second plot to `2D projections of candidate cluster`. 
In a preview run you can also display surrounding events to evaluate if the full candidate cluster is found. To do so, set `show surrounding` to `True` and add a custom x,y and t-padding in the first plot options.

![](Quick_Start/7_candidate_preview_neg_surrounding.png)

By the `Previous` and `Next` buttons you can simply click through all the clusters found to evaluate finding and fitting results. 

![](Quick_Start/7_candidate_preview_pos.png)

### Excute a full Run


![](Quick_Start/8_full_run.png)

### Visualize the Localization Results

![](Quick_Start/9_visualization.png)

### Apply a Drift Correction

![](Quick_Start/10_drift_correction_linux.png)![](Quick_Start/10_drift_correction_windows.png)

![](Quick_Start/11_updated_visualization.png)

### The final Localization List

![](Quick_Start/12_localization_list.png)

### Estimate the Localization Precision

![](Quick_Start/14_polarity_matching.png)

![](Quick_Start/15_NeNa.png)

### All files stored for one run

![](Quick_Start/17_saved_files.png)
