
from Utils import utilsHelper

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "GaussianKernel_fixedSigma": {
            "required_kwargs": [
                {"name": "px_size", "description": "Visualisation px size","default":10,"type":int,"display_text":"Pixel size (in nm)"},
                {"name": "sigma", "description": "Gaussian sigma value in pixel","default":20.,"type":float,"display_text":"Sigma (in nm)"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Draws Gaussians"
        },
        "GaussianKernel_locPrec": {
            "required_kwargs": [
                {"name": "px_size", "description": "Visualisation px size","default":10,"type":int,"display_text":"Pixel size (in nm)"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Draws Gaussians with sigma based on loc. prec (del_x, del_y)"
        }
    }



import numpy as np

def GaussianKernel_fixedSigma(resultArray,settings,**kwargs):

    from .dme.dme.native_api import NativeAPI
    
    # sigma = float(kwargs['FWHM'])/(2*np.sqrt(2*np.log(2)))*16 #not sure why times 16, but seems to be right
    sigma = float(kwargs['sigma'])  
    px_size = float(kwargs['px_size'])
    use_cuda= settings['UseCUDA']['value']>0
    zoom = (float(settings['PixelSize_nm']['value']))/px_size
    
    #Obtain the localizations from the resultArray
    resultArray=resultArray.dropna()
    xy = np.column_stack((resultArray['x'].values-min(resultArray['x']),resultArray['y'].values-min(resultArray['y'])))
    #Convert it to pixel-units
    xy/=(float(settings['PixelSize_nm']['value']))
    
    
    rendersize = int(np.max(xy))
    area = np.array([rendersize,rendersize])
    
    imgshape = np.ceil(area*zoom).astype(int)
    image = np.zeros((0, *imgshape))

    with NativeAPI(use_cuda) as dll:
        img = np.zeros(imgshape,dtype=np.float32)
        
        spots = np.zeros((len(xy), 5), dtype=np.float32)
        spots[:, 0] = xy[:,0] * zoom
        spots[:, 1] = xy[:,1] * zoom
        spots[:, 2] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
        spots[:, 3] = sigma/(float(settings['PixelSize_nm']['value'])) * zoom
        spots[:, 4] = 1
        

        image = dll.DrawGaussians(img, spots)
        
    #Scale should be the scale of pixel - to - um. E.g. a scale of 0.01 means 100 pixels = 1 um
    scale = (max(xy[:,0])*(float(settings['PixelSize_nm']['value']))/1000)/np.shape(image)[0]
    import logging
    logging.info('Gaussian image created!')
    
    
    return image, scale


def GaussianKernel_locPrec(resultArray,settings,**kwargs):
    from .dme.dme.native_api import NativeAPI
    px_size = float(kwargs['px_size'])
    use_cuda= settings['UseCUDA']['value']>0
    zoom = (float(settings['PixelSize_nm']['value']))/px_size
    
    import logging
    #Check if we have the del_x and del_y columns:
    if 'del_x' not in resultArray or 'del_y' not in resultArray:
        logging.error('No del_x or del_y column in resultArray, breaking off')
        return

    #check if there are nans in the del_x or del_y column, and if so, change them to the mean value of all other del_x, del_y values:
    if resultArray['del_x'].isna().values.any() or resultArray['del_y'].isna().values.any():
        n_del = resultArray['del_x'].isna().sum() + resultArray['del_y'].isna().sum()
        #Get the mean only from the non-nan entries:
        resultArrayMeanCalc = resultArray[(resultArray['del_x'].notna()) & (resultArray['del_y'].notna())]
        mean_del_x = resultArrayMeanCalc['del_x'].mean()
        mean_del_y = resultArrayMeanCalc['del_y'].mean()
        resultArray['del_x'] = resultArray['del_x'].fillna(mean_del_x)
        resultArray['del_y'] = resultArray['del_y'].fillna(mean_del_y)
        logging.warning(f"{n_del} NaN values in del_x or del_y column, replaced with their mean values")
    
    #the same but then checking if any of the values are zero rather than nans:
    if (resultArray['del_x'] == 0).any() or (resultArray['del_y'] == 0).any():
        n_zero = (resultArray['del_x'] == 0).sum() + (resultArray['del_y'] == 0).sum()
        #Get the mean only from the non-zero entries:
        resultArrayMeanCalc = resultArray[(resultArray['del_x'] != 0) & (resultArray['del_y'] != 0)]
        mean_del_x = resultArrayMeanCalc['del_x'].mean()
        mean_del_y = resultArrayMeanCalc['del_y'].mean()
        resultArray['del_x'] = resultArray['del_x'].replace(0,mean_del_x)
        resultArray['del_y'] = resultArray['del_y'].replace(0,mean_del_y)
        logging.warning(f"{n_zero} zero values in del_x or del_y column, replaced with their mean values")

    #Obtain the localizations from the resultArray
    resultArray=resultArray.dropna()
    xy = np.column_stack((resultArray['x'].values-min(resultArray['x']),resultArray['y'].values-min(resultArray['y'])))
    #Convert it to pixel-units
    xy/=(float(settings['PixelSize_nm']['value']))

    rendersize = int(np.max(xy))
    area = np.array([rendersize,rendersize])

    imgshape = np.ceil(area*zoom).astype(int)
    image = np.zeros((0, *imgshape))

    with NativeAPI(use_cuda) as dll:
        img = np.zeros(imgshape,dtype=np.float32)

        spots = np.zeros((len(xy), 5), dtype=np.float32)
        spots[:, 0] = xy[:,0] * zoom
        spots[:, 1] = xy[:,1] * zoom
        spots[:, 2] = resultArray['del_x'].values/(float(settings['PixelSize_nm']['value'])) * zoom
        spots[:, 3] = resultArray['del_y'].values/(float(settings['PixelSize_nm']['value'])) * zoom
        spots[:, 4] = 1

        image = dll.DrawGaussians(img, spots)
    
    #Scale should be the scale of pixel - to - um. E.g. a scale of 0.01 means 100 pixels = 1 um
    scale = (max(xy[:,0])*(float(settings['PixelSize_nm']['value']))/1000)/np.shape(image)[0]
    return image, scale
