import inspect
try:
    from eve_smlm.Utils import utilsHelper
    from eve_smlm.EventDistributions import eventDistributions
    from eve_smlm.TemporalFitting import timeFitting
except ImportError:
    from Utils import utilsHelper
    from EventDistributions import eventDistributions
    from TemporalFitting import timeFitting
import pandas as pd
import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import least_squares
from scipy import ndimage
from scipy.optimize import minimize
from scipy import linalg
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
import logging

from joblib import Parallel, delayed
import multiprocessing

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "RadialSym2D": {
            "time_kwarg" : {"base": "TemporalFits", "description": "Temporal fitting routine to get time estimate.", "default_option": "LognormCDFFirstEvents_weighted"},
            "required_kwargs": [
            ],
            "optional_kwargs": [
            ],
            "help_string": "Determines localization parameters via radial symmetry.",
            "display_name": "Radial Symmetry 2D"
        },
        "RadialSym3D": {
            "required_kwargs": [
                {"name": "time_bin_width", "display_text":"temporal bin width", "description": "Temporal bin width for 3d histogramming (in ms)","default":10.},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Determines localization parameters via 3d radial symmetry.",
            "display_name": "Radial Symmetry 3D"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def radialcenter(distfunc, candidateID, candidate, time_fit, pixel_size):
    I = distfunc(candidate['events']).dist2D
    # Number of grid points
    Ny, Nx = I.shape

    # grid coordinates are -n:n, where Nx (or Ny) = 2*n+1
    # grid midpoint coordinates are -n+0.5:n-0.5;
    # The two lines below replace
    #    xm = repmat(-(Nx-1)/2.0+0.5:(Nx-1)/2.0-0.5,Ny-1,1);
    # and are faster (by a factor of >15 !)
    # -- the idea is taken from the repmat source code
    xm_onerow = np.arange(-(Nx-1)/2.0+0.5, (Nx-1)/2.0-0.5+1)
    xm = np.tile(xm_onerow, (Ny-1, 1))
    # similarly replacing
    #    ym = repmat((-(Ny-1)/2.0+0.5:(Ny-1)/2.0-0.5)', 1, Nx-1);
    ym_onecol = np.arange(-(Ny-1)/2.0+0.5, (Ny-1)/2.0-0.5+1)[:, np.newaxis] #Note that y increases "downward"
    ym = np.tile(ym_onecol, (1, Nx-1))

    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    # Note that y increases "downward" (increasing row number) -- we'll deal
    # with this when calculating "m" below.
    dIdu = I[:-1, 1:] - I[1:, :-1]
    dIdv = I[:-1, :-1] - I[1:, 1:]

    # Smoothing --
    h = np.ones((3, 3)) / 9.0  #simple 3x3 averaging filter
    fdu = convolve(dIdu, h, mode='constant')
    fdv = convolve(dIdv, h, mode='constant')
    dImag2 = fdu**2 + fdv**2  # gradient magnitude, squared

    # Slope of the gradient .  Note that we need a 45 degree rotation of 
    # the u,v components to express the slope in the x-y coordinate system.
    # The negative sign "flips" the array to account for y increasing
    # "downward" 
    m = -(fdv + fdu) / (fdu - fdv)

    # *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
    # zero.  In this case, replace with the un-smoothed gradient.
    NNanm = np.sum(np.isnan(m))
    if NNanm > 0:
        unsmoothm = (dIdv + dIdu) / (dIdu - dIdv)
        m[np.isnan(m)] = unsmoothm[np.isnan(m)]

    # If it's still NaN, replace with zero. (Very unlikely.)
    NNanm = np.sum(np.isnan(m))
    if NNanm > 0:
        m[np.isnan(m)] = 0



    # Almost as rarely, an element of m can be infinite if the smoothed u and v
    # derivatives are identical.  To avoid NaNs later, replace these with some
    # large number -- 10x the largest non-infinite slope.  The sign of the
    # infinity doesn't matter
    try:
        m[np.isinf(m)] = 10 * np.max(m[~np.isinf(m)])
    except:
        # if this fails, it's because all the elements are infinite.  Replace
        # with the unsmoothed derivative.  There's probably a more elegant way
        # to do this.
        m = (dIdv + dIdu) / (dIdu - dIdv)



    # Shorthand "b", which also happens to be the
    # y intercept of the line of slope m that goes through each grid midpoint
    b = ym - m * xm

    # Weighting: weight by square of gradient magnitude and inverse 
    # distance to gradient intensity centroid.
    sdI2 = np.sum(dImag2)
    xcentroid = np.sum(np.sum(dImag2 * xm)) / sdI2
    ycentroid = np.sum(np.sum(dImag2 * ym)) / sdI2
    w = dImag2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # least-squares minimization to determine the translated coordinate
    # system origin (xc, yc) such that lines y = mx+b have
    # the minimal total distance^2 to the origin:
    # See function lsradialcenterfit (below)
    xc, yc = lsradialcenterfit(m, b, w)



    
    # Return output relative to upper left coordinate
    xc += (Nx - 1) / 2.0
    yc += (Ny - 1) / 2.0

    # A rough measure of the particle width.
    # Not at all connected to center determination, but may be useful for tracking applications; 
    # could eliminate for (very slightly) greater speed
    Isub = I - np.min(I)
    px, py = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    xoffset = px - xc
    yoffset = py - yc
    r2 = xoffset**2 + yoffset**2
    sigma = np.sqrt(np.sum(np.sum(Isub * r2)) / np.sum(Isub)) / 2  # second moment is 2*Gaussian width

    mean_polarity = candidate['events']['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
    opt = [xc, yc]
    t, del_t, t_fit_info, opt_t = time_fit(candidate['events'], opt)
    xc = (xc + np.min(candidate['events']['x']))*pixel_size # in nm
    yc = (yc + np.min(candidate['events']['y']))*pixel_size # in nm

    loc_df = pd.DataFrame({'candidate_id': candidateID, 'x': xc, 'y': yc, 'p': p, 't': t, 'del_t': del_t, 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][1], 'y_dim': candidate['cluster_size'][0], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': t_fit_info}, index=[0])
    return loc_df


def lsradialcenterfit(m, b, w):
    # Least squares solution to determine the radial symmetry center

    # inputs m, b, w are defined on a grid
    # w are the weights for each point
    wm2p1 = w / (m**2 + 1)
    sw = np.sum(np.sum(wm2p1))
    smmw = np.sum(np.sum(m**2 * wm2p1))
    smw = np.sum(np.sum(m * wm2p1))
    smbw = np.sum(np.sum(m * b * wm2p1))
    sbw = np.sum(np.sum(b * wm2p1))
    det = smw**2 - smmw * sw
    xc = (smbw * sw - smw * sbw) / det  # relative to image center
    yc = (smbw * smw - smmw * sbw) / det  # relative to image center

    return xc, yc

class radysm:
    def __init__(self, events, distfunc):
        self.dist = distfunc(events).dist2D
        self.ylim, self.xlim = self.dist.shape
        self.shift = np.array([(self.xlim-1.)/2., (self.ylim-1.)/2.])
        self.xmean = np.average(np.arange(1,self.xlim+1), weights=np.nansum(self.dist, axis=0))-1.-self.shift[0]
        self.ymean = np.average(np.arange(1,self.ylim+1), weights=np.nansum(self.dist, axis=1))-1.-self.shift[1]
        self.mesh = self.meshgrid()
        self.slope, self.magnitude = self.gradient()
        self.weights = self.weight()
        self.fit_info = ''

    def meshgrid(self):
        # x = np.arange(self.xlim)
        # y = np.arange(self.ylim)
        x = np.linspace(-(self.xlim-2.)/2., (self.xlim-2.)/2., self.xlim-1)
        y = np.linspace(-(self.ylim-2.)/2., (self.ylim-2.)/2., self.ylim-1)
        X,Y = np.meshgrid(x,y)
        return X.ravel(),Y.ravel()

    def gradient(self):
        # Calculate derivatives along 45-degree shifted coordinates (u and v)
        dIdu = self.dist[:-1, 1:] - self.dist[1:, :-1]
        dIdv = self.dist[:-1, :-1] - self.dist[1:, 1:]

        # Smoothing
        h = np.ones((3, 3)) / 9.0  #simple 3x3 averaging filter
        fdu = convolve(dIdu, h, mode='constant')
        fdv = convolve(dIdv, h, mode='constant')
        
        # Slope of the gradient.  Note that we need a 45 degree rotation of 
        # the u,v components to express the slope in the x-y coordinate system.
        # The negative sign "flips" the array to account for y increasing
        # "downward" 
        m = -(fdv + fdu) / (fdu - fdv)

        # *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
        # zero.  In this case, replace with the un-smoothed gradient.
        NNanm = np.sum(np.isnan(m))
        if NNanm > 0:
            unsmoothm = (dIdv + dIdu) / (dIdu - dIdv)
            m[np.isnan(m)] = unsmoothm[np.isnan(m)]

        # If it's still NaN, replace with zero. (Very unlikely.)
        NNanm = np.sum(np.isnan(m))
        if NNanm > 0:
            m[np.isnan(m)] = 0

        # Almost as rarely, an element of m can be infinite if the smoothed u and v
        # derivatives are identical.  To avoid NaNs later, replace these with some
        # large number -- 10x the largest non-infinite slope.  The sign of the
        # infinity doesn't matter
        try:
            m[np.isinf(m)] = 10 * np.nanmax(m[~np.isinf(m)])
        except:
            # if this fails, it's because all the elements are infinite.  Replace
            # with the unsmoothed derivative.  
            m = (dIdv + dIdu) / (dIdu - dIdv)

        # Weighting: weight by square of gradient magnitude and inverse 
        # distance to gradient intensity centroid.
        dImag2 = fdu**2 + fdv**2  # gradient magnitude, squared
        return m.ravel(), dImag2.ravel()

    def weight(self):
        # Weighting: weight by square of gradient magnitude and inverse 
        # distance to gradient intensity centroid.
        sdI2 = np.sum(self.magnitude)
        xcentroid = np.sum(np.sum(self.magnitude * self.mesh[0])) / sdI2
        ycentroid = np.sum(np.sum(self.magnitude * self.mesh[1])) / sdI2
        w = self.magnitude / np.sqrt((self.mesh[0] - xcentroid)**2 + (self.mesh[1] - ycentroid)**2)
        return w
    
    def p0(self):
        p0 = [self.xmean, self.ymean]
        return p0
    
    def bounds(self):
        bounds = ([-0.5-self.shift[0], -0.5-self.shift[1]], [self.xlim-0.5-self.shift[0], self.ylim-0.5-self.shift[1]]) # allow borders of pixels
        return bounds
    
    def distance(self, xy_c): # function to minimize with least-squares
        X, Y = self.mesh
        d = ((Y-xy_c[1])-self.slope*(X-xy_c[0]))/np.sqrt(self.slope**2+1.)*np.sqrt(self.weights)
        return d
        
    def __call__(self, candidate, time_fit, pixel_size, candidateID, **kwargs):
        try:
            res = least_squares(self.distance, self.p0(), bounds=self.bounds(), **kwargs)
            popt = res.x + self.shift
            U, s, Vh = linalg.svd(res.jac, full_matrices=False)
            tol = np.finfo(float).eps*s[0]*max(res.jac.shape)
            w = s > tol
            cov = (Vh[w].T/s[w]**2) @ Vh[w]  # robust covariance matrix
            perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted parameters
        except RuntimeError as warning:
            self.fit_info += 'RuntimeError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except ValueError as warning:
            self.fit_info += 'ValueError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except OptimizeWarning as warning:
            self.fit_info += 'OptimizeWarning: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])

        if self.fit_info != '':
            x = np.nan
            y = np.nan
            del_x = np.nan
            del_y = np.nan
            t = np.nan
            del_t = np.nan
        else: 
            x = (popt[0]+np.min(candidate['events']['x']))*pixel_size # in nm
            y = (popt[1]+np.min(candidate['events']['y']))*pixel_size # in nm
            del_x = perr[0]*pixel_size # in nm
            del_y = perr[1]*pixel_size # in nm
            t, del_t, t_fit_info, opt_t = time_fit(candidate['events'], popt) # t, del_t in ms
            if t_fit_info != '':
                self.fit_info = t_fit_info
        mean_polarity = candidate['events']['p'].mean()
        p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2
        loc_df = pd.DataFrame({'candidate_id': candidateID, 'x': x, 'y': y, 'del_x': del_x, 'del_y': del_y, 'p': p, 't': t, 'del_t': del_t, 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][1], 'y_dim': candidate['cluster_size'][0], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': self.fit_info}, index=[0])
        return loc_df

# perform localization for part of candidate dictionary
def localize_canditates2D(i, candidate_dic, distfunc, time_fit, pixel_size):
    logging.info('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    fails = pd.DataFrame()  # Initialize fails as an empty DataFrame
    fail = pd.DataFrame()
    for candidate_id in list(candidate_dic):
        radsym_fit = radysm(candidate_dic[candidate_id]['events'], distfunc)
        localization = radsym_fit(candidate_dic[candidate_id], time_fit, pixel_size, candidate_id) # radialcenter(distfunc, candidate_id, candidate_dic[candidate_id], time_fit, pixel_size)
        localizations.append(localization)
        if localization['fit_info'][0] != '':
            fail['candidate_id'] = localization['candidate_id']
            fail['fit_info'] = localization['fit_info'].str.split(':').str[0][0]
            fails = pd.concat([fails, fail], ignore_index=True)
    if localizations == []:
        localizations = pd.DataFrame()
    else:
        localizations = pd.concat(localizations, ignore_index=True)
    logging.info('Localizing PSFs (thread '+str(i)+') done!')
    return localizations, fails

# helper function to calculate distance in 3d radial symmetry algorithm
def distance(P, Gx ,Gy ,Gz ,w ):
    '''
    Outputs the sum of the perpendicular distances between point P(x,y,z) and lines with 
    direction vectors (Gix, Giy, Giz), weighted by the gradient magnitude squared.

    Parameters
    ----------
    P : tuple
        (x,y,z) coordinates of point P

    Gx : 3D array containing gradient calculated along x axis
    Gy : 3D array containing gradient calculated along y axis
    Gz : 3D array containing gradient calculated along z axis

    w : 3D weights to apply to the distances

    Note: All 3D arrays are of the same shape

    Returns
    -------
    sum_dsqrw : float
        Sum of perpendicular distances squared weighted by the gradient magnitude squared
    '''

    # We will now reshape the arrays to 2D for easier calculation and perform the same operation
    

    #For consistency we will index the arrays in the same way as the original code, not the default 'C' indexing
    x_coords, y_coords, z_coords = np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]), np.arange(w.shape[2]), indexing='ij')

    #First we create the distance array
    APx = x_coords - P[0]
    APy = y_coords - P[1]
    APz = z_coords - P[2]

    #Now we format the arrays in the same 2D way
    AP = np.vstack([APx.ravel(), APy.ravel(), APz.ravel()]).T
    GP = np.vstack([Gx.ravel(), Gy.ravel(), Gz.ravel()]).T

    #Now we take the cross product
    AP_cross_G = np.cross(AP, GP)

    #Now we calculate the magnitude of the cross product as well as the magnitude of the Gradient
    AP_cross_G_mag = np.linalg.norm(AP_cross_G, axis=1)
    G_mag = np.linalg.norm(GP, axis=1)

    #Now we calculate the perpendicular distance
    valid_index = np.where(G_mag != 0)
    D = np.zeros(len(G_mag))
    D[valid_index] = AP_cross_G_mag[valid_index] / G_mag[valid_index]

    #Now we square the distance and multiply by the weight
    sum_dsqrw = np.sum(D**2 * w.ravel())

    return sum_dsqrw

# calculate radial symmetry center in 3d
def radialcenter3d(candidateID, candidate, time_bin_width, pixel_size):
    '''
    Calculates the event-based radial center of a 3D image I

    Parameters
    ----------
    I : 3D array
        The image to be analyzed

    Returns
    -------
    center : array
        x, y, z coordinates of the radial center
    
    errors : array
        x, y, z center coordinate errors

    '''
    # get 3d event histogram
    I = eventDistributions.Hist3d_xyt(candidate['events'], t_bin_width=time_bin_width).dist3D

    # Number of grid points
    Nx, Ny, Nz = I.shape # is not transposed (see eventDistributions.Hist3d_xyt)

    # Grid Coorinates
    xm_onerow = np.arange(0, Nx)
    ym_onecol = np.arange(0, Ny) #Note that y increases "downward"
    zm_onetile = np.arange(0, Nz)

    # Using a 3D meshgrid to create the coordinates
    xm, ym, zm = np.meshgrid(xm_onerow, ym_onecol, zm_onetile, indexing='ij')
    ### The og code created a meshgrid corresponding to xy indexing but idk if that's what we still want in 3D
    #print(xm.shape, ym.shape, zm.shape)

    #Compute Derivatives
    #Using Central Difference instead of Sobel Operator, however the arrays must be big enough, otherwise Sobel must be used
    if np.shape(I)[0] > 1 and np.shape(I)[1] > 1 and np.shape(I)[2] > 1:
        Gx = np.gradient(I, axis=0)
        Gy = np.gradient(I, axis=1)
        Gz = np.gradient(I, axis=2)
    else:
        Gx = ndimage.sobel(I, axis=0)
        Gy = ndimage.sobel(I, axis=1)
        Gz = ndimage.sobel(I, axis=2)
   
    # Smoothing to reduce the effect of noise
    h = np.ones((3, 3, 3)) / 27.0 # 3x3x3 averaging filter
    Gx = ndimage.convolve(Gx, h, mode='constant')
    Gy = ndimage.convolve(Gy, h, mode='constant')
    Gz = ndimage.convolve(Gz, h, mode='constant')

    # Calculating the magnitude of the gradient
    r = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    rsqr = r**2

    # Weighting: square of gradient magnitude divided by distance to approximate centre
    rsqr_sum = np.sum(rsqr)
    xcentroid = np.sum(rsqr * xm) / rsqr_sum
    ycentroid = np.sum(rsqr * ym) / rsqr_sum
    zcentroid = np.sum(rsqr * zm) / rsqr_sum
    print(xcentroid, ycentroid, zcentroid)
    w = rsqr / (np.sqrt((xm-xcentroid)**2 + (ym-ycentroid)**2 + (zm-zcentroid)**2))

    #Instead of looping through every voxel, instead reshape the 3D arrays as 1D array in the function   

    # Minimizing the distance from each gradient line to a point P (optimized P coords will be xc, yc, zc)
    
    # Using the estimated centroids to define an initial centre location guess
    initial_guess = [xcentroid, ycentroid, zcentroid]
    # Ensuring the optimized center is within the bounds of the image
    bounds = [(-0.5, I.shape[0]-0.5), (-0.5, I.shape[1]-0.5), (0, I.shape[2])]

    result = minimize(distance, initial_guess, args=(Gx,Gy,Gz,w), method='L-BFGS-B', bounds=bounds, options={'ftol' : 0.001})
    # center = result.x
    # success = result.success
    # message = result.message
    fit_info = ''
    if not result.success:
        print(result.message)
        print(result.hess_inv)
        center = np.full(3, np.nan)
        error = np.full(3, np.nan)
        fit_info = 'MinimizeError: ' + result.message
    else:
        center = result.x
        print(center)
        center[0] = (center[0] + np.min(candidate['events']['x']))*pixel_size # xc in nm
        center[1] = (center[1] + np.min(candidate['events']['y']))*pixel_size # yc in nm
        center[2] = (center[2]*time_bin_width + np.min(candidate['events']['t'])*1e-3) # tc in ms
        # Calculate errors for L-BFGS-B minimization
        if result.hess_inv is not None:
            error = np.sqrt(np.diag(result.hess_inv.todense()))
            error[0] *= pixel_size
            error[1] *= pixel_size
            error[2] *= time_bin_width
        else:
            print("Optimization failed to provide Hessian matrix. No error estimates available.")
            error = np.full(3, np.nan)

    mean_polarity = candidate['events']['p'].mean()
    p = int(mean_polarity == 1) + int(mean_polarity == 0) * 0 + int(mean_polarity > 0 and mean_polarity < 1) * 2

    loc_df = pd.DataFrame({'candidate_id': candidateID, 'x': center[0], 'y': center[1], 'del_x': error[0], 'del_y': error[1], 'p': p, 't': center[2], 'del_t': error[2], 'N_events': candidate['N_events'], 'x_dim': candidate['cluster_size'][1], 'y_dim': candidate['cluster_size'][0], 't_dim': candidate['cluster_size'][2]*1e-3, 'fit_info': fit_info}, index=[0])
    return loc_df

# perform localization for part of candidate dictionary with 3d radial symmetry
def localize_canditates3D(i, candidate_dic, time_bin_width, pixel_size):
    logging.info('Localizing PSFs (thread '+str(i)+')...')
    localizations = []
    fails = pd.DataFrame()  # Initialize fails as an empty DataFrame
    fail = pd.DataFrame()
    for candidate_id in list(candidate_dic):
        localization = radialcenter3d(candidate_id, candidate_dic[candidate_id], time_bin_width, pixel_size)
        localizations.append(localization)
        if localization['fit_info'][0] != '':
            fail['candidate_id'] = localization['candidate_id']
            fail['fit_info'] = localization['fit_info'].str.split(':').str[0][0]
            fails = pd.concat([fails, fail], ignore_index=True)
    if localizations == []:
        localizations = pd.DataFrame()
    else:
        localizations = pd.concat(localizations, ignore_index=True)
    logging.info('Localizing PSFs (thread '+str(i)+') done!')
    return localizations, fails

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

# 2D radial symmetry
def RadialSym2D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    # maybe add a smoothing parameter here

    # Load the optional kwargs

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm

    distfunc = getattr(eventDistributions, 'Hist2d_xy')
    time_fit = getattr(timeFitting, kwargs['time_kwarg'])()

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates2D)(i, data_split[i], distfunc, time_fit, pixel_size) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list, ignore_index=True)

    fail_list = [res[1] for res in RES]
    fails = pd.concat(fail_list, ignore_index=True)
    
    # Fit performance information
    radSym_fit_info = utilsHelper.info(localizations, fails)

    return localizations, radSym_fit_info

# 3D radial symmetry
def RadialSym3D(candidate_dic,settings,**kwargs):
    # Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    logging.info("Load and initiate all parameters of candidate fitting...")

    # Load the required kwargs
    time_bin_width = float(kwargs['time_bin_width']) # in ms
    # maybe add a smoothing parameter here

    # Load the optional kwargs

    # Initializations - general
    multithread = bool(settings['Multithread']['value'])
    pixel_size = float(settings['PixelSize_nm']['value']) # in nm

    if multithread == True: num_cores = multiprocessing.cpu_count()
    else: num_cores = 1
    
    # Determine number of jobs on CPU and slice data accordingly
    njobs, num_cores = utilsHelper.nb_jobs(candidate_dic, num_cores)
    data_split = utilsHelper.slice_data(candidate_dic, njobs)

    logging.info("Candidate fitting split in "+str(njobs)+" job(s) and divided on "+str(num_cores)+" core(s).")

    # Determine all localizations
    RES = Parallel(n_jobs=num_cores,backend="loky")(delayed(localize_canditates3D)(i, data_split[i], time_bin_width, pixel_size) for i in range(len(data_split)))
    
    localization_list = [res[0] for res in RES]
    localizations = pd.concat(localization_list, ignore_index=True)

    fail_list = [res[1] for res in RES]
    fails = pd.concat(fail_list, ignore_index=True)
    
    # Fit performance information
    radSym_fit_info = utilsHelper.info(localizations, fails)

    return localizations, radSym_fit_info
