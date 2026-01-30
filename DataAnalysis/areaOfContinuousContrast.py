import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter, sobel
import os
import gc
from concurrent.futures import ProcessPoolExecutor

# --- Global container for worker processes ---
# This ensures data is not pickled/copied for every single task
worker_data = {}

def init_worker(t_shared, x_shared, y_shared):
    """
    Initialize the worker process by storing the large arrays in global memory.
    This runs once per process, not once per task.
    """
    worker_data['t'] = t_shared
    worker_data['x'] = x_shared
    worker_data['y'] = y_shared

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Area of Continuous Contrast',
            'help_string': 'Calculate the area of continuous contrast for a given event dataset',
            'required_kwargs': [
                {'name': 'x_res', 'type': int, 'default': 941, 'description': 'Sensor resolution (X-Axis) in microns.'},
                {'name': 'y_res', 'type': int, 'default': 483, 'description': 'Sensor resolution (Y-Axis) in microns.'}
            ],
            'optional_kwargs': [
                {'name': 'min_interval', 'type': int, 'default': 60000, 'description': 'Min interval (default: auto)'},
                {'name': 'max_interval', 'type': int, 'default': 10000000, 'description': 'Max interval (default: auto)'},
                {'name': 'step_interval', 'type': int, 'default': None, 'description': 'Step size (default: auto)'},
            ],
        }
    }

def compute_single_interval(args):
    """
    Worker function to process a single interval.
    Args now only contain metadata. Data is accessed via worker_data.
    """
    interval, width, height = args
    
    # Retrieve data from global storage (zero copy)
    t = worker_data['t']
    x = worker_data['x']
    y = worker_data['y']
    
    # 1. Define Time Bins
    t_start, t_end = t[0], t[-1]
    
    # RAM OPTIMIZATION: Instead of np.digitize (which allocates len(t) ints),
    # use searchsorted because 't' is already sorted.
    # This keeps memory usage near zero for the slicing logic.
    bins = np.arange(t_start, t_end + interval, interval)
    
    # Find indices where time bins start/end
    # This is much faster and lighter than digitize for sorted data
    idx_bounds = np.searchsorted(t, bins)
    
    contrasts = []
    
    # 2. Process each frame using slices
    # Iterate through the bins defined by indices
    for i in range(len(idx_bounds) - 1):
        start_idx = idx_bounds[i]
        end_idx = idx_bounds[i+1]
        
        # Skip empty frames
        if start_idx == end_idx:
            continue
            
        # Create views (zero copy) of the current frame's events
        x_slice = x[start_idx:end_idx]
        y_slice = y[start_idx:end_idx]
        
        # Fast histogram
        img, _, _ = np.histogram2d(y_slice, x_slice, 
                                   bins=[height, width], 
                                   range=[[0, height], [0, width]])
        
        # Convert to boolean contrast map
        img = (img > 0).astype(np.float32) * 255

        # 3. Contrast Calculation
        blurred = gaussian_filter(img, sigma=2)
        grad_x = sobel(blurred, axis=1)
        grad_y = sobel(blurred, axis=0)
        magnitude = np.hypot(grad_x, grad_y)
        contrasts.append(np.std(magnitude))
        del img, blurred, grad_x, grad_y, magnitude  # attempt to free memory

    mean_val = np.mean(contrasts) if contrasts else 0.0
    contrasts.clear()  # free memory
    gc.collect() # force garbage collection to free memory immediately

    return {'interval': interval, 'mean_contrast': mean_val}


def run_analysis(ev, x_res=256, y_res=256, min_interval=None, max_interval=None, step_interval=None):
    """
    Run the area of continuous contrast analysis optimized for memory usage.
    """
    x_res = int(x_res)
    y_res = int(y_res)
    
    # Ensure contiguous arrays for better performance
    t = np.ascontiguousarray(ev['t'].astype(np.float64))
    x = np.ascontiguousarray(ev['x'].astype(np.int32))
    y = np.ascontiguousarray(ev['y'].astype(np.int32))
    
    max_x_event = x.max()
    max_y_event = y.max()
    
    if max_x_event >= x_res:
        print(f"DEBUG: Auto-adjusting X resolution from {x_res} to {max_x_event + 1}")
        x_res = int(max_x_event + 1)
        
    if max_y_event >= y_res:
        print(f"DEBUG: Auto-adjusting Y resolution from {y_res} to {max_y_event + 1}")
        y_res = int(max_y_event + 1)

    duration = t[-1] - t[0]
    print(f"Recording Duration: {duration} | Resolution: {x_res}x{y_res}")

    if min_interval is None:
        min_interval = max(1000, int(duration * 0.001))
    else:
        min_interval = int(min_interval)
        
    if max_interval is None:
        max_interval = int(duration * 0.2)
    else:
        max_interval = int(max_interval)
        
    if step_interval is None:
        step_interval = int((max_interval - min_interval) / 5)
        step_interval = max(100, step_interval)
    else:
        step_interval = int(step_interval)
    
    if duration < min_interval:
        print("WARNING: Interval is larger than total recording duration!")
        return None, {'area': 0}

    print(f"Auto-Configured Analysis: Min={min_interval}, Max={max_interval}, Step={step_interval}")

    intervals = np.arange(min_interval, max_interval, step_interval)
    
    # MEMORY FIX: Only pass metadata in the tasks list
    # t, x, y are NOT passed here
    tasks = [(iv, x_res, y_res) for iv in intervals]
    
    results_list = []
    
    # Use max_workers to limit concurrent processes (optional, defaults to CPU count)
    # Using slightly fewer workers than CPU count can sometimes help with RAM if dataset is massive
    num_workers = os.cpu_count()
    print(f"Starting parallel processing with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=init_worker, 
                             initargs=(t, x, y)) as executor:
        results_gen = executor.map(compute_single_interval, tasks)
        results_list = list(results_gen)

    results = pd.DataFrame(results_list)
    results = results.sort_values('interval')
    
    if results.empty or results['mean_contrast'].sum() == 0:
        print("WARNING: All contrast values are 0.0. Check your x/y coordinates or kernel size.")

    plt.figure(figsize=(10, 6))
    plt.plot(results['interval'], results['mean_contrast'], marker='o', markersize=2)
    plt.title(f'Contrast Analysis ({x_res}x{y_res})')
    plt.xlabel('Interval (us)')
    plt.ylabel('Contrast')
    plt.grid(True)
    
    area = trapezoid(results['mean_contrast'], results['interval'])
    return plt.gcf(), {
        'area': area,
        'curve_x': results['interval'].tolist(),
        'curve_y': results['mean_contrast'].tolist()
    }