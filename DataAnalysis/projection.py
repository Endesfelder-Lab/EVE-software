
import matplotlib.pyplot as plt
import numpy as np

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': '2D Projection',
            'help_string': 'Generate a 2D projection plot of the data events.',
            'required_kwargs': [
                {'name': 'positive_events', 'type': bool, 'default': True, 'description': 'Include positive polarity events.'},
                {'name': 'negative_events', 'type': bool, 'default': True, 'description': 'Include negative polarity events.'},
                # {'name': 'plot_type', 'type': str, 'default': 'accumulation', 'description': 'Type of plot to generate (accumulation or time_surface).'},
                {'name': 'min_accumulation', 'type': int, 'default': 45, 'description': 'Minimum number of events to consider for accumulation.'},
            ],
            'optional_kwargs': [],
            # Removed empty dictionary kwargs which can cause issues with legacy utils functions
            #'dist_kwarg': {},
            #'time_kwarg': {}
        }
    }


def run_analysis(ev, positive_events=True, negative_events=True, min_accumulation=1, plot_type='accumulation'):
    if not positive_events and not negative_events:
        raise ValueError("At least one of Positive Events or Negative Events must be True.")
    min_accumulation = float(min_accumulation)
    # Filter events based on polarity
    # ev format: structured array with fields 'x', 'y', 'p', 't'
    if positive_events and not negative_events:
        ev = ev[ev['p'] > 0]
    elif negative_events and not positive_events:
        ev = ev[ev['p'] <= 0]

    if len(ev) == 0:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'No events found', ha='center', va='center')
        # We need to return the figure and a result dict
        return plt.gcf(), {'status': 'No events'}

    # Determine sensor resolution
    width = int(np.max(ev['x'])) + 1
    height = int(np.max(ev['y'])) + 1

    plt.figure(figsize=(10, 8))

    if plot_type == 'accumulation':
        # Create a 2D histogram to count events per pixel
        # bins correspond to pixel coordinates
        img, x_edges, y_edges = np.histogram2d(
            ev['x'], ev['y'],
            bins=[width, height],
            range=[[0, width], [0, height]]
        )

        # Transpose image
        img_T = img.T

        # Calculate robust limits for visualization (1st and 99th percentile of non-zero data)
        if np.any(img_T > 0):
            vmin = np.percentile(img_T[img_T > 0], 1)
            vmax = np.percentile(img_T[img_T > 0], 99)
        else:
            vmin, vmax = 0, 1
        img[img < min_accumulation] = 0
        plt.imshow(img, origin='lower', cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Event Count (Density)')
        plt.title('2D Accumulation (Density Map)')

    elif plot_type == 'time_surface':
        # Time surface usually shows the LATEST timestamp at each pixel
        img = np.zeros((height, width))
        # Fill the array with timestamps; later events overwrite earlier ones
        for e in ev:
            img[int(e['y']), int(e['x'])] = e['t']

        plt.imshow(img, origin='lower', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Timestamp (Recency)')
        plt.title('Time Surface (Last Event)')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # plt.show()


    return plt.gcf(), {'Status': 'Success', 'min_accumulation': min_accumulation, 'plot_type': plot_type, 'positive_events': positive_events, 'negative_events': negative_events}


if __name__ == '__main__':
    # Simulated data: [timestamp, x, y, polarity]
    num_events = 50000
    events = np.random.rand(num_events, 4)
    events[:, 0] *= 10.0  # 10 seconds
    events[:, 1] *= 640  # Width
    events[:, 2] *= 480  # Height
    events[:, 3] = np.where(events[:, 3] > 0.5, 1, -1)

    # Run fixed analysis
    run_analysis(events, plot_type='time_surface')