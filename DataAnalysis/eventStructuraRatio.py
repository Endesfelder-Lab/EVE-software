import numpy as np
import matplotlib.pyplot as plt


def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Event Structural Ratio',
            'help_string': 'Calculate the event structural ratio for a given event dataset',
            'required_kwargs': [
                {'name': 'x_res', 'type': int, 'default': 941, 'description': 'Sensor resolution (X-Axis) in microns.'},
                {'name': 'y_res', 'type': int, 'default': 483, 'description': 'Sensor resolution (Y-Axis) in microns.'},
                {'name': 'refN', 'type': int, 'default': 800000, 'description': 'Reference number of events for interpolation'},
                {'name': 'count', 'type': int, 'default': 1000000, 'description': 'Number of events in the dataset'}
            ],
            'optional_kwargs': [],
        }
    }

def run_analysis(ev, x_res, y_res, count=30000, refN=20000):
    count = int(count)
    refN = int(refN)
    x_res = int(x_res)
    y_res = int(y_res)

    # Auto-adjust resolution if events are out of bounds
    if len(ev) > 0:
        x_res = max(x_res, int(ev['x'].max()) + 1)
        y_res = max(y_res, int(ev['y'].max()) + 1)
    print(ev.shape)
    if len(ev) < 2 * count: 
        fig = plt.figure()
        plt.text(0.5, 0.5, f"Not enough events: {len(ev)} < {2*count}", ha='center', va='center')
        return fig, {'score': 0.5, 'warning': f'Not enough events: {len(ev)} < {2*count}'}
    score = np.zeros(int(len(ev)/count) - 1)

    for i in range(0, len(score)):
        st_idx = i * count
        ed_idx = st_idx + count
        packet = ev[st_idx:ed_idx]

        cnt = count_distribution(packet, [x_res, y_res], use_polarity=False)

        N = cnt.sum()
        L = cnt.size - ((1 - refN/N) ** cnt).sum()

        score[i] = (cnt * (cnt-1) / (N + np.spacing(1)) / (N - 1 + np.spacing(1))).sum() * L

    img = projection_image(ev, [x_res, y_res])
    plt.imshow(img)

    return plt.gcf(), {'score': float(score.mean())}

def count_distribution(ev, size, use_polarity=True):
    bins_, range_  = [size[0], size[1]], [[0, size[0]], [0, size[1]]]

    if use_polarity:
        weights = (-1) ** (1 + ev['p'].astype(np.float64))
    else:
        weights = (+1) ** (0 + ev['p'].astype(np.float64))

    counts, *_ = np.histogram2d(ev['x'], ev['y'], weights=weights, bins=bins_, range=range_)

    return counts

def projection_image(ev, size, max_count=1, flip=True):
    # Filter out events that are outside the specified resolution
    mask = (ev['x'] >= 0) & (ev['x'] < size[0]) & (ev['y'] >= 0) & (ev['y'] < size[1])
    ev = ev[mask]

    cnt = count_distribution(ev, size)
    cnt = np.clip(np.abs(cnt), 0, max_count).astype(np.int64)

    color = np.linspace(0, 255, max_count + 1)

    img = np.ones((*size, 3)) * 255
    img[ev['x'], ev['y'], 1 - ev['p']] = color[-1 - cnt[ev['x'], ev['y']]]
    img[ev['x'], ev['y'], 2 - ev['p']] = color[-1 - cnt[ev['x'], ev['y']]]

    if flip:
        img = np.flip(np.rot90(img, 1), axis=0).astype(np.uint8)

    return img

if __name__ == '__main__':
    
    events = np.random.rand(100000, 4)
    x_res = 256
    y_res = 256
    count = 20000
    refN = 20000

    graph, results = run_analysis(events, x_res, y_res, count, refN)

