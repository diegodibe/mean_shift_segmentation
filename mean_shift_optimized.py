import numpy as np


def find_peak_opt(data, idx, radius, c, threshold=0.01):
    """
        find peak optimized
        :param data: data vector of dimension (number of points/pixels x number of features)
        :param idx: index of current point/pixel
        :param radius: window radius used in search path
        :param threshold: amount of minimum movement to continuing searching peak
        :param c: search path threshold
        :return: peak
    """
    centroid = data[idx, :]
    window = np.zeros((data.shape[0], 1))
    while True:
        dist = np.linalg.norm(data - centroid, axis=-1)
        points = np.argwhere(dist <= radius)  # always at leas one
        # optimization number 2
        idx = np.argwhere(np.linalg.norm(data - centroid, axis=-1) <= (radius / c))
        window[idx, :] = 1
        if points.size == 1:  # single point with dim(1,3), mean would return float
            peak = data[points, :]
        else:
            peak = np.mean(np.squeeze(data[points, :]), axis=0)
        if np.linalg.norm(centroid - peak) < threshold:
            break
        else:
            centroid = np.ravel(peak)
    return peak, window


def mean_shift_opt(data, radius, c, verbose=False):
    """

        :param data: data vector of dimension (number of points/pixels x number of features)
        :param radius: window radius used in search path
        :param threshold: threshold used to stop iteration in find_peak()
        :param c: search path threshold
        :param verbose: boolean to print the progress
        :return: centroids: list of peak for each point with dimension equal to data,
                peak_list: list of unique values of peaks
        """

    max = 10  # value that is far enough to not be taken in account
    # initialization of centroids and peaks list
    centroids = np.ones(data.shape) * max
    # initialization of first centroid with high values, distant to each possible peak
    peaks_list = np.ones((1, data.shape[-1])) * max
    discarded_points = 0

    # print('data', data.shape)
    # for each point in data vector calculate new peak
    for i in range(centroids.shape[0]):
        if i % 20000 == 0 and verbose:
            print('pixel processed: ', i)  # print progress
        if np.all(centroids[i] == max):
            peak, points = find_peak_opt(data, i, radius, c)
            # calculate if a peak in peak_list radius/2 distant from new peak
            same_peak = np.argwhere(np.linalg.norm(peaks_list - peak, axis=-1) < radius / 2)
            if same_peak.shape[0] != 0:  # if similar point exists, use it
                peak = peaks_list[same_peak[0][0]]
            else:
                # append new peak in peak list
                peaks_list = np.concatenate((peaks_list, peak.reshape((1, data.shape[-1]))), axis=0)
            centroids[i, :] = peak

            # optimization 1/2, saving peak for neighbours
            indexes = np.argwhere(np.linalg.norm(data - peak, axis=-1) <= radius)
            window = np.argwhere(points == 1)
            centroids[window, :] = peak
            centroids[indexes, :] = peak
        else:
            discarded_points += 1
    discard_ratio = discarded_points/data.shape[0]
    print('discarded_points: ', discarded_points, f'=> {discarded_points}/{data.shape[0]} = ', discard_ratio)
    return centroids, discard_ratio, peaks_list[1:]
