import numpy as np


def find_peak(data, idx, radius, threshold=0.01):
    """

    :param data: data vector of dimension (number of points/pixels x number of features)
    :param idx: index of current point/pixel
    :param radius: window radius used in search path
    :param threshold: amount of minimum movement to continuing searching peak
    :return: peak
    """

    centroid = data[idx, :]
    while True:
        dist = np.linalg.norm(data - centroid, axis=-1)
        points = np.argwhere(dist <= radius)
        peak = np.mean(np.squeeze(data[points, :]), axis=0)
        if np.linalg.norm(centroid - peak) < threshold:
            break
        else:
            centroid = np.ravel(peak)
    return peak


def mean_shift(data, radius):
    """

    :param data: data vector of dimension (number of points/pixels x number of features)
    :param radius: window radius used in search path
    :param threshold: threshold used to stop iteration in find_peak()
    :return: centroids: list of peak for each point with dimension equal to data,
            peak_list: list of unique values of peaks
    """

    # initialization of centroids and peaks list
    centroids = np.zeros(data.shape)  # ones(data.shape) * 999 TODO check if correct, 999 should not be necessary
    # initialization of first centroid with high values, distant to each possible peak
    peaks_list = np.ones((1, data.shape[-1])) * 999

    # print('data', data.shape)
    # for each point in data vector calculate new peak
    for i in range(centroids.shape[0]):
        if i % 500 == 0:  # print progress
            print(i)
        peak = find_peak(data, i, radius)
        # calculate if a peak in peak_list radius/2 distant from new peak
        same_peak = np.argwhere(np.linalg.norm(peaks_list - peak, axis=-1) < radius / 2)
        if same_peak.shape[0] != 0:  # if similar point exists, use it
            peak = peaks_list[same_peak[0][0]]
        else:
            # append new peak in peak list
            peaks_list = np.concatenate((peaks_list, peak.reshape((1, data.shape[-1]))), axis=0)
        centroids[i, :] = peak
    return centroids, peaks_list[1:]
