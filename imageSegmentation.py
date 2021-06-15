import sys
from itertools import product
from plotclusters3D import plotclusters3D
from mean_shift import mean_shift
from mean_shift_optimized import mean_shift_opt
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2lab, lab2rgb
from skimage.filters import gaussian
import scipy.io
import time
import pandas as pd


def scale(X, x_min, x_max):
    # normalization of the array within the range [x_min, x_max]
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom


def de_normalize(img, normalize=True, x_min=-127, x_max=127):
    # (de)normalize CIELAB space, where channels have different range.
    # if normalize is set to True, it returns normalized data
    # if normalize is set to False, it returns denormalized data
    channel1 = img[:, :, 0]
    channel2 = img[:, :, 1]
    channel3 = img[:, :, 2]

    if normalize:
        norm_channel1 = channel1 / 100  # luminance channel
        norm_channel2 = (channel2 - x_min) / (x_max - x_min)
        norm_channel3 = (channel3 - x_min) / (x_max - x_min)
        return np.stack((norm_channel1, norm_channel2, norm_channel3), axis=-1)
    else:
        denorm_channel1 = channel1 * 100
        denorm_channel2 = (channel2 * (x_max - x_min)) + x_min
        denorm_channel3 = (channel3 * (x_max - x_min)) + x_min
        return np.stack((denorm_channel1, denorm_channel2, denorm_channel3), axis=-1)


def preprocessing(img, dim_5):
    cielab = de_normalize(rgb2lab(img), normalize=True)
    if dim_5:
        channel_x = np.asarray([np.arange(img.shape[1]) for _ in range(img.shape[0])])
        channel_y = np.asarray([np.repeat(i, img.shape[1]) for i in range(img.shape[0])])
        channel_xy = scale(np.stack((channel_x, channel_y), axis=-1), np.amin(cielab), np.amax(cielab))
        colors_locations = np.concatenate((cielab, channel_xy), axis=-1)
        return colors_locations.reshape(img.shape[0] * img.shape[1], 5)
    else:
        return cielab.reshape(img.shape[0] * img.shape[1], 3)


def visualize_img(img, peaks, clusters, labels, dim, radius, c, show=False):
    lab = peaks.reshape((img.shape[0], img.shape[1], peaks.shape[-1]))
    rgb = lab2rgb(de_normalize(lab[:, :, :3], normalize=False))

    colors = np.asarray([np.random.choice(range(256), size=3) for _ in range(labels.shape[0])])
    random_rgb = []
    for entry in np.squeeze(clusters):
        random_rgb.append(colors[int(entry)])

    fig, ax = plt.subplots(1, 3)
    fig.suptitle(f'image segmentation ({dim}D r={radius} c={c}) number of clusters={labels.shape[0]}')
    ax[0].set_title('original image')
    ax[0].imshow(img)
    ax[1].set_title('segmented image')
    ax[1].imshow(rgb)
    ax[2].set_title('random color clusters')
    ax[2].imshow(np.asarray(random_rgb).reshape(img.shape))
    if show:
        plt.show()
    #plt.savefig(f'images/img1test{dim}Dr{radius}c{c}.jpg')


def segment_image(img, radius, c, dim_5, verbose=True, show_fig=False, show_plt=False):
    data = preprocessing(img, dim_5)

    discarded = 0
    start_time = time.time()
    if c:
        peaks, discarded, labels = mean_shift_opt(data, radius, c, verbose=verbose)
    else:
        peaks, labels = mean_shift(data, radius)
    s = time.time() - start_time
    print(f'time:  {s}s seconds ---')

    print('number of centroids: ', labels.shape[0])
    # print(labels.shape, peaks.shape)

    clusters = np.empty((peaks.shape[0], 1))

    for i, centroid in enumerate(labels):
        clusters[np.argwhere(np.linalg.norm(peaks - centroid, axis=-1) == 0), :] = i

    visualize_img(img, peaks, clusters, labels, dim=str(peaks.shape[1]), radius=str(radius), c=str(c), show=show_fig)
    if show_plt:
        labels = de_normalize(labels[:, :3].reshape(1, labels.shape[0], 3), normalize=False)
        rgb = lab2rgb(labels)
        plotclusters3D(data, clusters, rgb[0, :, :])
    return labels.shape[0], discarded, s


def run_points():
    # test mean shift implementation on pts dataset
    pts = scipy.io.loadmat('data/pts.mat')['data'].T

    start_time = time.time()
    peaks, labels = mean_shift(pts, 2)
    s = time.time() - start_time
    print(s)

    # assign label for each point
    clusters = np.empty((peaks.shape[0], 1))
    for i, centroid in enumerate(labels):
        clusters[np.argwhere(np.linalg.norm(peaks - centroid, axis=-1) == 0), :] = i

    plotclusters3D(pts, clusters, peaks)


def run_complete_test():
    # run complete parameter test and save results
    # load data
    img = io.imread('images/img1.jpg')

    parameters = {
        'radius': (0.1, 0.2),  # (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
        'c': (2, 4, 6, 8),
        'dim_5': (False, True)
    }

    hyper_parameter_combinations = list(product(*[parameters[param] for param in parameters.keys()]))
    hyper_parameter_set_list = [dict(zip(parameters.keys(), hyper_parameter_combinations[i])) for i in
                                range(len(hyper_parameter_combinations))]
    print('number of tests: ', len(hyper_parameter_set_list))
    # sys.stdout = open('log.txt', 'w')
    evaluation_overview = pd.DataFrame(columns=list(parameters.keys())[:3] + ['clusters', 'discard ratio', 'time'])
    for i, hyper_parameter in enumerate(hyper_parameter_set_list):
        print(i, f' data {img.shape[:2], 5 if hyper_parameter["dim_5"] else 3},'
                 f' radius={hyper_parameter["radius"]}, c={hyper_parameter["c"]}')
        clusters, discarded, seconds = segment_image(img, hyper_parameter['radius'],
                                                     hyper_parameter['c'],
                                                     hyper_parameter['dim_5'],
                                                     verbose=False)
        hyper_parameter.update({'clusters': clusters, 'discard ratio': discarded, 'time': seconds})
        evaluation_overview = evaluation_overview.append(hyper_parameter, ignore_index=True)
    # sys.stdout.close()
    evaluation_overview.to_csv('mean_shift_evaluation1.csv')


def run_best_config():
    parameters = {
        'radius': 0.1,
        'c': 2,
        'dim_5': True,
        'gauss': True
    }
    print(f'parameters : r={parameters["radius"]}, c={parameters["c"]}, '
          f'5D={parameters["dim_5"]}, gauss_filter={parameters["gauss"]}')
    img = io.imread('images/img2.jpg')
    if parameters['gauss']:
        img = gaussian(img)
    segment_image(img, parameters['radius'], parameters['c'], parameters['dim_5'],
                  verbose=True, show_fig=True, show_plt=True)


if __name__ == '__main__':
    # run_points()
    # run_complete_test()
    if len(sys.argv) < 4:
        print("Not enough parameters given. Expected format: imageSegmentation.py –image -r -feature_type\n"
              "running best configuration.")
        run_best_config()
    else:
        img = io.imread(sys.argv[1])
        r = float(sys.argv[2])
        feature_type = sys.argv[3]
        dim_5 = False
        if feature_type == '3d':
            dim_5 = True
        print(f'running custom configuration. \nparameters inserted : r={r}, c=2(default), {feature_type}')
        segment_image(img, r, 2, dim_5, verbose=True, show_fig=True, show_plt=True)


