import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

def get_pattern(depth_map, f = 5, n = 3):
    '''
    Returns a matrix of randomly generated values to be tiled over
    the depth map.
    :param depth_map: depth map dimensions are needed to determine
        the optimal dimensions of the pattern tile
    :param f: fraction of the depth map height and width that a single
        pattern tile will cover
    :param n: upper bound on the set from which random numbers for the
        pattern tile are generated
    :return: resulting pattern tile matrix
    '''
    shape = (depth_map.shape[0]//f, depth_map.shape[1]//f)
    pattern = np.random.randint(0, n, shape) / n
    return pattern


def magic(depth_map, pattern, focal_length = 0.2):
    '''
    Creates an autostereogram given a depth map of the image and
    a tiling pattern.
    :param depth_map: depth map from which to generate autostereogram
    :param focal_length: depth of perspective for resulting image
    :return: autostereogram matrix of values
    '''

    # height and width of the pattern tile
    h, w = pattern.shape

    # this step is not neccessary but can be used to center the
    # object in the frame
    depth_map = np.concatenate((depth_map[:, :w//2], depth_map), axis=1)

    # Depth map values are normalized to between 0 and 1
    dMin = depth_map.min()
    dMax = depth_map.max()
    depth_map = (depth_map - dMin) / (dMax - dMin)

    # Autostereogram matrix is initialized with same proportions
    # as the depth map image
    autostereogram = np.zeros_like(depth_map)

    for r in range(autostereogram.shape[0]):
        for c in range(autostereogram.shape[1]):
            # no values are shifted in the first row of tiles since
            # it is a reference point
            if c < w:
                autostereogram[r, c] = pattern[r % h, c]
            # values are shifted proportionally to the corresponding
            # depth map values to create the illusion of volume
            else:
                shift = int(depth_map[r, c] * focal_length * w)
                autostereogram[r, c] = autostereogram[r, c - w + shift]
    return autostereogram


depth_map = imread("lambda.jpg", mode = 'L')
pattern = get_pattern(depth_map)

autostereogram = magic(depth_map, pattern)

plt.figure(figsize=(26,20))
plt.imshow(autostereogram, cmap = 'Reds')
plt.show()
