from typing import Tuple, List
from collections import deque

import numpy as np


def poisson_disc_sampling(radius: float, sample_region_size: np.ndarray, sample_rejection_threshold: int):
    sampling_space_dimension = sample_region_size.shape[0]
    N = sampling_space_dimension
    cell_size: float = radius/np.sqrt(N)

    grid_shape = sample_region_size // cell_size
    # contains indexes of points in the 'points' list
    grid: np.ndarray = np.full(shape=grid_shape, fill_value=-1)
    points: List[np.ndarray] = []
    active_points: List[np.ndarray] = []

    initial_sample = np.random.rand(*sample_region_size, 1) * sample_region_size
    active_points.append(initial_sample)

    while len(active_points) > 0:
        random_sample = np.random.choice(active_points)
        for _ in range(sample_rejection_threshold):

            sample_candidate = random_sample


def __get_random_n_dim_vector(dimension ,min_length, max_length):
    # need n-1 angles, all in range [0; PI] except the last one which is in range [0; 2*PI]
    random_angles = [*(np.random.rand(dimension - 2) * np.pi), np.random.rand() * 2 * np.pi]
    length = np.random.uniform(min_length, max_length)
    direction = np.empty(dimension)
    for i in range(dimension):
        x_i = length
        for angle in random_angles[:i]:
            x_i *= np.sin(angle)
        if i != dimension - 1:
            x_i *= np.cos(random_angles[i])
        direction[i] = x_i

    return direction * length


if __name__ == '__main__':
    print(__get_random_n_dim_vector(2, 2, 4))