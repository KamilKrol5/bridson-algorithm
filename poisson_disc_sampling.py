from itertools import product
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def is_valid(sample: np.ndarray, cell_size: float, sample_region_size: np.ndarray, radius, grid: np.ndarray,
             points: List[np.ndarray]):
    if any(sample < 0) or any(sample > sample_region_size):
        return False
    cell_index = (sample // cell_size).astype(int)  # np. [4,2,1,7]
    start = np.fromiter((max(x_i, 0) for x_i in cell_index - 2), dtype=int)
    grid_shape = grid.shape
    print(grid_shape)
    end = np.fromiter((min(x_i, length - 1) for x_i, length in zip(cell_index + 2, grid_shape)), dtype=int)

    ranges = [list(range(start_, end_ + 1)) for (start_, end_) in zip(start, end)]

    print(f'    Cell index = {cell_index}')
    for index in product(*ranges):
        point_index = grid[index] - 1
        print(f'index = {index}')
        print(f'point_index = {point_index}')
        if point_index != -1:
            distance = np.linalg.norm(sample - points[point_index])
            if distance < radius:
                return False
    return True


def poisson_disc_sampling(radius: float, sample_region_size: np.ndarray, sample_rejection_threshold: int):
    sampling_space_dimension = sample_region_size.shape[0]
    N = sampling_space_dimension
    cell_size: float = radius / np.sqrt(N)

    grid_shape = np.ceil(sample_region_size / cell_size).astype(int)
    # contains indexes of points in the 'points' list
    grid: np.ndarray = np.full(shape=grid_shape, fill_value=0)
    points: List[np.ndarray] = []
    active_points: List[np.ndarray] = []

    initial_sample = np.random.rand(N)
    active_points.append(initial_sample)

    while len(active_points) > 0:
        sample_found = False
        random_index = np.random.choice(len(active_points))
        random_sample = active_points[random_index]

        for _ in range(sample_rejection_threshold):
            sample_candidate = np.add(random_sample, __get_random_n_dim_vector(N, radius, 2 * radius))
            if is_valid(sample_candidate, cell_size, sample_region_size, radius, grid, points):
                points.append(sample_candidate)
                active_points.append(sample_candidate)
                ind = [int(x) for x in (sample_candidate // cell_size)]
                grid[ind] = len(points)
                sample_found = True
                break
        if not sample_found:
            active_points.pop(random_index)

    return points


def __get_random_n_dim_vector(dimension, min_length, max_length):
    # need n-1 angles, all in range [0; PI] except the last one which is in range [0; 2*PI]
    random_angles = [*(np.random.rand(dimension - 2) * np.pi), np.random.rand() * 2 * np.pi]
    length = np.random.uniform(min_length, max_length)
    direction = np.empty(dimension)
    for i in range(dimension):
        x_i = 1.0
        for angle in random_angles[:i]:
            x_i *= np.sin(angle)
        if i != dimension - 1:
            x_i *= np.cos(random_angles[i])
        direction[i] = x_i

    return direction * length


def draw(points):
    plt.plot(points, markersize=10, marker='*', mec='b')
    plt.xlim(-5, 8)
    plt.ylim(-5, 9)
    plt.show()


if __name__ == '__main__':
    # print(__get_random_n_dim_vector(2, 2, 4))
    # poisson_disc_sampling(1.0, np.array([4, 4]), 2)

    draw(poisson_disc_sampling(1.0, np.array([4, 4]), 30))
