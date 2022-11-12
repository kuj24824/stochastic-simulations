"""Sampler class that generates specific amount of sampling points in an area using various sampling methods."""

# Imports
import math
import numpy as np


class sampler:
    """
    A class that generates sampling points in an area.

    Attributes
    ----------
    x_min : float
        minimal x-value of the sampling area
    x_max : float
        maximal x-value of the sampling area
    y_min : float
        minimal y-value of the sampling area
    y_max : float
        maximal y-value of the sampling area
    method : str
        sampling method

    Methods
    -------
    generate_samples(par_s)
        Generates par_s sampling points using the self.method sampling technique.

    >>> test = sampler([-2, 2, -1, 2])
    >>> x, y = test.scale_coordinate(0.4, 0.6)
    >>> np.allclose((x, y), (-0.4, 0.8))
    True
    """

    def __init__(self, par_a, method = None):
        """
        Initializes an object of the class sampler.

        Parameters
        ----------
        par_a : list
            contains the edge values of the sampling area
        method : str, optional
            Sampling method (default is uniform)
        """
        # Total area defined by the edges
        self.x_min = par_a[0]
        self.x_max = par_a[1]
        self.y_min = par_a[2]
        self.y_max = par_a[3]

        if method is None:
            # Default sampling method is uniform random
            self.method = 'uniform'
        else:
            self.method = method

    def scale_coordinate(self, par_x, par_y):
        """
        Scales the coordinates to the sampling area.

        Parameters
        ----------
        par_x : float
            x-coordinate between 0 and 1
        par_x : float
            y-coordinate between 0 and 1

        Returns
        -------
        x, y : tuple
            scaled x- and y-coordinate
        """
        dist_x = self.x_max - self.x_min
        dist_y = self.y_max - self.y_min

        # Scale the variables to the sampling area
        x = (dist_x * par_x) + self.x_min
        y = (dist_y * par_y) + self.y_min

        return x, y

    def random_permutation(self, array):
        """
        Creates a random permutation of a given array.

        Parameters
        ----------
        array : numpy.ndarray
            array containing elements that needs to be permutated

        Returns
        -------
        array : numpy.ndarray
            random permutation of the original array
        """
        # Set index k equal to the final element in the array
        k = len(array) - 1

        while k > 0:
            # Generate random index to swap with index k
            index = int(k * np.random.uniform()) + 1
            element = array[k]
            array[k] = array[index]
            array[index] = element

            k -= 1

        return array

    def uniform(self, par_s):
        """
        Generates par_s sampling points uniformly distributed over the sampling area.

        Parameters
        ----------
        par_s : int
            number of sampling points

        Returns
        -------
        samples : numpy.ndarray
            array with x- and y-coordinate of the par_s sampling points
        """
        # Array to store the sample points
        samples = np.empty((par_s, 2))

        for s in range(par_s):
            # Generate 2 uniform random variables
            x = np.random.uniform()
            y = np.random.uniform()

            # Scale the coordinate to the sampling area
            x_coord, y_coord = self.scale_coordinate(x, y)
            samples[s, 0] = x_coord
            samples[s, 1] = y_coord

        return samples

    def latin_hypercube_1d(self, par_s):
        """
        Generates par_s points in [0, 1 each in a separate interval [j/par_s, (j + 1)/par_s] for j = 0, 1, ..., par_s - 1.

        Parameters
        ----------
        par_s : int
            number of sampling points

        Returns
        -------
        samples : numpy.ndarray
            array with par_s sampling points in 1D
        """
        # Array to store the sample coordinates
        samples = np.empty(par_s)

        # Generate one sample point per interval 1/s
        for s in range(par_s):
            samples[s] = (s + np.random.uniform()) / par_s

        return samples

    def latin_hypercube_2d(self, par_s):
        """
        Generates par_s sampling points using the latin hypercube sampling method.

        Parameters
        ----------
        par_s : int
            number of sampling points

        Returns
        -------
        samples : numpy.ndarray
            array with x- and y-coordinate of the par_s sampling points
        """
        # Array to store the sample points
        samples = np.empty((par_s, 2))

        x = self.random_permutation(self.latin_hypercube_1d(par_s))
        y = self.random_permutation(self.latin_hypercube_1d(par_s))

        # Combine the x and y coordinate to form a sampling point
        for s in range(par_s):
            index_x = 0
            index_y = 0

            # Sampling points are not allowed to be on the same row or column
            if math.trunc(y[index_y] * par_s) == math.trunc(x[index_x] * par_s):
                # Generate a random index to swap the y-coordinate
                index_y = int((len(y) - 1) * np.random.uniform())
                if index_y == 0:
                    index_y += 1

            # Scale the coordinate to the sampling area
            x_coord, y_coord = self.scale_coordinate(x[index_x], y[index_y])
            samples[s, 0] = x_coord
            samples[s, 1] = y_coord

            # Delete the used coordinates, so they can't be used in the swap
            x = np.delete(x, index_x)
            y = np.delete(y, index_y)

        return samples

    def orthogonal(self, par_s):
        """
        Generates par_s sampling points using the orthogonal sampling method.

        Parameters
        ----------
        par_s : int
            number of sampling points

        Returns
        -------
        samples : numpy.ndarray
            array with x- and y-coordinate of the par_s sampling points
        """
        # Array to store the sample points
        samples = np.empty((par_s, 2))

        """
        Orthogonal sampling code
        """

        return samples

    def generate_samples(self, par_s):
        """
        Generates par_s sampling points using the self.method sampling technique.

        Parameters
        ----------
        par_s : int
            number of sampling points

        Returns
        -------
        samples : numpy.ndarray
            array with x- and y-coordinate of the par_s sampling points
        """
        if self.method == "uniform":
            samples = self.uniform(par_s)
        elif self.method == "latin_hypercube":
            samples = self.latin_hypercube_2d(par_s)
        else:
            samples = self.orthogonal(par_s)

        return samples


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
