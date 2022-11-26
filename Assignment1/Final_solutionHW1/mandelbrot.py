"""Mandelbrot class that estimates the area using a Monte Carlo approach."""

# Imports
import numpy as np
import sampling


class mandel:
    """
    A class that estimates the Mandelbrot area.

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
    area : float
        total sampling area
    sampler : .sampler
        object of the sampler class

    Methods
    -------
    est_area(par_s, par_i, method=None)
        Estimates the Mandelbrot area given par_s sampling points and par_i iterations.

    >>> test = mandel([-2, 2, -1, 1])
    >>> test.area == 8
    True

    >>> np.allclose(test.step([0,0.1], [0.0, 0.1]), [-0.01, 0.1])
    True

    >>> test.condition([0, 1])
    0
    >>> test.condition([1, 1])
    0
    >>> test.condition([4, 3])
    1
    """

    def __init__(self, par_a):
        """
        Initializes an object of the class mandel.

        Parameters
        ----------
        par_a : list
            contains the edge values of the sampling area
        """
        # Total area defined by the edges
        self.x_min = par_a[0]
        self.x_max = par_a[1]
        self.y_min = par_a[2]
        self.y_max = par_a[3]

        # Calculate the total sampling area
        self.area = self.calc_area(par_a)
        # Create a sample generator for this area
        self.sampler = sampling.sampler(par_a)
        # Array to store the used sample points
        self.samples = None

    def calc_area(self, par_a):
        """
        Calculates the area of a rectangle.

        Parameters
        ----------
        par_a : list
            contains the edge values of the sampling area

        Returns
        -------
        area : float
            total sampling area
        """
        [x_min, x_max, y_min, y_max] = par_a
        return (x_max - x_min) * (y_max - y_min)

    def step(self, par_z, par_c):
        """
        Calculates the next value in the sequence [z(n+1) = z(n)^2 + c].

        Parameters
        ----------
        par_z : [float, float]
            list with the real and imaginary part of z
        par_c : [float, float]
            list with the real and imaginary part of c

        Returns
        -------
        z(n+1) : [float, float]
            list with the real and imaginary part of the next z value
        """
        [z_re, z_im] = par_z
        [c_re, c_im] = par_c

        return [z_re**2 - z_im**2 + c_re, 2 * z_re * z_im + c_im]

    def condition(self, par_z):
        """
        Checks if the sequence will diverge.

        Parameters
        ----------
        par_z : [float, float]
            list with the real and imaginary part of z

        Returns
        -------
        1 : int
            if the magnitude of z is bigger than 2
        0 : int
            otherwise
        """
        [z_re, z_im] = par_z

        magn = np.sqrt((z_re**2 + z_im**2))
        if magn > 2:
            return 1

        return 0

    def check_point(self, par_c, par_i):
        """
        Checks if a single point (par_c) is in the Mandelbrot set given par_i iterations.

        Parameters
        ----------
        par_c : [float, float]
            list with the real and imaginary part of the sampling point
        par_i : int
            maximum number of iterations of the sequence

        Returns
        -------
        par_i - i : int
            difference between the maximum number of iterations and the number of iterations
            after which the the magnitude of z is above 2
        """
        z = [0, 0]
        for i in range(par_i):
            z = self.step(z, par_c)
            if self.condition(z):
                break

        return par_i - i - 1

    def est_area(self, par_s, par_i, method=None, store_samples=False):
        """
        Estimates the Mandelbrot area given par_s sampling points and par_i iterations.

        Parameters
        ----------
        par_s : int
            number of sampling points
        par_i : int
            maximum number of iterations of the sequence
        method : str, optional
            sampling method (default in uniform)
        store_samples : boolean
            Indicator if the used samples should be stored (default is false)
        Returns
        -------
        area : float
            estimated area of the Mandelbrot set
        """
        # Number of samples in the mandelbrot set
        inside = 0

        # Generate samples
        if method is None:
            method = 'uniform'
        self.sampler.method = method
        samples = self.sampler.generate_samples(par_s)

        # Check for every sampling point if it is in the Mandelbrot set
        for s in samples:
            if self.check_point(s, par_i) == 0:
                inside += 1

        if store_samples:
            self.samples = samples

        # Estimated area is the fraction of sampling points in the set multiplied with the total area
        return (inside/par_s) * self.area

    def avg_est_area(self, runs, par_s, par_i, method=None):
        """
        Estimates the Mandelbrot area given par_s sampling points and par_i iterations for a number of runs.

        Parameters
        ----------
        runs : int
            number of simulations to run
        par_s : int
            number of sampling points
        par_i : int
            maximum number of iterations of the sequence
        method : str, optional
            sampling method (default in uniform)

        Returns
        -------
        average_area : float
            average estimated area of the Mandelbrot set
        sample_var : float
            the sample variance over the runs
        """
        # Calculate the average area over all the different runs
        area = []
        for _ in range(runs):
            area.append(self.est_area(par_s, par_i, method))
        average_area = np.sum(area) / runs

        # Calculate the sample variance
        sample_var = 0
        for i in range(runs):
            sample_var += (area[i] - average_area)**2
        sample_var /= (runs - 1)

        return average_area, sample_var

    def grid(self, spacing):
        """
        Creates a grid for the sampling area with a certain spacing.

        Parameters
        ----------
        spacing : float
            distance between 2 points on the grid

        Returns
        -------
        xx, yy : numpy.ndarray, numpy.ndarray
            2d arrays with the x- and y-coordinates of the gridpoints
        """
        # Create the axis of the grid
        x_axis = np.arange(self.x_min, self.x_max + spacing, spacing)
        y_axis = np.arange(self.y_min, self.y_max + spacing, spacing)

        yy, xx = np.meshgrid(y_axis, x_axis)

        return xx, yy

    def generate_fractal(self, par_i, spacing):
        """
        Creates a fractal plane of the Mandelbrot set.

        Parameters
        ----------
        par_i : int
            maximum number of iterations of the sequence
        spacing : float
            distance between 2 points on the grid

        Returns
        -------
        plane : numpy.ndarray
            2d array with each value related to the speed with which that gridpoint diverges
        """
        # Create the axis of the grid
        x_axis = np.arange(self.x_min, self.x_max + spacing, spacing)
        y_axis = np.arange(self.y_min, self.y_max + spacing, spacing)

        # Array to store the values of the plane
        plane = np.empty((len(x_axis), len(y_axis)))

        for x in range(len(x_axis)):
            for y in range(len(y_axis)):
                # Starting point of the sequence
                c = [x_axis[x], y_axis[y]]
                plane[x, y] = self.check_point(c, par_i)

        return plane


class subarea:
    """
    A class that subdivides an area to estimate the Mandelbrot area in every subarea separately.

    Attributes
    ----------
    mandel : .mandel
        object of the class mandel
    probability : float
        sampling probability in the area
    iterations : int
        number of iterations used to check a sample point
    prob_incr : boolean
        Indicator for the probability increase if it is an important subarea

    Methods
    -------
    subdivide(n_subareas)
        Subdivides the area in n subareas.

    >>> test = subarea([-8, 8, -8, 8], 200, 0.5)
    >>> subareas = test.subdivide(16)
    >>> len(subareas) == 16
    True
    >>> [subareas[0].mandel.x_min, subareas[0].mandel.x_max, subareas[0].mandel.y_min, subareas[0].mandel.y_max] == [-8, -4.0, -8, -4.0]
    True
    >>> [subareas[1].mandel.x_min, subareas[1].mandel.x_max, subareas[1].mandel.y_min, subareas[1].mandel.y_max] == [-8, -4.0, -4.0, 0.0]
    True
    >>> [subareas[2].mandel.x_min, subareas[2].mandel.x_max, subareas[2].mandel.y_min, subareas[2].mandel.y_max] == [-8, -4.0, 0.0, 4.0]
    True
    >>> [subareas[3].mandel.x_min, subareas[3].mandel.x_max, subareas[3].mandel.y_min, subareas[3].mandel.y_max] == [-8, -4.0, 4.0, 8]
    True
    >>> [subareas[4].mandel.x_min, subareas[4].mandel.x_max, subareas[4].mandel.y_min, subareas[4].mandel.y_max] == [-4.0, 0.0, -8, -4.0]
    True
    >>> subareas[0].probability == 0.5 / 16
    True
    """

    def __init__(self, par_a, par_i, prob):
        """
        Initializes an object of the class subarea.

        Parameters
        ----------
        par_a : list
            contains the edge values of the sampling area
        par_i : int
            number of iterations used to check a sample point
        prob : float
            sampling probability in the area
        """
        # Create a mandel object for this area
        self.mandel = mandel(par_a)

        # Sampling probability
        self.probability = prob
        self.iterations = par_i

        # Variable that indicates if the probability should be increased
        self.prob_incr = False

    def subdivide(self, n_subareas):
        """
        Subdivides the area in n subareas.

        Parameters
        ----------
        n_subareas : int
            number of subareas that need to be created, should have a integer square root

        Returns
        -------
        new_subareas : list
            contains n .subarea objects
        """
        # Set the initial coordinates at the minimal value
        x_min = self.mandel.x_min
        x_max = self.mandel.x_min
        y_min = self.mandel.y_min
        y_max = self.mandel.y_min

        # List to store the newly created subareas
        new_subareas = []
        # Equally distribute the sampling probability over the subareas
        new_prob = self.probability / n_subareas

        # Calculate the length of the subareas
        distance_x = (self.mandel.x_max - self.mandel.x_min) / np.sqrt(n_subareas)
        distance_y = (self.mandel.y_max - self.mandel.y_min) / np.sqrt(n_subareas)

        # Determine the edges of the subareas
        for _ in range(int(np.sqrt(n_subareas))):
            # The minimal value is the maximal value of the previous subarea
            x_min = x_max
            x_max += distance_x

            for _ in range(int(np.sqrt(n_subareas))):
                y_min = y_max
                y_max += distance_y

                # Create subarea
                new_subareas.append(subarea([x_min, x_max, y_min, y_max], self.iterations, new_prob))

            y_min = self.mandel.y_min
            y_max = self.mandel.y_min

        return new_subareas


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
