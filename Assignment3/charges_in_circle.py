"""Class to find the minimal energy configuration of n particles in a unit circle."""

# Imports
import numpy as np


class configuration():
    """
    A class that optimizes the configuration of n particles in a unit circle.

    Attributes
    ----------
    n_samples : int
        number of particles
    coordinates : numpy.ndarry
        array with the coordinates of the particles (n x 2)
    energy : float
        total energy of the current configuration
    cooling_scheme : str
        scheme used to calculate the temperature
    method : str
        method used to update the coordinates
    path : list
        list with all the visited points during the optimization
    energy_list : list
        list with the energy of every configuration during the optimization

    Methods
    -------
    generate_samples(n)
        Generate n random points in the unit circle.
    calc_energy()
        Calculate the energy of the current configuration.
    calc_force(index)
        Calculate the force on 1 particle in the configuration.
    optimize(iterations)
        Optimize the configuration.
    """

    def __init__(self, n=None, scheme='logarithmic', method='brownian'):
        """
        Initialize a configuration with n particles in a unit circle.

        Parameters
        ----------
        n : int, optional
            number of particles (default is None)
        scheme : str, optional
            cooling scheme (default is logarithmic)
        method : str, optional
            method to update the coordinates (default is brownian)
        """
        if n:
            # Generate n samples and calculate the initial energy
            self.n_samples = n
            self.coordinates = self.generate_samples()
            self.energy = self.calc_energy()
        else:
            # Initialize without charges
            self.n_samples = None
            self.coordinates = None
            self.energy = None

        # Set the cooling scheme
        if scheme != 'logarithmic' and scheme != 'hyperbolic_fast' and scheme != 'hyperbolic_slow':
            raise NameError("Unkown cooling scheme (%s) given, expected 'logarithmic', 'hyperbolic_fast' or 'hyperbolic_slow'." % scheme)
        self.cooling_scheme = scheme

        # Set the method to use for updating the coordinates of the particles
        if method != 'brownian' and method != 'force' and method != 'force_brownian':
            raise NameError("Unkown method (%s) given, expected 'brownian', 'force' or 'force_brownian'." % method)
        self.updating_method = method

        # List to keep track of all the visited points and the energy in every iteration
        self.path = []
        self.energy_list = []

    def generate_samples(self, n=None):
        """
        Generate n random points in the unit circle.

        Parameters
        ----------
        n : int, optional
            number of points (default is self.n_samples)

        Returns
        -------
        coordinates : numpy.ndarray
            array with the x- and y-coordinate of all the points ([n x 2])
        """
        if n is None:
            # Use the internal parameter
            n = self.n_samples
        else:
            self.n_samples = n

        coordinates = []
        for _ in range(n):
            x = np.random.uniform(low=-1, high=1)
            y = np.random.uniform(low=-1, high=1)

            # If the point is outside the circle, project it onto the edge
            if np.sqrt(x**2 + y**2) > 1:
                theta = np.arctan2(y, x)
                x = np.cos(theta)
                y = np.sin(theta)
            coordinates.append(np.array([x, y]))

        # Store the coordinates and energy
        self.coordinates = np.array(coordinates)
        self.energy = self.calc_energy()

        return np.array(coordinates)

    def calc_energy(self):
        """
        Calculate the energy of the current configuration.

        Returns
        -------
        tot_energy: float
            energy of the configuration
        """
        tot_energy = 0

        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                # Energy is 1/distance
                tot_energy += 1/np.linalg.norm(self.coordinates[i] - self.coordinates[j])

        return tot_energy

    def calc_force(self, index):
        """
        Calculate the force on 1 particle in the configuration.

        Parameters
        ----------
        index : int
            index of the particle

        Returns
        -------
        net_force : numpy.ndarray
            array with the normalized net force in the x- and y-direction
        """
        coordinate = self.coordinates[index]
        net_force = np.zeros((2))

        for i in range(self.n_samples):
            if i != index:
                distance = np.array(coordinate) - np.array(self.coordinates[i])
                net_force += distance / (np.linalg.norm(distance))**3

        return net_force / np.linalg.norm(net_force)

    def calc_temperature(self, i):
        """
        Calculate the temperature at a specific iteration.

        Parameters
        ----------
        i : int
            iteration

        Returns
        -------
        T : float
            temperature
        """
        if self.cooling_scheme == 'logarithmic':
            return 0.8 / (np.log(i + 2))
        elif self.cooling_scheme == 'hyperbolic_fast':
            return 1 / (0.1 * i + 2)
        else:
            return 3 / (0.008 * i + 2)

    def move_coord(self, i, index):
        """
        Determine the change in the x- and y-direction of a particle.

        Parameters
        ----------
        i : int
            iteration
        index : int
            index of the particle

        Returns
        -------
        dx, dy : float, float
            change in the x- and y-direction
        """
        dx = 0
        dy = 0

        # Change in the direction of the force (force and force_brownian method)
        if self.updating_method != 'brownian':
            net_force = self.calc_force(index)
            dx = net_force[0] * 0.2
            dy = net_force[1] * 0.2

        # Random change (brownian and force_brownian method)
        if self.updating_method != 'force':
            # Smaller range in a later stage of process
            range = 2 / (0.01 * i + 5)
            dx += np.random.uniform(low=-range, high=range)
            dy += np.random.uniform(low=-range, high=range)

        return dx, dy

    def update_coord(self, T, i):
        """
        Update the coordinates of every particle in a single iteration.

        Parameters
        ----------
        T : float
            temperature
        i : int
            iteration
        """
        for index, particle in enumerate(self.coordinates):
            # Current coordinates of the particle
            x = particle[0]
            y = particle[1]

            # Change in the coordinates
            dx, dy = self.move_coord(i, index)

            # New coordinates lie outside the circle
            if np.sqrt((x + dx)**2 + (y + dy)**2) > 1:

                # Project onto the circle
                theta = np.arctan2(y + dy, x + dx)
                particle[0] = np.cos(theta)
                particle[1] = np.sin(theta)

                # Energy of the new configuration
                energy = self.calc_energy()

            else:
                # Update coordinates
                particle[0] += dx
                particle[1] += dy

                # Energy of the new configuration
                energy = self.calc_energy()

            if energy > self.energy:
                # New configuration has a higher energy
                u = np.random.uniform()
                p_accept = np.exp((self.energy - energy) / T)
                if u > p_accept:
                    # Not accepted, reset coordinates
                    particle[0] = x
                    particle[1] = y
                else:
                    # Accept the new coordinates
                    self.energy = energy
            else:
                # New configuration has a lower energy
                self.energy = energy

    def optimize(self, iterations=5000):
        """
        Optimize the configuration.

        Parameters
        ----------
        iterations : int, optional
            total number of iterations (default is 5000)
        """
        # Store the initial coordinates and energy
        self.path.append(np.copy(self.coordinates))
        self.energy_list.append(self.energy)

        for i in range(iterations):
            # Update temperature and coordinates
            T = self.calc_temperature(i)
            self.update_coord(T, i)

            # Save the coordinates and energy
            self.path.append(self.coordinates)
            self.energy_list.append(self.energy)

def generate_optimal_circle(n, r = 1):
    """
    Calculate n equidistant points on a circle.

    Parameters
    ----------
    n : int
        number of points
    r : float, optional
        radius of the circle (default is 1)
    
    Returns
    -------
    coordinates : numpy.ndarray
        array with the coordinates of the points (n x 2)
    """
    # Calculate the distance
    avg_dist = (2 * np.pi) / n
    angles = np.arange(n) * avg_dist

    coordinates = np.zeros((2, n))
    coordinates[0] = r * np.cos(angles)
    coordinates[1] = r * np.sin(angles)

    return coordinates.T
