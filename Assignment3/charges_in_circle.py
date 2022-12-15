import numpy as np
import matplotlib.pyplot as plt

class configuration():

    def __init__(self, n = None, scheme = 'logarithmic', method = 'brownian'):
        
        if n:
            self.n_samples = n
            self.coordinates = self.generate_samples()
            self.energy = self.calc_energy()
        else:
            # Initialize without charges
            self.n_samples = None
            self.coordinates = None
            self.energy = None

        if scheme != 'logarithmic' and scheme != 'hyperbolic_fast' and scheme != 'hyperbolic_slow':
            raise NameError("Unkown cooling scheme (%s) given, expected 'logarithmic', 'hyperbolic_fast' or 'hyperbolic_slow'." % scheme)
        self.cooling_scheme = scheme

        if method != 'brownian' and method != 'force' and method != 'force_brownian':
            raise NameError("Unkown method (%s) given, expected 'brownian', 'force' or 'force_brownian'." % method)
        self.updating_method = method

        self.path = []
        self.energy_list = []

    def generate_samples(self, n = None):

        if n is None:
            n = self.n_samples
        else:
            self.n_samples = n

        coordinates = []
        for _ in range(n):
            x = np.random.uniform(low = -1, high = 1)
            y = np.random.uniform(low = -1, high = 1)

            if np.sqrt(x**2 + y**2) > 1:
                theta = np.arctan2(y, x)
                x = np.cos(theta)
                y = np.sin(theta)
            coordinates.append(np.array([x, y]))
        
        return np.array(coordinates)
    
    def calc_energy(self):

        tot_energy = 0

        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                tot_energy += 1/np.linalg.norm(self.coordinates[i] - self.coordinates[j])

        return tot_energy

    def calc_force(self, index):

        coordinate = self.coordinates[index]
        net_force = np.zeros((2))

        for i in range(self.n_samples):
            if i != index:
                distance = np.array(coordinate) - np.array(self.coordinates[i])
                net_force += distance / (np.linalg.norm(distance))**3
            
        return net_force / np.linalg.norm(net_force)
    
    def calc_temperature(self, iteration):

        if self.cooling_scheme == 'logarithmic':
            return 0.8 / (np.log(iteration + 2))
        elif self.cooling_scheme == 'hyperbolic_fast':
            return 1/(0.1*iteration + 2)
        else:
            return 3/(0.008*iteration + 2)
    
    def move_coord(self, iteration, index):

        dx = 0
        dy = 0

        
        if self.updating_method != 'brownian':
            net_force = self.calc_force(index)
            dx = net_force[0] * 0.2
            dy = net_force[1] * 0.2
        
        if self.updating_method != 'force':
            range = 2/(0.01*iteration + 5)
            dx += np.random.uniform(low = -range, high = range)
            dy += np.random.uniform(low = -range, high = range)

        return dx, dy

    def update_coord(self, T, iteration):

        for index, particle in enumerate(self.coordinates):

            x = particle[0]
            y = particle[1]

            dx, dy = self.move_coord(iteration, index)

            if np.sqrt((x + dx)**2 + (y + dy)**2) > 1:

                theta = np.arctan2(y + dy, x + dx)
                particle[0] = np.cos(theta)
                particle[1] = np.sin(theta)
                energy = self.calc_energy()

            else:
                particle[0] += dx
                particle[1] += dy
                energy = self.calc_energy()

            if energy > self.energy:
                u = np.random.uniform()
                p_accept = np.exp((self.energy - energy) / T)
                if u > p_accept:
                    # Not accepted, reset coordinate
                    particle[0] = x
                    particle[1] = y
                else:
                    self.energy = energy
            
            else:
                self.energy = energy
    
    def plot(self):
        fig, axs = plt.subplots(1, figsize = (10, 10), sharex = True)
        
        circle1 = plt.Circle((0, 0), 1, edgecolor = 'red', facecolor = 'none')
        axs.add_patch(circle1)
        axs.scatter(np.transpose(self.coordinates)[0], np.transpose(self.coordinates)[1], color = 'black')
        axs.scatter(0, 0, color = 'blue')
        plt.show()

    def optimize(self, iterations = 5000):

        self.path.append(np.copy(self.coordinates))
        self.energy_list.append(self.energy)
        for i in range(iterations):
            T = self.calc_temperature(i)
            self.update_coord(T, i)
            self.path.append(self.coordinates)
            self.energy_list.append(self.energy)
