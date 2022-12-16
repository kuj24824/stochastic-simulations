import unittest
import numpy as np

from charges_in_circle import configuration
from charges_in_circle import generate_optimal_circle

class test_confiugration(unittest.TestCase):

    def setUp(self):

        self.config_1 = configuration(3)
        self.config_2 = configuration(5, scheme = 'hyperbolic_slow', method = 'force_brownian')
        self.config_3 = configuration(scheme = 'hyperbolic_fast', method = 'force')
        
        # Configuration with predetermined points
        self.config_4 = configuration()
        self.config_4.n_samples = 3
        self.config_4.coordinates = np.array(([np.array([0, 0]), np.array([1, 0]), np.array([0.5, 0.5])]))

    def test_n_particles(self):

        self.assertEqual(self.config_1.n_samples, 3)
        self.assertEqual(self.config_2.n_samples, 5)
        self.assertEqual(self.config_3.n_samples, None)
    
    def test_temperature(self):

        # Logarithmic
        self.assertEqual(self.config_1.calc_temperature(0), 1.1541560327111708)
        self.assertEqual(self.config_1.calc_temperature(2000), 0.10523682155105525)

        # Hyperbolic slow
        self.assertEqual(self.config_2.calc_temperature(0), 1.5)
        self.assertEqual(self.config_2.calc_temperature(2500), 0.13636363636363635)

        # Hyperbolic fast
        self.assertEqual(self.config_3.calc_temperature(0), 0.5)
        self.assertEqual(self.config_3.calc_temperature(3000), 0.0033112582781456954)
    
    
    def test_energy(self):

        self.assertEqual(self.config_4.calc_energy(), 3.82842712474619)
    
    def test_force(self):

        force = self.config_4.calc_force(0)
        self.assertEqual(force[0], -0.8628562094610168)
        self.assertEqual(force[1], -0.5054494651244235)

    def test_change_coord(self):
        
        iteration = 500

        # Brownian motion
        np.random.seed(0)
        dx, dy = self.config_1.move_coord(iteration, index = 0)
        self.assertEqual(dx, 0.019525401570929912)
        self.assertEqual(dy, 0.08607574654896777)

        # Force + brownian
        np.random.seed(0)
        dx, dy = self.config_2.move_coord(iteration, index = 0)

        force = self.config_2.calc_force(index = 0)
        x_force = force[0]
        y_force = force[1]

        self.assertEqual(dx, 0.019525401570929912 + x_force * 0.2)
        self.assertEqual(dy, 0.08607574654896777 + y_force * 0.2)

        # Force
        self.config_2.updating_method = 'force'
        dx, dy = self.config_2.move_coord(iteration, index = 0)
        self.assertEqual(dx, x_force * 0.2)
        self.assertEqual(dy, y_force * 0.2)

    def test_optimization(self):
        
        # Coordinates of 3 equidistant points
        coordinates = generate_optimal_circle(3)

        true_solution = configuration(3)
        true_solution.coordinates = coordinates
        true_energy = true_solution.calc_energy()

        self.config_3.generate_samples(3)
        self.config_3.optimize()
        self.assertAlmostEqual(self.config_3.energy, true_energy)

if __name__ == '__main__':
    unittest.main()
    