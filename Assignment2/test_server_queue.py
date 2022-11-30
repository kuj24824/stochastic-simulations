import unittest
import simpy
from server_queue import server_system


class test_server_queue(unittest.TestCase):

    def setUp(self):

        self.n_jobs = 5
        self.n_servers = 1
        self.arrival_rate = 2.0
        self.service_time = 3.0

        self.system = server_system(self.n_jobs, self.n_servers, self.arrival_rate, self.service_time)

    def test_n_jobs(self):

        self.system.run()
        self.assertEqual(self.system.n_jobs, self.n_jobs)
        self.assertEqual(len(self.system.waiting_times), self.n_jobs)

    def test_capacity(self):

        self.assertEqual(self.system.servers.capacity, self.n_servers)

        # Change the capacity
        self.system.set_n_servers(5)
        self.assertEqual(self.system.servers.capacity, 5)

    def test_queuing_model(self):

        self.assertEqual(self.system.queue_model, 'fifo')
        self.assertEqual(type(self.system.servers), simpy.resources.resource.Resource)

        # Change the queuing model
        self.assertRaises(NameError, self.system.set_queue_model, 'lifo')

        self.system.set_queue_model('Priority')
        self.assertEqual(type(self.system.servers), simpy.resources.resource.PriorityResource)

    def test_fifo(self):

        # 1 server
        self.system.arrival_process = 'deterministic'
        self.system.service_process = 'deterministic'
        self.system.run()
        self.assertEqual(self.system.waiting_times, [0, 2.5, 5, 7.5, 10])

        # 2 servers
        self.system.set_n_servers(2)
        self.system.run()
        self.assertEqual(self.system.waiting_times, [0, 0, 2, 2, 4])

    def test_determine_priority(self):

        service_time = 2
        priority = self.system.determine_priority(service_time)

        self.assertEqual(priority, 0)
        self.assertEqual(self.system.service_times, [2])
        self.assertEqual(self.system.priority_list, [0])

        service_time = 1
        priority = self.system.determine_priority(service_time)

        self.assertEqual(priority, -1)
        self.assertEqual(self.system.service_times, [1, 2])
        self.assertEqual(self.system.priority_list, [-1, 0])

        service_time = 8
        priority = self.system.determine_priority(service_time)

        self.assertEqual(priority, 1)
        self.assertEqual(self.system.service_times, [1, 2, 8])
        self.assertEqual(self.system.priority_list, [-1, 0, 1])

        service_time = 5
        priority = self.system.determine_priority(service_time)

        self.assertEqual(priority, .5)
        self.assertEqual(self.system.service_times, [1, 2, 5, 8])
        self.assertEqual(self.system.priority_list, [-1, 0, .5, 1])

        service_time = 2
        priority = self.system.determine_priority(service_time)

        self.assertEqual(priority, -.5)
        self.assertEqual(self.system.service_times, [1, 2, 2, 5, 8])
        self.assertEqual(self.system.priority_list, [-1, -.5, 0, .5, 1])


if __name__ == '__main__':
    unittest.main()
