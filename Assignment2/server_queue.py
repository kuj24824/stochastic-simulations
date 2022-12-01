"""Server system class to simulate discrete events in a queue."""

# Imports
import random
import simpy


class server_system:
    """
    A class that simulates jobs being processed by a server system.

    Attributes
    ----------
    n_jobs : int
        number of jobs that need to be processed
    n_servers : int
        number of servers in the system
    arrival_rate : float
        mean arrival rate of the jobs
    service_time : float
        mean service time per job
    env : simpy.core.Environment
        simpy environment
    queue_model : str
        type of the queue ("fifo" or "priority")
    arrival_process : str
        "markovian" or "deterministic" arrival rate
    service_process : str
        "markovian", "deterministic" or "hyperexponential" distribution of the service time
    servers : simpy.resources.resource.(Priority)Resource
        simpy resource object
    waiting_times : list
        list with the waiting times of all the jobs
    service_times : list
        list with the service times of the jobs in the queue (empty at the start and end of the simulation)
    priority_list : list
        list with the priority numbers of the jobs in the queue (empty at the sart and end of the simulation)

    Methods
    -------
    set_n_servers(n)
        Set the number of servers in the system.
    set_queue_model(model)
        Set the queuing model of the system.
    run()
        Runs a simulation of the system.
    """

    def __init__(self, n_jobs, n_servers, arrival_rate, service_time, arrival_process='markovian', service_process='markovian', queue_model='fifo'):
        """
        Initializes a server system.

        Parameters
        ----------
        n_jobs : int
            number of jobs that need to be processed
        n_servers : int
            number of servers in the system
        arrival_rate : float
            mean arrival rate of the jobs
        service_time : float
            mean service time per job
        arrival_process : str, optional
            "markovian" or "deterministic" arrival rate (default is markovian)
        service_process : str, optional
            "markovian", "deterministic" or "hyperexponential" distribution of the service time (default is markovian)
        queue_model : str, optional
            type of the queue (default is fifo, other option is priority)
        """
        # Settings of the queue
        self.n_jobs = n_jobs
        self.n_servers = n_servers
        self.arrival_rate = arrival_rate
        self.service_time = service_time

        self.env = simpy.Environment()

        # Queuing model
        self.queue_model = queue_model.lower()
        self.arrival_process = arrival_process.lower()
        self.service_process = service_process.lower()

        if self.arrival_process != 'markovian' and self.arrival_process != 'deterministic':
            raise NameError('Unexpected arrival process, "%s", expected "markovian" or "deterministic".' % arrival_process)
        if self.service_process != 'markovian' and self.service_process != 'deterministic' and self.service_process != 'hyperexponential':
            raise NameError('Unexpected service time distribution, "%s", expected "markovian", "deterministic" or "hyperexponential".' % service_process)

        # Create the queuing model of the server system
        if queue_model.lower() == 'fifo':
            self.servers = simpy.Resource(self.env, capacity=n_servers)
        elif queue_model.lower() == 'priority':
            self.servers = simpy.PriorityResource(self.env, capacity=n_servers)
        else:
            raise NameError('Unexpected queuing model, "%s", expected "fifo" or "priority".' % queue_model)

        # Lists to store important data
        self.waiting_times = []
        self.service_times = []
        self.priority_list = []

    def set_n_servers(self, n):
        """
        Set the number of servers in the system.

        Parameters
        ----------
        n : int
            number of servers in the system
        """
        self.n_servers = n
        # Adjust the capacity
        if self.queue_model == 'fifo':
            self.servers = simpy.Resource(self.env, capacity=n)
        elif self.queue_model == 'priority':
            self.servers = simpy.PriorityResource(self.env, capacity=n)

    def set_queue_model(self, model):
        """
        Set the queuing model of the system.

        Parameters
        ----------
        model, str
            type of the queue, "fifo" or "priority"
        """
        # Create the queuing model of the server system
        if model.lower() == 'fifo':
            self.servers = simpy.Resource(self.env, capacity=self.n_servers)
        elif model.lower() == 'priority':
            self.servers = simpy.PriorityResource(self.env, capacity=self.n_servers)
        else:
            raise NameError('Unexpected queuing model, "%s", expected "fifo" or "priority".' % model)
        self.queue_model = model.lower()

    def job_generator(self):
        """Generates jobs to be processed by the system."""
        for i in range(self.n_jobs):
            # Create job
            job = self.job_creator(i)
            self.env.process(job)

            # Determine the arrival of the next job
            if self.arrival_process == 'markovian':
                t = random.expovariate(self.arrival_rate)
            else:
                t = 1.0 / self.arrival_rate

            yield self.env.timeout(t)

    def job_creator(self, number):
        """
        Creates jobs to be processed by the system.

        Parameters
        ----------
        number : int
            jobnumber
        """
        # Arrival time
        t_arrival = self.env.now
        #print('Job %s: arrived at %s' % (number, t_arrival))

        # Determine the service time
        if self.service_process == 'markovian':
            t_service = random.expovariate(1 / self.service_time)
        elif self.service_process == 'deterministic':
            t_service = self.service_time
        else:
            # Hyperexponential
            x = random.uniform(0, 1)
            if x <= 0.75:
                t_service = random.expovariate(1 / self.service_time)
            else:
                t_service = random.expovariate(1 / (5 * self.service_time))

        #print(t_service)

        if self.queue_model == 'priority':
            place = self.determine_priority(t_service)
            # Add the job to the queue
            request = self.servers.request(priority=place)
        else:
            # Place the job at the end of the queue in the case of FIFO
            request = self.servers.request()
        yield request

        if self.queue_model == 'priority':
            # Remove jobs from the queue
            self.service_times.remove(t_service)
            self.priority_list.remove(place)

        # Determine the waiting time
        self.waiting_times.append(self.env.now - t_arrival)
        #print('Job %s: waited for %s' % (number, self.waiting_times[-1]))
        #print('Job %s: started at %s' % (number, self.env.now))

        # Execute the job
        yield self.env.timeout(t_service)
        self.servers.release(request)
        #print('Job %s: finished at %s' % (number, self.env.now))

    def determine_priority(self, t_service):
        """
        Determines the priority of the job in the queue.

        Parameters
        ----------
        t_service : float
            service time of the job that needs to be added to the queue
        """
        if len(self.service_times) == 0:
            # First job in the queue
            priority = 0
            place = 0

        elif t_service <= self.service_times[0]:
            # Shortest job so far
            priority = self.priority_list[0] - 1
            place = 0

        elif t_service >= self.service_times[-1]:
            # Longest job so far
            priority = self.priority_list[-1] + 1
            place = len(self.service_times)

        else:
            # Determine the place in the queue
            place = 0
            while t_service > self.service_times[place] and place < len(self.service_times) - 1:
                place += 1

            # Calculate the priority
            priority = self.priority_list[place - 1] + (self.priority_list[place] - self.priority_list[place - 1]) / 2

        # Store the service time and priority
        self.service_times.insert(place, t_service)
        self.priority_list.insert(place, priority)

        return priority

    def run(self):
        """Runs a simulation of the system."""
        # Empty list with the waiting times (if there was a previous run)
        self.waiting_times = []
        self.env.process(self.job_generator())
        self.env.run()
