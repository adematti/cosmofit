import numpy as np
from mpi4py import MPI
from mpytools import CurrentMPIComm

from .utils import BaseClass
from . import utils


def enum(*sequential, **named):
    """Enumeration values to serve as status tags passed between processes."""
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def split_ranks(nranks, nranks_per_worker, include_all=False):
    """
    Divide the ranks into chunks, attempting to have `nranks_per_worker` ranks
    in each chunk. This removes the root (0) rank, such
    that `nranks - 1` ranks are available to be grouped.

    Parameters
    ----------
    nranks : int
        The total number of ranks available.

    nranks_per_worker : int
        The desired number of ranks per worker.

    include_all : bool, optional
        if `True`, then do not force each group to have
        exactly `nranks_per_worker` ranks, instead including the remainder as well;
        default is `False`.
    """
    available = list(range(1, nranks))  # available ranks to do work
    total = len(available)
    extra_ranks = total % nranks_per_worker

    if include_all:
        for i, chunk in enumerate(np.array_split(available, max(total // nranks_per_worker, 1))):
            yield i, list(chunk)
    else:
        for i in range(total // nranks_per_worker):
            yield i, available[i * nranks_per_worker:(i + 1) * nranks_per_worker]

        i = total // nranks_per_worker
        if extra_ranks and extra_ranks >= nranks_per_worker // 2:
            remove = extra_ranks % 2  # make it an even number
            ranks = available[-extra_ranks:]
            if remove: ranks = ranks[:-remove]
            if len(ranks):
                yield i + 1, ranks


def barrier_idle(mpicomm, tag=0, sleep=0.01):
    """
    MPI barrier fonction that solves the problem that idle processes occupy 100% CPU.
    See: https://goo.gl/NofOO9.
    """
    size = mpicomm.size
    if size == 1: return
    rank = mpicomm.rank
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = mpicomm.isend(None, dst, tag)
        while not mpicomm.Iprobe(src, tag):
            time.sleep(sleep)
        mpicomm.recv(None, src, tag)
        req.Wait()
        mask <<= 1


class MPITaskManager(BaseClass):
    """
    A MPI task manager that distributes tasks over a set of MPI processes,
    using a specified number of independent workers to compute each task.

    Given the specified number of independent workers (which compute
    tasks in parallel), the total number of available CPUs will be
    divided evenly.

    The main function is ``iterate`` which iterates through a set of tasks,
    distributing the tasks in parallel over the available ranks.
    """
    @CurrentMPIComm.enable
    def __init__(self, nprocs_per_task=1, use_all_nprocs=False, mpicomm=None):
        """
        Initialize MPITaskManager.

        Parameters
        ----------
        nprocs_per_task : int, optional
            The desired number of processes assigned to compute each task.

        mpicomm : MPI communicator, optional
            The global communicator that will be split so each worker
            has a subset of CPUs available; default is COMM_WORLD.

        use_all_nprocs : bool, optional
            If `True`, use all available CPUs, including the remainder
            if `nprocs_per_task` does not divide the total number of CPUs
            evenly; default is `False`.
        """
        self.nprocs_per_task = nprocs_per_task
        self.use_all_nprocs = use_all_nprocs

        # the base communicator
        self.basecomm = mpicomm
        self.rank = self.basecomm.rank
        self.size = self.basecomm.size

        # need at least one
        if self.size == 1:
            raise ValueError('Need at least two processes to use a MPITaskManager')

        # communication tags
        self.tags = enum('READY', 'DONE', 'EXIT', 'START')

        # the task communicator
        self.mpicomm = None

        # store a MPI status
        self.status = MPI.Status()

    def __enter__(self):
        """
        Split the base communicator such that each task gets allocated
        the specified number of nranks to perform the task with.
        """
        self.self_worker_ranks = []
        color = 0
        total_ranks = 0
        nworkers = 0

        # split the ranks
        for i, ranks in split_ranks(self.size, self.nprocs_per_task, include_all=self.use_all_nprocs):
            if self.rank in ranks:
                color = i + 1
                self.self_worker_ranks = ranks
            total_ranks += len(ranks)
            nworkers = nworkers + 1
        self.other_ranks = [rank for rank in range(self.size) if rank not in self.self_worker_ranks]

        self.workers = nworkers  # store the total number of workers
        if self.rank == 0:
            self.log_info('Entering {} with {:d} workers.'.format(self.__class__.__name__, self.workers))

        # check for no workers!
        if self.workers == 0:
            raise ValueError('No pool workers available; try setting `use_all_nprocs` = True')

        leftover = (self.size - 1) - total_ranks
        if leftover and self.rank == 0:
            self.log_warning('With `nprocs_per_task` = {:d} and {:d} available rank(s), \
                             {:d} rank(s) will do no work'.format(self.nprocs_per_task, self.size - 1, leftover))
            self.log_warning('Set `use_all_nprocs=True` to use all available nranks')

        # crash if we only have one process or one worker
        if self.size <= self.workers:
            raise ValueError('Only have {:d} ranks; need at least {:d} to use the desired {:d} workers'.format(self.size, self.workers + 1, self.workers))

        # ranks that will do work have a nonzero color now
        self._valid_worker = color > 0

        # split the comm between the workers
        self.mpicomm = self.basecomm.Split(color, 0)

        return self

    def is_root(self):
        """
        Is the current process the root process?
        Root is responsible for distributing the tasks to the other available ranks.
        """
        return self.rank == 0

    def is_worker(self):
        """
        Is the current process a valid worker?
        Workers wait for instructions from the root.
        """
        try:
            return self._valid_worker
        except AttributeError:
            raise ValueError('Workers are only defined when inside the ``with MPITaskManager()`` context')

    def _get_tasks(self):
        """Internal generator that yields the next available task from a worker."""

        if self.is_root():
            raise RuntimeError('Root rank mistakenly told to await tasks')

        # logging info
        if self.mpicomm.rank == 0:
            self.log_debug('Worker root rank is {:d} on {} with {:d} processes available'.format(self.rank, MPI.Get_processor_name(), self.mpicomm.size))

        # continously loop and wait for instructions
        while True:
            args = None
            tag = -1

            # have the root rank of the subcomm ask for task and then broadcast
            if self.mpicomm.rank == 0:
                self.basecomm.send(None, dest=0, tag=self.tags.READY)
                args = self.basecomm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
                tag = self.status.Get_tag()

            # bcast to everyone in the worker subcomm
            args = self.mpicomm.bcast(args)  # args is [task_number, task_value]
            tag = self.mpicomm.bcast(tag)

            # yield the task
            if tag == self.tags.START:

                # yield the task value
                yield args

                # wait for everyone in task group before telling root this task is done
                self.mpicomm.Barrier()
                if self.mpicomm.rank == 0:
                    self.basecomm.send([args[0], None], dest=0, tag=self.tags.DONE)

            # see ya later
            elif tag == self.tags.EXIT:
                break

        # wait for everyone in task group and exit
        self.mpicomm.Barrier()
        if self.mpicomm.rank == 0:
            self.basecomm.send(None, dest=0, tag=self.tags.EXIT)

        # debug logging
        self.log_debug('Rank %d process is done waiting', self.rank)

    def _distribute_tasks(self, tasks):
        """Internal function that distributes the tasks from the root to the workers."""

        if not self.is_root():
            raise ValueError('only the root rank should distribute the tasks')

        ntasks = len(tasks)
        task_index = 0
        closed_workers = 0

        # logging info
        self.log_debug('root starting with {:d} worker(s) with {:d} total tasks'.format(self.workers, ntasks))

        # loop until all workers have finished with no more tasks
        while closed_workers < self.workers:

            # look for tags from the workers
            data = self.basecomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            source = self.status.Get_source()
            tag = self.status.Get_tag()

            # worker is ready, so send it a task
            if tag == self.tags.READY:

                # still more tasks to compute
                if task_index < ntasks:
                    this_task = [task_index, tasks[task_index]]
                    self.basecomm.send(this_task, dest=source, tag=self.tags.START)
                    self.log_debug('sending task `{}` to worker {:d}'.format(str(tasks[task_index]), source))
                    task_index += 1

                # all tasks sent -- tell worker to exit
                else:
                    self.basecomm.send(None, dest=source, tag=self.tags.EXIT)

            # store the results from finished tasks
            elif tag == self.tags.DONE:
                self.log_debug('received result from worker {:d}'.format(source))

            # track workers that exited
            elif tag == self.tags.EXIT:
                closed_workers += 1
                self.log_debug('worker {:d} has exited, closed workers = {:d}'.format(source, closed_workers))

    def iterate(self, tasks):
        """
        Iterate through a series of tasks in parallel.

        Notes
        -----
        This is a collective operation and should be called by all ranks.

        Parameters
        ----------
        tasks : iterable
            An iterable of `task` items that will be yielded in parallel
            across all ranks.

        Yields
        -------
        task :
            The individual items of `tasks`, iterated through in parallel.
        """
        # root distributes the tasks and tracks closed workers
        if self.is_root():
            self._distribute_tasks(tasks)

        # workers will wait for instructions
        elif self.is_worker():
            for tasknum, args in self._get_tasks():
                yield args

    def map(self, function, tasks):
        """
        Apply a function to all of the values in a list and return the list of results.

        If ``tasks`` contains tuples, the arguments are passed to
        ``function`` using the ``*args`` syntax.

        Notes
        -----
        This is a collective operation and should be called by
        all ranks.

        Parameters
        ----------
        function : callable
            The function to apply to the list.
        tasks : list
            The list of tasks.

        Returns
        -------
        results : list
            The list of the return values of ``function``.
        """
        results = []

        # root distributes the tasks and tracks closed workers
        if self.is_root():
            self._distribute_tasks(tasks)

        # workers will wait for instructions
        elif self.is_worker():

            # iterate through tasks in parallel
            for tasknum, args in self._get_tasks():

                # make function arguments consistent with *args
                if not isinstance(args, tuple):
                    args = (args,)

                # compute the result (only worker root needs to save)
                result = function(*args)
                if self.mpicomm.rank == 0:
                    results.append((tasknum, result))

        # put the results in the correct order
        results = self.basecomm.allgather(results)
        results = [item for sublist in results for item in sublist]
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit gracefully by closing and freeing the MPI-related variables."""

        if exc_value is not None:
            utils.exception_handler(exc_type, exc_value, exc_traceback)

        # wait and exit
        self.log_debug('Rank {:d} process finished'.format(self.rank))
        self.basecomm.Barrier()

        if self.is_root():
            self.log_debug('Root is finished; terminating')

        if self.mpicomm is not None:
            self.mpicomm.Free()
