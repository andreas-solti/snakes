# coding: utf-8
from collections import defaultdict

__author__ = 'andreas'

from datetime import datetime, timedelta

from enum import Enum
import collections

import snakes.plugins
import pmlab
import pmlab.log
import pmlab.log.reencoders

snakes.plugins.load("gv", "snakes.nets", "nets")
from nets import *
#
# snakes.plugins.load('pos', 'snakes.nets')
# snakes.plugins.load('clusters', 'snakes.nets')
# snakes.plugins.load('gv', 'snakes.nets')
# snakes.plugins.load('timecpn', 'snakes.nets')

from snakes.plugins.timecpn import Distribution, DistributionType, TimeSimulator, TokenType, Token, Transition, Place
from snakes.nets import Expression, Variable, Tuple

def firings(*vargs, **kwargs):
    return False
    # max(0,(len(pat._history))/3) == num

class State(Enum):
    queued = 1
    service = 2
    finished = 3


class Action(Enum):
    enter = 1
    exit = 2


class FoldingOperation(Enum):
    remove_parallelism = 1
    remove_shared_resources = 2
    fuse_queues_and_resources = 3
    remove_individual_information = 4
    remove_resources = 5


class TaskSet(object):
    def __init__(self, net, task_set, structure):
        """Creates a new task set with the queuing construct.

        """
        self.net = net
        self.task_set = task_set
        self.structure = structure





class Unfolder(object):
    """The class that creates a Petri net from a schedule.
    Usage:
    >>> unfolder = Unfolder()
    # create the instance
    >>> unfolder.unfold(log)
    """

    def __init__(self):
        self.fuse_queues = False
        self.net = None
        self.resources_used = []
        self.resource_places = {}  # a dict storing the resources
        self.task_queues = {}  # a dict storing the structure of a task per resource combination
        # (i.e queuing place, entrance transition, service place, exit transition, finished place)
        pass

    def cleanup(self):
        self.net = None
        self.resources_used = []
        self.task_queues = {}
        self.resource_places = {}

    @staticmethod
    def get_overlapping_tasks(trace, sequentialize=False):
        """Transitively adds running tasks to the running set of tasks to build parallel groups
        of running tasks.
        Example tasks [A,B,C,D1,D2,E]: ( -> time -> )
        [-------A-------]
                 [-----B------]
                        [--C--]
                                [-D1-]    [----D2----]
                                            [-E-]
        will yield three sets of concurrently active tasks:
        [{A,B,C}, {D1}, {D2,E}]
        """
        overlapping_tasks = []
        current_time = 0
        currently_active_tasks = []  # list of concurrently planned tasks
        # end_time = datetime(1970,1,1,12,30)
        end_time = 0  # last time of the currently active concurrent tasks

        if sequentialize: # do not care about parallelism
            for task in iter(trace):
                overlapping_tasks.append([task])
        else: # assume tasks are ordered by start time!
            for task in iter(trace):
                task_start_time = task.get('timestamp', 0)
                if type(task_start_time) == str:
                    timestr = str(task_start_time[0:19])
                    # print timestr
                    task_start_time = datetime.strptime(timestr, '%Y-%m-%dT%H:%M:%S')
                duration = task.get('duration', 5)
                if duration > 0:
                    task_end_time = task_start_time + timedelta(seconds=duration).total_seconds()
                if end_time > task_start_time:  # belongs to the currently active set
                    name = Unfolder.get_coded(task['name'], task['resources'])
                    tasks = Unfolder.get_tasks(currently_active_tasks)
                    if name in tasks:
                        tasks[name]['end_time'] = max(tasks[name]['end_time'], task['end_time'])
                    else:
                        currently_active_tasks.append(task)

                    if task_end_time > end_time:
                        end_time = max(end_time, task_end_time)
                else:  # belongs to a new set (will become the only one currently enabled)
                    if currently_active_tasks:
                        overlapping_tasks.append(currently_active_tasks)
                    currently_active_tasks = [task]
                    end_time = task_end_time
            overlapping_tasks.append(currently_active_tasks)
        return overlapping_tasks

    @staticmethod
    def get_tasks(currently_active_tasks):
        tasks = {}
        for task in currently_active_tasks:
            name = Unfolder.get_coded(task['name'], task['resources'])
            tasks[name] = task
        return tasks

    @staticmethod
    def get_task_place_name(task, state):
        return "p{}_{}".format(task, state.name)

    @staticmethod
    def get_resource_place_name(resource):
        return "p_res_{}".format(resource)

    @staticmethod
    def get_resource_token_name(resource):
        return "r{}".format(resource)

    @staticmethod
    def get_transition_name(task, action):
        return "t{}_{}".format(task, action.name)

    @staticmethod
    def get_task_transition_name(task, nexttask):
        return "t{}_{}".format(task, nexttask)

    @staticmethod
    def get_coded(task='DUMMY', resources=[]):
        # coded_list = []
        # for i, key in enumerate(self.resources_used):
        #     if key in resources:
        #         coded_list.append(i)
        # return "{}._{}".format(task, ".".join(str(x) for x in coded_list))
        return "{}.{}".format(task, "".join(sorted(resources)))

    def connect_tasks(self, last_task_connectors, new_task_connectors, task_start, trace_name,
                      trace_token, task_counts, scheduled=True, split=False):
        """Establishes a routing connection for a certain trace
        """
        first = not last_task_connectors[0]
        if first:  # the first task gets the token
            # trace_token.set_time(task_start)
            last_task_connectors[1].add(trace_token)

        # add a routing scheduled transition from the last task
        p_last_finished = last_task_connectors[1]
        if self.fuse_queues:
            t_name = '{}_{}'.format(p_last_finished.get_name(), new_task_connectors[0].get_name())
        else:
            t_name = '{}_{}_{}'.format(trace_name, p_last_finished.get_name(), new_task_connectors[0].get_name())
        expression = self.get_expression(trace_name, last_task_connectors, task_counts, split=split)
        try:
            mytransition = self.net.transition(t_name)
            expr = Expression(str(mytransition.guard) + " or " + expression)
            expr.globals.update(mytransition.guard.globals)
            mytransition.guard = expr
            # trace revisits activity/resource pair
            pass
        except ConstraintError:
            expr = Expression(expression)
            #expr = Expression("pat=='{}'".format(trace_name))
            if scheduled:
                self.net.add_transition(Transition(t_name, guard=expr,
                                                          dist=Distribution(DistributionType.scheduled,
                                                                                   mintime=task_start)))
            else:
                self.net.add_transition(Transition(t_name, guard=expr,
                                                          dist=Distribution(DistributionType.immediate)))
            self.net.add_input(p_last_finished.get_name(), t_name, Variable('pat'))
            self.net.add_output(new_task_connectors[0].get_name(), t_name, Expression('pat'))


        return new_task_connectors

    def unfold(self, schedule_log, fuse_queues=False, scheduled_start=False, scheduled=True, resource_capacity={}):
        """Unfolds the log into a colored Petri net model
        capturing resource dependencies

        returns: - the net,
                 - the list of tasks sets (for all traces),
                 - the task queues assigning each task_name to the net constructs (queuing stations)
        """
        self.cleanup()
        self.fuse_queues = fuse_queues

        self.net = PetriNet(schedule_log.filename)
        self.net.globals.update(Evaluator(firings=firings))
        task_sets = []
        trace_counter = 0

        arrival_place = Place("arrivals")
        self.net.add_place(arrival_place)
        exit_place = Place("exits")
        self.net.add_place(exit_place)

        for trace in iter(schedule_log.get_cases(full_info=True)):
            trace_counter += 1
            trace_name = trace[0].get('traceId', 'pat{}'.format(trace_counter))
            if scheduled_start:
                trace_token = Token(trace_name, time=trace[0].get('timestamp', 0),
                                           schedule=self.get_schedule(trace), type=TokenType.instance,
                                           start_end=Unfolder.get_start_end(trace))
            else:
                trace_token = Token(trace_name, time=trace[0].get('start_time', 0),
                                           schedule=self.get_schedule(trace), type=TokenType.instance,
                                           start_end=Unfolder.get_start_end(trace))

            last_task_connectors = [None, arrival_place, 'tArrival']
            concurrent_task_sets = Unfolder.get_overlapping_tasks(trace)
            task_counts = defaultdict(int)
            for step, task_set in enumerate(concurrent_task_sets):
                task = task_set[0]
                task_start = task.get('timestamp', 0)
                if len(task_set) > 1:
                    p_entrance = Place("{}_in_{}".format(trace_name, step))
                    self.net.add_place(p_entrance)
                    p_exit = Place("{}_out_{}".format(trace_name, step))
                    self.net.add_place(p_exit)

                    ins_and_outs = []
                    new_task_names = []
                    for task in task_set:
                        new_task_connectors = self.add_or_wire(task, trace_name, trace_token, last_task_connectors,
                                                               trace_counter, step,
                                                               resource_capacity=resource_capacity)
                        name = Unfolder.get_coded(task['name'], task['resources'])
                        task_counts[name] += 1
                        new_task_names.append(name)
                        ins_and_outs.append(new_task_connectors)

                    # create the split transition + scheduling transitions
                    t_name = "split_{}_{}".format(trace_name, step)
                    self.net.add_transition(
                        Transition(t_name, guard=Expression(self.get_expression(trace_name, last_task_connectors, task_counts, split=last_task_connectors[2] in new_task_names)),
                                          dist=Distribution(DistributionType.immediate)))
                    self.net.add_input(p_entrance.get_name(), t_name, Variable('pat'))
                    for i, in_and_out in enumerate(ins_and_outs):
                        this_task = task_set[i]
                        p_split = Place("{}_splitted_{}_{}".format(trace_name, step, i))
                        self.net.add_place(p_split)
                        self.net.add_output(p_split.get_name(), t_name, Expression('pat'))
                        parallel_task_connectors = self.connect_tasks([p_entrance, p_split, t_name], in_and_out,
                                           this_task.get('timestamp', 0), trace_name, trace_token, task_counts,
                                           scheduled=scheduled)
                    # create the join transition to synchronize continuations
                    t_name = "join_{}_{}".format(trace_name, step)
                    self.net.add_transition(
                        Transition(t_name, guard=Expression(self.get_expression(trace_name, parallel_task_connectors, task_counts)),
                                          dist=Distribution(DistributionType.immediate)))
                    for in_and_out in ins_and_outs:
                        try:
                            self.net.add_input(in_and_out[1].get_name(), t_name, Variable('pat'))
                        except ConstraintError:
                            print "is some thing wrong with this resource?"
                    self.net.add_output(p_exit.get_name(), t_name, Expression('pat'))
                    new_task_connectors = tuple([p_entrance, p_exit, parallel_task_connectors[2]])
                    last_task_connectors = self.connect_tasks(last_task_connectors, new_task_connectors,
                                                              task_start, trace_name, trace_token,
                                                              task_counts, scheduled=scheduled, split=last_task_connectors[2] in new_task_names)
                    # for task in task_set:
                    #     task_counts[self.get_coded(task['name'], task['resources'])] += 1
                else:
                    new_task_connectors = self.add_or_wire(task, trace_name, trace_token, last_task_connectors,
                                                           trace_counter, step,
                                                           resource_capacity=resource_capacity)
                    last_task_connectors = self.connect_tasks(last_task_connectors, new_task_connectors,
                                                              task_start, trace_name, trace_token,
                                                              task_counts, scheduled=scheduled)
                    task_counts[Unfolder.get_coded(task['name'], task['resources'])] += 1
                task_sets.append(TaskSet(self.net, task_set, new_task_connectors))
            # after setting up the patient path, we add an immediate transition to the exit place
            t_name = "exit.{}".format(trace_counter)
            self.net.add_transition(Transition(t_name, guard=Expression(self.get_expression(trace_name, last_task_connectors, task_counts)),
                                                      dist=Distribution(DistributionType.immediate)))
            self.net.add_input(last_task_connectors[1].get_name(), t_name, Variable('pat'))
            self.net.add_output(exit_place.get_name(), t_name, Expression('pat'))

        return self.net  # , task_sets, self.task_queues

    def add_or_wire(self, task, trace_name, trace_token, last_task_connectors, trace_counter, trace_step,
                    resource_capacity={}):

        task_duration = task.get('duration', 1)
        resources = task.get('resources', [])

        # gather used resources
        task_resource_places = {}
        for res in resources:
            if res not in self.resource_places:
                place = Place(self.get_resource_place_name(res), [])
                if '$' in res:
                    # siamezation:
                    resources_to_join = str(res).split("$")
                    res_capacity = 100000000
                    for resource_to_join in resources_to_join:
                        res_capacity = min(res_capacity, resource_capacity.get(resource_to_join, 1))
                else:
                    res_capacity = resource_capacity.get(res, 1)
                for r in range(1, res_capacity + 1):
                    tok = Token(self.get_resource_token_name(res), time=0.0)
                    place.add(tok)
                self.net.add_place(place)
                self.resource_places[res] = place
                self.resources_used.append(res)
            task_resource_places[res] = self.resource_places.get(res)

        if self.fuse_queues:
            key = Unfolder.get_coded(task.get('name', 'DUMMY'), resources)
        else:
            task_name = "{}.{}.{}".format(Unfolder.get_coded(task.get('name', 'DUMMY'), resources), trace_counter,
                                          trace_step)
            key = task_name

        # add task to net
        if key not in self.task_queues:
            # create places for task:
            p_names = {State.queued: self.get_task_place_name(key, State.queued),
                       State.service: self.get_task_place_name(key, State.service),
                       State.finished: self.get_task_place_name(key, State.finished)}
            p_queue = Place(p_names[State.queued], [])
            self.net.add_place(p_queue)
            p_service = Place(p_names[State.service], [])
            self.net.add_place(p_service)
            p_finish = Place(p_names[State.finished], [])
            self.net.add_place(p_finish)
            # create transitions for task:
            # enter:
            t_names = {Action.enter: self.get_transition_name(key, Action.enter),
                       Action.exit: self.get_transition_name(key, Action.exit)}
            self.net.add_transition(Transition(t_names[Action.enter], guard=Expression("True"),
                                                      dist=Distribution(DistributionType.deterministic,
                                                                               time=task.get('duration', 1))))
            self.net.add_input(p_names[State.queued], t_names[Action.enter], Variable('pat'))
            tuple_parts = [Expression('pat')]
            variables = [Variable('pat')]
            counter = 0
            for res, task_res in task_resource_places.iteritems():
                counter += 1
                res_name = 'res{}'.format(counter)
                self.net.add_input(self.get_resource_place_name(res), t_names[Action.enter], Variable(res_name))
                tuple_parts.append(Expression(res_name))
                variables.append(Variable(res_name))
            self.net.add_output(p_names[State.service], t_names[Action.enter], Tuple(tuple_parts))
            # exit:
            self.net.add_transition(Transition(t_names[Action.exit], guard=Expression("True"),
                                                      dist=Distribution(DistributionType.immediate)))
            self.net.add_input(p_names[State.service], t_names[Action.exit], Tuple(variables))
            self.net.add_output(p_names[State.finished], t_names[Action.exit], Expression('pat'))
            counter = 0
            for res, task_res in task_resource_places.iteritems():
                counter += 1
                res_name = 'res{}'.format(counter)
                self.net.add_output(self.get_resource_place_name(res), t_names[Action.exit], Variable(res_name))
            self.task_queues[key] = {State.queued: p_queue,
                                     State.service: p_service,
                                     State.finished: p_finish,
                                     Action.enter: t_names[Action.enter],
                                     Action.exit: t_names[Action.exit]}
        else:  # task is already added to the net.
            # TODO: make sensitive to multiple resource configurations
            pass
        p_queue = self.task_queues[key][State.queued]
        p_finish = self.task_queues[key][State.finished]
        return tuple([p_queue, p_finish, key])

    def get_schedule(self, trace):
        schedule = []
        for task in trace:
            key = Unfolder.get_coded(task.get('name', 'DUMMY'), task.get('resources', []))
            schedule.append(key)
        return schedule

    @staticmethod
    def get_start_end(trace):
        start = trace[0].get('start_time', 0)
        end = trace[-1].get('end_time', 0)
        return [start, end]

    def get_expression(self, trace_name, last_task_connectors, task_counts, split=False):
        before = 1 if split else 0
        return "pat=='{}' and firings(pat,'{}',{})".format(trace_name, self.get_transition_name(last_task_connectors[2],Action.enter), task_counts[last_task_connectors[2]]-before)


class Folder(object):
    def __init__(self):
        self.net = None
        self.task_sets = None
        self.task_queues = None

    def cleanup(self):
        self.net = None
        self.task_sets = None
        self.task_queues = None

    @staticmethod
    def fold_log(log, operation):
        new_traces = []
        if operation == FoldingOperation.remove_parallelism:
            for trace in iter(log.get_cases(full_info=True)):
                new_trace = []
                concurrent_task_sets = Unfolder.get_overlapping_tasks(trace)
                for task_set in iter(concurrent_task_sets):
                    if len(task_set) > 1:
                        task_merged_name = ""  # build task name
                        resources = set([])
                        duration = 0
                        times = []
                        start_time = sys.maxint
                        end_time = 0
                        for task in task_set:
                            task_merged_name = task_merged_name + task.get('name', 'DUMMY')
                            task_resources = task.get('resources', [])
                            task_duration = task.get('duration', 1)
                            resources = resources.union(set(task_resources))
                            duration = max(duration, task_duration)
                            times.append(task.get('timestamp', 0))
                            start_time = min(start_time, task.get('start_time'))
                            end_time = max(end_time, task.get('end_time'))
                        timestamp = min(times)
                        new_trace.append({'name': task_merged_name, 'timestamp': timestamp, 'duration': duration,
                                          'resources': resources, 'start_time': start_time, 'end_time': end_time})
                    else:
                        new_trace.append(task_set[0])
                new_traces.append(new_trace)
        elif operation == FoldingOperation.remove_shared_resources:
            for trace in iter(log.get_cases(full_info=True)):
                new_trace = []
                concurrent_task_sets = Unfolder.get_overlapping_tasks(trace)
                for task_set in iter(concurrent_task_sets):
                    for task in task_set:
                        task_resources = task.get('resources', [])
                        if len(task_resources) > 1:
                            # merge them
                            new_resource = "$".join(sorted(task_resources))
                            task['resources'] = [new_resource]
                        new_trace.append(task)
                new_traces.append(new_trace)
        elif operation == FoldingOperation.remove_resources:
            for trace in iter(log.get_cases(full_info=True)):
                new_trace = []
                for task in trace:
                    new_trace.append({'name': task.get('name', 'DUMMY'), 'timestamp': task.get('timestamp', 0),
                                      'duration': task.get('duration', 1), 'resources': [],
                                      'start_time': task.get('start_time'), 'end_time': task.get('end_time')})
                new_traces.append(new_trace)
        return pmlab.log.EnhancedLog(filename=log.filename, cases=new_traces)

    def fold(self, net, task_sets, task_queues, operation):
        """Folds a net according to an operation FoldingOperation
         task_sets contains All task sets in PI of all traces
         task_queues contains the corresponding model structures
         Idea is to return again the triple of the (folded) net, the remaining task_sets, and the remaining task_queues
        """
        self.net = net
        self.task_sets = task_sets
        self.task_queues = task_queues
        if operation == FoldingOperation.remove_parallelism:
            self.remove_parallelism()
        elif operation == FoldingOperation.remove_shared_resources:
            self.remove_shared_resources()
        elif operation == FoldingOperation.fuse_queues_and_resources:
            self.merge_resources()
            self.fuse_queues()
        elif operation == FoldingOperation.remove_individual_information:
            self.remove_individual_information()
        return self.net, self.task_sets, self.task_queues

    def remove_parallelism(self):
        """Removes concurrency by joining all parallel tasks into a single big task.
         (Time is computed as the maximum of individual tasks)
        """
        pass
        # for i, task_set in enumerate(self.task_sets):
        #     if len(task_set) > 1:
        #         task_merged_name = ""  # build task name
        #         for task in task_set:
        #             task_merged_name = task_merged_name + task.get('name', 'DUMMY')
        #         self.net.add_transition(Transition(t_names[Action.enter], guard=Expression("True"), dist=Distribution(DistributionType.exponential, rate=1)))
        #         # merged_task =
        #         new_task_set = {}
        #         self.task_sets[i] = self.new_task_set
        #     else:
        #         pass  # nothing to do for single tasks

    def remove_shared_resources(self):
        """Removes shared resources by joining them to one (copied) resource
        """
        # todo implement
        pass

    def merge_queues(self):
        """Merges queues of the same task/resource
        adds routing transitions after merging the queues
        """
        # todo implement
        pass

    def merge_resources(self):
        """Merges resources of the same kind -> here "kind" is defined by the role these resources play
         that is, all resources performing "blood draw" will be collected in one single place with multiple tokens
        """
        # todo implement
        pass

    def remove_individual_information(self):
        """Fuses queues of the same task/resource
        adds routing transitions after merging the queues
        """
        # todo implement
        pass


class Enricher(object):
    def __init__(self, distribution_type):
        self._dist_type = distribution_type

    def enrich(self, cpn, log, resources=None, time_factor=1):
        # for all task/resource combinations
        durations = {}  # stores real durations per key (task x resource²)
        waitingTimes = defaultdict(list) # stores waiting times per key (task x resource²)
        scheduled_durations = {}  # stores scheduled durations per key (task x resource²)
        for trace in log:
            pos = 0
            lastEvent = None
            for event in trace:
                waitingTime = get_waiting_time(trace, event, pos)
                pos += 1
                key = "{}.{}".format(event['name'], "".join(sorted(event['resources'])))
                waitingTimes[key].append(waitingTime)

                key = event['name']
                value = event.get('end_time', 0) - event.get('start_time', 0)
                if key not in durations.keys():
                    durations[key] = [value]
                    scheduled_durations[key] = [event.get('duration')]
                else:
                    durations[key].append(value)
                    scheduled_durations[key].append(event.get('duration'))
                lastEvent = event
        for key in durations:
            durations[key] = [duration * time_factor for duration in durations[key]]

        if resources is not None:
            for place in cpn.place():
                if place.name.startswith('p_res_'):
                    placeResource = place.name[6:]
                    if placeResource in resources and len(place.get_tokens(0)) > 0:
                        place.get_tokens(0).items()[0].resource = resources[placeResource]
                        place.resource = resources[placeResource]

        for transition in cpn.transition():
            t_name = str(transition)
            if t_name.endswith("_enter"):
                # enrich transition with historical waitingTimes
                key = t_name[1:-6]  # cut off the "t" and the "_enter"
                if key in waitingTimes.keys():
                    transition.waitingTimes = waitingTimes.get(key)
                else:
                    transition.waitingTimes = [0]

                # enrich transition with historical durations
                key = ".".join(t_name[1:-6].split(".")[:1])
                if key not in durations.keys():
                    values = [0]
                    pass
                else:
                    values = durations[key]
                dist = None
                if len(values) < 2 or self.same_values(values):
                    if key in scheduled_durations:
                        dist = Distribution(dist_type=DistributionType.exponential,
                                                   rate=1.0 / scheduled_durations[key][0])
                        # if values[0] > 0:
                        #     dist = Distribution(dist_type=DistributionType.exponential, rate=1./values[0])
                        # else:
                        #     dist = Distribution(dist_type=DistributionType.exponential, rate=1)
                else:
                    dist = Distribution(dist_type=DistributionType.empirical, fit=True, values=values)
                if dist is not None:
                    transition.set_dist(dist)
        return cpn

    def same_values(self, values):
        if len(values) == 0:
            return True
        value = values[0]
        for v in values:
            if v != value:
                return False
        return True

def get_waiting_time(trace, event, pos):
    """Naive waiting time computation. Finds the last event in the trace that ended before the current activity."""
    if (pos == 0):
        return 0
    waiting_time = event.get('start_time',0) - trace[pos].get('end_time',0)
    if waiting_time < 0:
        return get_waiting_time(trace, event, pos-1)
    else:
        return waiting_time


def normalize_resources(log):
    """Replaces duplicate log events that use different resources with one entry that uses the set of resources
    assumption: resources are stored in the "resource" key
    """
    # go through all cases
    resource_capacity = collections.defaultdict(int)  # resource_capacity['RNTanya'] = 5
    all_events = []

    for case in log:
        last_event = None
        for event in case:
            all_events.append({'timestamp': event.get('timestamp'), 'resource': event.get('resource'), 'in': True})
            all_events.append({'timestamp': (int(event.get('timestamp')) + int(event.get('duration'))),
                               'resource': event.get('resource'), 'in': False})
            if 'resource' in event:
                event['resources'] = set()
                if last_event and last_event['name'] == event['name'] and last_event['end_time'] > event['start_time']:
                    # TODO: make sure that also times match!
                    last_event['delete'] = True
                    event['resources'] = set(last_event['resources'])
                if event['resource']:
                    event['resources'].add(str(event['resource']))
                del event['resource']
                last_event = event
        case[:] = [evt for evt in case if 'delete' not in evt]

    all_events_sorted = sorted(all_events, key=lambda k: k['timestamp'])
    current_utilization = collections.defaultdict(int)
    for ev in all_events_sorted:
        if isinstance(ev.get('resource'), list):
            # empty
            pass
        else:
            if ev.get('in'):
                current_utilization[ev.get('resource')] += 1
                resource_capacity[ev.get('resource')] = max(current_utilization[ev.get('resource')],
                                                            resource_capacity[ev.get('resource')])
            else:
                current_utilization[ev.get('resource')] -= 1

    return log, resource_capacity


def extract_task_name(task_name):
    """Returns the name of a task without the id that is appended.
    Example: given "BloodDraw.72" it returns "BloodDraw"
    Used in fusing tasks that have the same name
    """
    trimmed = re.sub('\.[0-9]+$', '', task_name)
    trimmed

@snakes.plugins.plugin("snakes.nets",
                       depends=["snakes.nets", "snakes.plugins.gv", "snakes.plugins.timecpn"])
def extend(module):
    # class State(Enum):
    #     queued = 1
    #     service = 2
    #     finished = 3
    #
    # class Unfolder(object):
    #     """The class that creates a Petri net from a schedule.
    #     Usage:
    #     >>> unfolder = Unfolder()
    #     # create the instance
    #     >>> unfolder.unfold(log)
    #     """
    #
    #     def __init__(self):
    #         self.net = None
    #         self.resources_used = []
    #         self.resource_places = {}  # a dict storing the resources
    #         self.task_queues = {}  # a dict storing the structure of a task per resource combination
    #         # (i.e queuing place, entrance transition, service place, exit transition, finished place)
    #         pass
    #
    #     def cleanup(self):
    #         self.net = None
    #         self.resources_used = []
    #         self.task_queues = {}
    #         self.resource_places = {}
    #
    #     def get_overlapping_tasks(self, trace):
    #         """Transitively adds running tasks to the running set of tasks to build parallel groups
    #         of running tasks.
    #         Example tasks [A,B,C,D1,D2,E]: ( -> time -> )
    #         [-------A-------]
    #                  [-----B------]
    #                         [--C--]
    #                                 [-D1-]    [----D2----]
    #                                             [-E-]
    #         will yield three sets of concurrently active tasks:
    #         [{A,B,C}, {D1}, {D2,E}]
    #         """
    #         overlapping_tasks = []
    #         current_time = 0
    #         currently_active_tasks = []  # list of concurrently planned tasks
    #         # end_time = datetime(1970,1,1,12,30)
    #         end_time = 0  # last time of the currently active concurrent tasks
    #
    #         # assume tasks are ordered by start time!
    #         for task in iter(trace):
    #             task_start_time = task.get('timestamp', 0)
    #             if type(task_start_time) == str:
    #                 timestr = str(task_start_time[0:19])
    #                 print timestr
    #                 task_start_time = datetime.strptime(timestr, '%Y-%m-%dT%H:%M:%S')
    #             duration = task.get('duration', 5)
    #             if duration > 0:
    #                 task_end_time = task_start_time + timedelta(seconds=duration).total_seconds()
    #             if end_time > task_start_time:  # belongs to the currently active set
    #                 currently_active_tasks.append(task)
    #                 if task_end_time > end_time:
    #                     end_time = max(end_time, task_end_time)
    #             else:  # belongs to a new set (will become the only one currently enabled)
    #                 if currently_active_tasks:
    #                     overlapping_tasks.append(currently_active_tasks)
    #                 currently_active_tasks = [task]
    #                 end_time = task_end_time
    #         overlapping_tasks.append(currently_active_tasks)
    #         return overlapping_tasks
    #
    #     @staticmethod
    #     def get_task_place_name(task, state):
    #         return "p{}_{}".format(task, state.name)
    #
    #     @staticmethod
    #     def get_resource_place_name(resource):
    #         return "p_res_{}".format(resource)
    #
    #     @staticmethod
    #     def get_resource_token_name(resource):
    #         return "r{}".format(resource)
    #
    #     @staticmethod
    #     def get_transition_name(task, action):
    #         return "t{}_{}".format(task, action.name)
    #
    #     @staticmethod
    #     def get_task_transition_name(task, nexttask):
    #         return "t{}_{}".format(task, nexttask)
    #
    #     def get_coded(self, task='DUMMY', resources=[]):
    #         coded_list = []
    #         for i, key in enumerate(self.resources_used):
    #             if key in resources:
    #                 coded_list.append(i)
    #         return "{}_{}".format(task, ".".join(str(x) for x in coded_list))
    #
    #     def connect_tasks(self, first, last_task_connectors, new_task_connectors, task_start, trace_name, trace_token):
    #         """Establishes a routing connection for a certain trace
    #         """
    #         if first:  # the first task gets the token
    #             trace_token.set_time(task_start)
    #             new_task_connectors[0].add(trace_token)
    #         else:  # add a routing scheduled transition from the last task
    #             p_last_finished = last_task_connectors[1]
    #             t_name = '{}_{}_{}'.format(trace_name, p_last_finished.get_name(), new_task_connectors[0].get_name())
    #             self.net.add_transition(
    #                 module.Transition(t_name, guard=module.Expression("pat=='{}'".format(trace_name)),
    #                                   dist=module.Distribution("scheduled", mintime=task_start)))
    #             self.net.add_input(p_last_finished.get_name(), t_name, module.Variable('pat'))
    #             self.net.add_output(new_task_connectors[0].get_name(), t_name, module.Expression('pat'))
    #         return new_task_connectors
    #
    #     def unfold(self, schedule_log):
    #         """Unfolds the log into a colored Petri net model
    #         capturing resource dependencies
    #
    #         """
    #         self.cleanup()
    #
    #         self.net = module.PetriNet(schedule_log.filename)
    #         existing_tasks = {}
    #         trace_counter = 0
    #         for trace in iter(schedule_log.get_cases(full_info=True)):
    #             trace_counter += 1
    #             trace_name = 'pat{}'.format(trace_counter)
    #             trace_token = module.Token(trace_name, time=trace[0].get('timestamp', 0))
    #
    #             last_task_connectors = None
    #             # TODO: preprocess log to identify parallel tasks!!!
    #             concurrent_task_sets = self.get_overlapping_tasks(trace)
    #             concurrent_set_id = 0
    #             for task_set in iter(concurrent_task_sets):
    #                 first = not last_task_connectors
    #                 task = task_set[0]
    #                 task_start = task.get('timestamp', 0)
    #
    #                 if len(task_set) > 1:
    #                     concurrent_set_id += 1
    #                     p_entrance = module.Place("{}_in_{}".format(trace_name, concurrent_set_id))
    #                     self.net.add_place(p_entrance)
    #                     p_exit = module.Place("{}_out_{}".format(trace_name, concurrent_set_id))
    #                     self.net.add_place(p_exit)
    #
    #                     ins_and_outs = []
    #                     for task in task_set:
    #                         new_task_connectors = self.add_or_wire(task, trace_name, trace_token, last_task_connectors)
    #                         ins_and_outs.append(new_task_connectors)
    #
    #                     # create the split transition + scheduling transitions
    #                     t_name = "split_{}_{}".format(trace_name, concurrent_set_id)
    #                     self.net.add_transition(module.Transition(t_name, guard=module.Expression("True"),
    #                                                               dist=module.Distribution("immediate")))
    #                     self.net.add_input(p_entrance.get_name(), t_name, module.Variable('pat'))
    #                     for in_and_out in ins_and_outs:
    #                         self.net.add_output(in_and_out[0].get_name(), t_name, module.Expression('pat'))
    #                         # TODO: add scheduled transitions in between here! (it's a mess...)
    #
    #                     # create the join transition to synchronize continuations
    #                     t_name = "join_{}_{}".format(trace_name, concurrent_set_id)
    #                     self.net.add_transition(module.Transition(t_name, guard=module.Expression("True"),
    #                                                               dist=module.Distribution("immediate")))
    #                     for in_and_out in ins_and_outs:
    #                         self.net.add_input(in_and_out[1].get_name(), t_name, module.Variable('pat'))
    #                     self.net.add_output(p_exit.get_name(), t_name, module.Expression('pat'))
    #                     new_task_connectors = tuple([p_entrance, p_exit])
    #                     last_task_connectors = self.connect_tasks(first, last_task_connectors, new_task_connectors,
    #                                                               task_start, trace_name, trace_token)
    #                 else:
    #                     new_task_connectors = self.add_or_wire(task, trace_name, trace_token, last_task_connectors)
    #                     last_task_connectors = self.connect_tasks(first, last_task_connectors, new_task_connectors,
    #                                                               task_start, trace_name, trace_token)
    #         return self.net
    #
    #     def add_or_wire(self, task, trace_name, trace_token, last_task_connectors):
    #         task_name = task.get('name', 'DUMMY')
    #         task_duration = task.get('duration', 1)
    #         resources = task.get('resources', [])
    #
    #         # gather used resources
    #         task_resource_places = {}
    #         for res in resources:
    #             if res not in self.resource_places:
    #                 place = module.Place(self.get_resource_place_name(res), [])
    #                 tok = module.Token(self.get_resource_token_name(res), time=0.0)
    #                 place.add(tok)
    #                 self.net.add_place(place)
    #                 self.resource_places[res] = place
    #                 self.resources_used.append(res)
    #             task_resource_places[res] = self.resource_places.get(res)
    #
    #         key = self.get_coded(task_name, resources)
    #
    #         # add task to net
    #         if key not in self.task_queues:
    #             # create places for task:
    #             p_names = {State.queued: self.get_task_place_name(key, State.queued),
    #                        State.service: self.get_task_place_name(key, State.service),
    #                        State.finished: self.get_task_place_name(key, State.finished)}
    #             p_queue = module.Place(p_names[State.queued], [])
    #             self.net.add_place(p_queue)
    #             p_service = module.Place(p_names[State.service], [])
    #             self.net.add_place(p_service)
    #             p_finish = module.Place(p_names[State.finished], [])
    #             self.net.add_place(p_finish)
    #             # create transitions for task:
    #             # enter:
    #             t_names = {module.Action.enter: self.get_transition_name(key, module.Action.enter),
    #                        module.Action.exit: self.get_transition_name(key, module.Action.exit)}
    #             self.net.add_transition(module.Transition(t_names[module.Action.enter], guard=module.Expression("True"),
    #                                                       dist=module.Distribution("exponential", rate=1)))
    #             self.net.add_input(p_names[State.queued], t_names[module.Action.enter], module.Variable('pat'))
    #             tuple_parts = [module.Expression('pat')]
    #             variables = [module.Variable('pat')]
    #             counter = 0
    #             for res, task_res in task_resource_places.iteritems():
    #                 counter += 1
    #                 res_name = 'res{}'.format(counter)
    #                 self.net.add_input(self.get_resource_place_name(res), t_names[module.Action.enter],
    #                                    module.Variable(res_name))
    #                 tuple_parts.append(module.Expression(res_name))
    #                 variables.append(module.Variable(res_name))
    #             self.net.add_output(p_names[State.service], t_names[module.Action.enter], module.Tuple(tuple_parts))
    #             # exit:
    #             self.net.add_transition(module.Transition(t_names[module.Action.exit], guard=module.Expression("True"),
    #                                                       dist=module.Distribution("immediate")))
    #             self.net.add_input(p_names[State.service], t_names[module.Action.exit], module.Tuple(variables))
    #             self.net.add_output(p_names[State.finished], t_names[module.Action.exit], module.Expression('pat'))
    #             counter = 0
    #             for res, task_res in task_resource_places.iteritems():
    #                 counter += 1
    #                 res_name = 'res{}'.format(counter)
    #                 self.net.add_output(self.get_resource_place_name(res), t_names[module.Action.exit],
    #                                     module.Variable(res_name))
    #             self.task_queues[key] = {State.queued: p_queue,
    #                                      State.service: p_service,
    #                                      State.finished: p_finish,
    #                                      module.Action.enter: t_names[module.Action.enter],
    #                                      module.Action.exit: t_names[module.Action.exit]}
    #         else:  # task is already added to the net.
    #             # TODO: make sensitive to multiple resource configurations
    #             pass
    #         p_queue = self.task_queues[key][State.queued]
    #         p_finish = self.task_queues[key][State.finished]
    #         return tuple([p_queue, p_finish])
    #
    # def normalize_resources(log):
    #     """Replaces duplicate log events that use different resources with one entry that uses the set of resources
    #     assumption: resources are stored in the "resource" key
    #     """
    #     # go through all cases
    #     for case in log:
    #         last_event = None
    #         for event in case:
    #             if 'resource' in event:
    #                 event['resources'] = [str(event['resource'])]
    #                 del event['resource']
    #                 if last_event and last_event['name'] == event['name']:
    #                     # TODO: make sure that also times match!
    #                     if event['resources']:
    #                         event['resources'] = last_event['resources'] + event['resources']
    #                         last_event['delete'] = True
    #                 last_event = event
    #         case[:] = [evt for evt in case if 'delete' not in evt]
    #     return log

    return PetriNet, State, Action, Unfolder, Folder, Enricher, normalize_resources
