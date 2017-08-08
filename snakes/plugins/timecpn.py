"""A plugin that enriches tokens with timing information."""

import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import re
from enum import Enum
from json_logic import jsonLogic
import json
from scipy import stats
from sortedcontainers import SortedSet
from statistics import mean

import snakes.plugins
from snakes import DomainError
from snakes.data import cross, Substitution
from nets import Evaluator


def check_resources_availability(resource_tokens, global_time, day_of_week, times_to_check, probability_based=False):
    """
    Each resource has a field .resource with a dictionary of days
    @param resource_tokens:
    @return: pair of (boolean if resources are available, the array of times to check in the simulation)
    """
    all_available = True
    for resource_token in resource_tokens:
        tok = resource_token
        if isinstance(resource_token, tuple):
            tok = resource_token[0]
        if hasattr(tok, 'resource'):
            max_activations = tok.resource[max(tok.resource, key=tok.resource.get)]
            skip_day = day_of_week not in tok.resource
            if probability_based:
                if hasattr(tok, 'lastDay'):
                    if tok.lastDay == day_of_week:
                        # do not throw dice again for this day
                        skip_day = tok.skipDay
                    else:
                        skip_day = not random.random() < tok.resource[day_of_week] / float(max_activations)
                else:
                    skip_day = not random.random() < tok.resource[day_of_week] / float(max_activations)
                tok.lastDay = day_of_week
                tok.skipDay = skip_day
                if skip_day:
                    all_available = False
                    # make sure that resource's time is advanced one day:
                    time = global_time + (24 * 60 * 60)
                    tok.set_time(time)
                    times_to_check.add(time)

    return all_available, times_to_check

def normalize_weekday(day_string):
    if day_string == "Thurs":
        return "Thu"
    if day_string == "Tues":
        return "Tue"
    return day_string

@snakes.plugins.plugin("snakes.nets",
                       depends=["snakes.plugins.gv"])
def extend(module):
    """Extends `module`"""

    class Action(Enum):
        enter = 1
        exit = 2

    class EnsureFinish(Enum):
        NONE = 1
        NO_UPSTREAM_CASES = 2
        LONG_WAITING_TIME = 3

    class Token(module.Token):
        """Extension of the class `Token` in `module`"""

        def __init__(self, value, **kwargs):
            """Add new time parameter `time`

            >>> print Token(1).get_time()
            0
            >>> print Token(1, time=1000).get_time()
            1000

            @param kwargs: plugin options
            @keyword time: the time of the token.
            @type hello: `float`
            """
            self._time = kwargs.pop("time", 0)
            self._data = kwargs.pop("data", {'priority': 0})
            self._schedule = kwargs.pop("schedule", [])
            self.start_end = kwargs.pop("start_end", [0, 0])
            self._type = kwargs.pop("type", TokenType.resource)

            # self._history = []  # the firing history
            self._snapshot_estimate = 0
            self._queueing_estimate = 0
            module.Token.__init__(self, value)

        def get_time(self):
            return self._time

        def get_start_end(self):
            return self.start_end

        def get_data(self):
            return self._data

        def set_time(self, time):
            self._time = time

        def add_time(self, time):
            self._time += time

        # def trace_firing(self, firing):
        #     """
        #     Records a specific firing in the token's history
        #     expects something like:
        #     >>> token = Token(42)
        #     >>> token.trace_firing(FiringEvent('A', 13588182, 500})
        #     :param firing: a FiringEvent containing information of the firing
        #     """
        #     self._history.append(firing)

        # def export_duration(self, model_name="Default"):
        #     """
        #     returns a string representing the total duration in the model:
        #     time of last transition, which is the exit - time of first transition
        #
        #     :param model_name:
        #     :return: duration in seconds
        #     """
        #     duration = 0 if len(self._history) < 2 else self._history[-1].get_time() - self._history[0].get_time()
        #     return "{};{};{}".format(model_name, self.value, duration)
        #
        # def export_start_end(self, model_name="Default", **kwargs):
        #     """
        #     returns a string representing the total duration in the model:
        #     time of last transition, which is the exit - time of first transition
        #
        #     :param model_name:
        #     :return: duration in seconds
        #     """
        #     # start_end = [0, 0] if len(self._history) < 2 else [self._history[0].get_time(),
        #     #                                                    self._history[-1].get_time()]
        #     # vals = [model_name, self.value, str(start_end[0]), str(start_end[1])] + kwargs.values()
        #     pred_start_end = [0, 0] if len(self._history) < 2 else [self._history[0].get_time(),
        #                                                             self._history[-1].get_time()]
        #     vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(pred_start_end[0]),
        #             str(pred_start_end[1])] + kwargs.values()
        #     return ";".join(map(str, vals))



        def export_snapshot_predictor(self, model_name="Default", **kwargs):
            start = 0 if len(self._history) < 1 else self._history[0].get_time()
            end = float('nan') if self._snapshot_estimate < 0 else start + self._snapshot_estimate
            vals = [model_name, self.value, str(start), str(end)] + kwargs.values()
            return ";".join(map(str, vals))

        def export_queueing_predictor(self, model_name="Default", **kwargs):
            start = 0 if len(self._history) < 1 else self._history[0].get_time()
            end = float('nan') if self._queueing_estimate < 0 else start + self._queueing_estimate
            vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(start),
                    str(end)] + kwargs.values()
            return ";".join(map(str, vals))

        def export_mixed_predictor(self, model_name="Default", **kwargs):
            start = 0 if len(self._history) < 1 else self._history[0].get_time()
            end = float('nan') if self.get_mixed_estimate() < 0 else start + self.get_mixed_estimate()
            vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(start),
                    str(end)] + kwargs.values()
            return ";".join(map(str, vals))

        def export_history(self):
            return "\n".join(str(hist) for hist in self._history)

        def get_last_sojourn_time(self):
            if len(self._history) < 3:
                raise Exception(
                    "token {} doesn't have enough firing history to compute sojourn time!".format(str(self)))
            return self._history[-1].get_time() - self._history[-3].get_time()

        def get_snapshot_estimate(self):
            return self._snapshot_estimate

        def set_snapshot_estimate(self, estimate):
            self._snapshot_estimate = estimate

        def get_queueing_estimate(self):
            return self._queueing_estimate

        def set_queueing_estimate(self, estimate):
            self._queueing_estimate = estimate

        def get_mixed_estimate(self):
            if self.get_snapshot_estimate() < 0:
                return self.get_queueing_estimate()
            else:
                return self._snapshot_estimate

        def __str__(self):
            """Simple string representation (that of the value)

            >>> str(Token(42))
            '42'

            @return: simple string representation
            @rtype: `str`
            """
            return str("{}@{}".format(self.value, self._time))

        def __repr__(self):
            return self.__str__()

        def __eq__(self, other):
            """Tests Tokens for equality.

            >>> Token('joe', time=2) == 'joe'
            True

            @param other: the value to compare against
            @type other: `any`
            @rtype: `Boolean`
            """
            return self.value == other

        def __cmp__(self, other):
            """Compares two tokens by their time value
            :param other:
            :return:
            """
            if hasattr(other, '_time'):
                if self._time == other.get_time():
                    if self.value != other.value:
                        print("stop for a sec.")
                    return cmp(self.value, other.value)
                return int(self._time - other.get_time())
            else:
                return -1

    class Place(module.Place):

        def __init__(self, name, tokens=[], check=None):
            self._last_sojourn_time = -1
            module.Place.__init__(self, name, tokens, check)

        def get_tokens(self, global_time):
            tokens = module.MultiSet([])
            for tok in iter(self.tokens):
                if isinstance(tok, tuple):
                    if tok[0].get_time() <= global_time:
                        tokens.add([tok], 1)
                elif tok.get_time() <= global_time:
                    tokens.add(tok, 1)
            return tokens

        def get_name(self):
            return self.name

        def get_last_sojourn_time(self):
            if self._last_sojourn_time < 0:
                raise Exception("no last sojourn time recorded")
            else:
                return self._last_sojourn_time

        def set_last_sojourn_time(self, sojourn_time):
            self._last_sojourn_time = sojourn_time

        def reset_last_sojourn_time(self):
            self._last_sojourn_time = -1

    class Transition(module.Transition):
        "Extension of the class `Transition` in `module`"

        def __init__(self, name, guard=None, **kwargs):
            """Add new time parameter `weight`
            >>> print Transition(1).get_weight()
            1
            >>> print Transition(1, weight=2).get_weight()
            2

            :param kwargs: plugin options
            :keyword weight: the weight of the transition.
            :keyword dist: the distribution of the transition.
            :type hello: `float`
            :type dist: `Distribution`
            """
            print("transition kwargs {}".format(kwargs))
            self._weight = kwargs.pop("weight", 1)
            self._invisible = kwargs.pop("invisible", False)
            self._dist = kwargs.pop("dist", Distribution("exponential"))
            self._batch_strategy = BatchStrategy(batchlogic=kwargs.pop("batchlogic", True),
                                                 batchdata=kwargs.pop("batchdata", {}))
            self._execution_plan = kwargs.pop("execution_plan", None)
            self._start = False
            module.Transition.__init__(self, name, guard)
            print("created transition {} with weight: {} and dist: {}".format(name, self._weight, self._dist))

        # def __str__(self):
        #     """Simple string representation (that of the value)
        #
        #     >>> str(Transition('t1'))
        #     't1'
        #
        #     @return: simple string representation
        #     @rtype: `str`
        #     """
        #     return str("({} w:{})".format(self.name, self._weight))

        @staticmethod
        def is_first_transition(name):
            return Transition.expression.search(name) is not None

        def set_dist(self, dist):
            """Sets the distribution for the transition.
            :param dist: the distribution to set
            :type dist: `Distribution`
            """

            self._dist = dist

        def get_dist(self):
            return self._dist

        def get_batch_strategy(self):
            return self._batch_strategy

        def set_batch_strategy(self, batch_strategy):
            self._batch_strategy = batch_strategy

        def get_execution_plan(self):
            return self._execution_plan

        def set_execution_plan(self, plan):
            self._execution_plan = plan

        def set_start(self, start):
            self._start = start

        def get_start(self):
            return self._start

        def get_weight(self):
            return self._weight

        def modes(self, **kwargs):
            """Return the list of bindings which enable the transition.
            Note that the modes are usually considered to be the list of
            bindings that _activate_ a transitions. However, this list may
            be infinite so we restricted ourselves to _actual modes_,
            taking into account only the tokens actually present in the
            input places.
            >>> t = Transition('t', Expression('x!=y'))
            >>> px = Place('px', [Token(0,time=1),Token(1,time=10)])
            >>> t.add_input(px, Variable('x'))
            >>> py = Place('py', [Token(0, time=3), Token(1,time=4)])
            >>> t.add_input(py, Variable('y'))
            >>> m = t.modes(time=5)
            >>> len(m)
            1
            >>> Substitution(y=0, x=1) in m
            True
            >>> Substitution(y=1, x=0) in m
            False

            @return: a list of substitutions usable to fire the transition
            @rtype: `list`
            """
            global_time = kwargs.pop('time', 0)
            # print "global time at binding: {}".format(global_time)
            parts = []
            try:
                for place, label in self.input():
                    m = label.modes(place.get_tokens(global_time))
                    # if len(m) > 5:
                    #     m = m[:5] # This hack is valid only, if there is no policy like first come first serve
                    parts.append(m)
            except module.ModeError:
                return []
            result = []
            for x in cross(parts):
                try:
                    if len(x) == 0:
                        sub = Substitution()
                    else:
                        sub = reduce(Substitution.__add__, x)
                    if self._check(sub, False, False):
                        result.append(sub)
                except DomainError:
                    pass
            return result

        def fire(self, binding, **kwargs):
            """Fires a transition and returns the time of availability of the token
            for further progress in the model.
            :return: time
            :rtype: numeric (double)
            """
            if self.enabled(binding):
                duration = int(self._dist.sample())
                global_time = kwargs.get('time')
                snapshot_principle = kwargs.get('snapshot', False)
                if self._dist._name == DistributionType.scheduled:
                    time = max(global_time, duration)
                else:
                    time = global_time + duration
                for place, label in self.input():
                    place.remove(label.flow(binding))
                for place, label in self.output():
                    tokens = label.flow(binding)
                    for tok in tokens:
                        if isinstance(tok, tuple):
                            for t in tok:
                                t.set_time(time)
                                # t.trace_firing(FiringEvent(self.name, global_time, duration))
                        else:
                            tok.set_time(time)
                            # tok.trace_firing(FiringEvent(self.name, global_time, duration))
                    place.add(tokens)
                    if snapshot_principle and place.get_name().endswith("_finished"):
                        # store sojourn time:
                        if len(tokens) > 1:
                            print("debug me!")
                        sojourn_time = iter(tokens).next().get_last_sojourn_time()
                        place.set_last_sojourn_time(sojourn_time)
                return time
            else:
                raise ValueError("transition not enabled for %s" % binding)

        def is_invisible(self):
            return self._invisible

        def set_invisible(self, invisible):
            self._invisible = invisible

    class FiringEvent(object):
        """
        Information collected at tokens
        TODO: We need to merge information on joining parallel tokens!! (if we need more detailed analysis)
        """

        def __init__(self, transition, firing_time, duration):
            self._transition = transition
            self._firing_time = firing_time
            self._duration = duration

        def __str__(self):
            return "Firing; {}; {}; {}".format(self._transition, self._firing_time, self._duration)

        @staticmethod
        def get_header():
            return "Type; Transition; Time; Duration"

        def get_time(self):
            return self._firing_time

        def get_activity(self):
            return self._transition

    class FiringHistory(object):
        """
        An aggregated history of firing events that were performed during simulation.
        Also contains information about the token like when the real case started and ended.
        """
        def __init__(self, token):
            self.history = []
            self.start_end = token.get_start_end()
            self.value = token.value

        def append(self, firingEvent):
            self.history.append(firingEvent)

        def export_start_end(self, model_name="Default", **kwargs):
            """
            returns a string representing the total duration in the model:
            time of last transition, which is the exit - time of first transition

            :param model_name:
            :return: duration in seconds
            """
            pred_start_end = [0, 0] if len(self.history) < 2 else [self.history[0].get_time(),
                                                                    self.history[-1].get_time()]
            vals = [model_name, self.value, str(self.start_end[0]), str(self.start_end[1]), str(pred_start_end[0]),
                    str(pred_start_end[1])] + kwargs.values()
            return ";".join(map(str, vals))

        def get_case_start_time(self):
            return -1 if len(self.history) < 1 else self.history[0].get_time()

        def __iter__(self):
            return iter(self.history)

    class QueueingPolicy(Enum):
        random = 1
        earliest_due_date = 2

    class BatchStrategy(object):
        timevars = ['hour_of_day', 'part_of_day', 'day_of_week', 'longest_waiting_time', 'mean_waiting_time',
                    'maximum_flow_time', 'time_since_last_arrival']
        times_of_day = {'morning': 8, 'noon': 11, 'afternoon': 13, 'evening': 16, 'night': 19}
        days_of_week = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

        def __init__(self, **kwargs):
            self.data = kwargs.get('batchdata', {})
            # self.params['value'] = kwargs.get('value', 'var%value')
            # self.params['value1'] = kwargs.get('value1', 'var%value1')
            # self.params['value2'] = kwargs.get('value2', 'var%value2')
            # self.params['value2'] = kwargs.get('value2', 'var%value3')

            self.logic = kwargs.get('batchlogic', True)

            if self.logic == True:
                self._has_timeout = False
            else:
                varnames = extractVarNamesFromLogic(self.logic)
                if len([name for name in varnames if name not in BatchStrategy.timevars]) > 0:
                    print("check varnames! {}".format([name for name in varnames if name not in BatchStrategy.timevars]))
                self._has_timeout = len(set(BatchStrategy.timevars) & set(varnames)) > 0

        def has_timeout(self):
            return self._has_timeout
            # return 'max_wait' in self.params

        def get_activate(self, context={}):
            context.update(self.data)
            return jsonLogic(tests=self.logic, data=context)
            # activate = False
            # if self.strategy == BatchStrategy.always_start:
            #     activate = True
            # elif self.strategy == BatchStrategy.wait_till_n_cases:
            #     numberwaiting = kwargs.get('numberwaiting', 1)
            #     activate = numberwaiting >= self.params['n']
            # return activate

        def get_next_time_to_check(self, trans, time_waiting, mean_waiting_time, maximum_flow_time, time_since_last_arrival,
                                   simulation_time):
            varnames = extractVarNamesFromLogic(self.logic)
            nextTimes = []
            simulation_datetime = datetime.utcfromtimestamp(simulation_time)
            for varname in varnames:
                if varname == 'hour_of_day':
                    currentHour = simulation_datetime.hour
                    targetHour = self.data.get('hour_of_day_value', currentHour+1)

                    if targetHour < currentHour:
                        targetHour += 24
                    overtime = simulation_datetime.minute * 60 + simulation_datetime.second
                    nextTimes += [timedelta(hours=targetHour - currentHour).total_seconds() - overtime]

                elif varname == 'part_of_day':
                    currentHour = simulation_datetime.hour
                    currentPartOfDay = TimeSimulator.PART_OF_DAY_LOOKUP[currentHour]
                    targetTimeOfDay = self.data.get('part_of_day_value', TimeSimulator.NEXT_PART_OF_DAY[currentPartOfDay])
                    targetHour = BatchStrategy.times_of_day[targetTimeOfDay]
                    if targetHour < currentHour:
                        targetHour += 24
                    overtime = simulation_datetime.minute * 60 + simulation_datetime.second
                    nextTimes += [timedelta(hours=targetHour - currentHour).total_seconds() - overtime]
                elif varname == 'day_of_week':
                    currentDay = simulation_datetime.weekday()
                    if 'day_of_week_value' in self.data:
                        targetDay = BatchStrategy.days_of_week[self.data.get('day_of_week_value', 'Mon')]
                    else:
                        targetDay = currentDay+1
                    if targetDay < currentDay:
                        targetDay += 7
                    overtime = simulation_datetime.hour * 3600 + simulation_datetime.minute * 60 + simulation_datetime.second
                    nextTimes += [timedelta(days=targetDay - currentDay).total_seconds() - overtime]
                elif varname == 'longest_waiting_time':
                    # hack: if last or rule is "longest_waiting_time > XYZ
                    if 'or' in self.logic and ">=" in self.logic['or'][-1] and type(self.logic['or'][-1][">="][1]) is float:
                        nextTimes += [self.logic['or'][-1][">="][1] * 60 - time_waiting + 1]
                    nextTimes += [self.data.get('longest_waiting_time_value', time_waiting/60 + 30) * 60 - time_waiting]  # add 5 minutes
                elif varname == 'mean_waiting_time':
                    nextTimes += [self.data.get('mean_waiting_time_value', mean_waiting_time/60 + 30)*60 - mean_waiting_time] # add 5 minutes
                elif varname == 'maximum_flow_time':
                    nextTimes += [self.data.get('maximum_flow_time_value', maximum_flow_time/60 + 30)*60 - maximum_flow_time] # add 5 minutes
                elif varname == 'time_since_last_arrival':
                    nextTimes += [self.data.get('time_since_last_arrival_value',
                                                time_since_last_arrival/60 + 30)*60 - time_since_last_arrival] # add 5 minutes
            next_valid_times = [nexttime for nexttime in nextTimes if nexttime >= 0.5]
            # print list(varnames)
            if len(next_valid_times) == 0:
                return None  # make sure to increment the time at least by one unit, otherwise there is no point in addint a new time.

            return min(next_valid_times)
            # return self.params['max_wait'] - time + 1

        def is_always_starting(self):
            return self.logic == True

        def __str__(self):
            return "data: {}\n" \
                   "logic: {}".format(repr(self.data), repr(self.logic))

    def extractVarNamesFromLogic(tests):
        # You've recursed to a primitive, stop!
        if tests in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            return None
        if tests is None or (type(tests) not in [dict, float]):
            return tests
        if type(tests) in [float]:
            return None

        op = list(tests.keys())[0]
        values = tests[op]
        operations = {
            "==": (lambda a, b: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            "===": (lambda a, b: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            "!=": (lambda a, b: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            "!==": (lambda a, b: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            ">": (lambda a, b: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            ">=": (lambda a, b: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            "<": (lambda a, b, c=None: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            "<=": (lambda a, b, c=None: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            "!": (lambda a: (a if type(a) == list else [a])),
            "%": (lambda a, b: (a if type(a) == list else [a]) + (b if type(b) == list else [b])),
            "in": (lambda a,b: (a if type(a) == list else [a])),
            "and": (lambda *args:
                    reduce(lambda total, arg: total + arg if type(arg) == list else [arg], args, [])
                    ),
            "or": (lambda *args:
                   reduce(lambda total, arg: total + arg if type(arg) == list else [arg], args, [])
                   ),
            "var": (lambda a, not_found=None:
                    str(a).split(".")[-1]
                    )
        }

        if op not in operations:
            raise ValueError("Unrecognized operation %s" % op)

        # Easy syntax for unary operators, like {"var": "x"} instead of strict
        # {"var": ["x"]}
        if type(values) not in [list, tuple]:
            values = [values]

        # Recursion!
        values = map(lambda val: extractVarNamesFromLogic(val), values)

        return [x for x in list(np.array(operations[op](*values)).flat) if x is not None]

    def instantiateRandomDict(logic):
        # You've recursed to a primitive, stop!
        if logic is None or type(logic) != dict:
            return logic

        result = {}
        op = list(logic.keys())[0]
        values = logic[op]
        operations = {
            "<=": (lambda a, b, c:
                   {extractVarNamesFromLogic(b)[0]: random.choice(range(a, c))}
                   ),
            "in": (lambda a, b:
                   {extractVarNamesFromLogic(a)[0]: random.choice(b)}
                   ),
        }

        if op not in operations:
            raise ValueError("Unrecognized operation %s" % op)

        # Easy syntax for unary operators, like {"var": "x"} instead of strict
        # {"var": ["x"]}
        if type(values) not in [list, tuple]:
            values = [values]

        # Recursion!
        # values = map(lambda val: jsonLogic(val, data), values)

        return operations[op](*values)

    class ExecutionPlan(object):
        def __init__(self, modes):
            self.modes = modes

        def get_plan(self):
            return self.modes

        def set_plan(self, modes):
            self.modes = modes

    class BatchAnnotator(object):
        def __init__(self, rulesfile, ensure_finish=EnsureFinish.NONE):
            self.ensure_finish = ensure_finish
            self.rules = self.get_rules(rulesfile)


        def get_rules(self, rulesfile):
            rules = {}
            prog = re.compile("Rule [0-9]+: \$\$\$([^\$]*)\$\$\$")


            with open(rulesfile) as f:
                content = f.readlines()
                for line in content:
                    line = line.strip()
                    if len(line) > 1:
                        line = self.clean_rule_names(line)
                        rulesAll = prog.findall(line)
                        combined_rule = {"or":[]}
                        for rule in rulesAll:
                            ruleparts = rule.split("\t")
                            innerRule = {"and":[]}
                            for rulepart in ruleparts:
                                if len(rulepart) > 0:
                                    innerRule["and"].append(self.get_rule(rulepart))
                            combined_rule["or"].append(innerRule)
                        if self.ensure_finish == EnsureFinish.NO_UPSTREAM_CASES:
                            combined_rule["or"].append({"<=": [{"var": "n_upstream_cases"}, 0.0]})
                        elif self.ensure_finish == EnsureFinish.LONG_WAITING_TIME:
                            combined_rule["or"].append({">=": [{"var": "longest_waiting_time"}, 1]})
                        parts = line.split("\t")
                        name = parts[0]
                        name = name.replace("\"", "")
                        nameparts = name.split(" / ")
                        if len(nameparts) == 1:
                            nameparts = name.split("/")
                        name = "t"+nameparts[1] +"."+nameparts[0]+"_enter"
                        rules[name] = BatchStrategy(batchlogic=combined_rule, batchdata={})
            return rules

        def clean_rule_names(self, line):
            line = re.sub('wt_longest_queueing_case','longest_waiting_time', line)
            line = re.sub('mean_wt_queueing_cases', 'mean_waiting_time', line)
            line = re.sub('res_workload', 'workload', line)
            line = re.sub('Thurs', 'Thu', line)
            line = re.sub('Tues', 'Tue', line)
            return line

        def annotate_net(self, net):
            annotated = []
            not_annotated = list(self.rules.keys())
            for trans in net.transition():
                name = trans.name
                if name in self.rules:
                    annotated.append(name)
                    not_annotated.remove(name)
                    rule = self.rules[name]
                    if self.ensure_finish == EnsureFinish.LONG_WAITING_TIME:  #  and not rule.has_timeout():
                        threshold_waiting_time = (mean(trans.waitingTimes) + 2 * online_variance(trans.waitingTimes)**0.5) / 60
                        if threshold_waiting_time != threshold_waiting_time: # nan
                             threshold_waiting_time = mean(trans.waitingTimes)
                        rule.logic["or"][-1][">="][-1] = threshold_waiting_time

                    trans.set_batch_strategy(rule)
            print("-- Annotated {} transitions with batch rules! --".format(len(annotated)))
            print("-- not annotated transitions: {} --".format(str(not_annotated)))
            return net

        def get_rule(self, rulepart):
            ruletokens = rulepart.split(" ")
            # assume infix notation:
            lhs = ruletokens[0]
            operator = ruletokens[1]
            if operator == "in":
                rhs = json.loads(rulepart.split(" in ")[1])
            else:
                if operator == "=":
                    operator = "=="
                rhs = ruletokens[2]
                if rhs in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
                    pass
                else:
                    rhs = float(rhs)
            return {operator: [{"var":lhs}, rhs]}

    class DistributionType(Enum):
        normal = 1
        exponential = 2
        uniform = 3
        scheduled = 4
        immediate = 5
        kernel_density_estimate = 6
        triangular = 7
        deterministic = 8
        empirical = 9

    class TokenType(Enum):
        instance = 0
        resource = 1

    class Distribution:
        """
        a class to capture supported distributions that can be fit to data
        """

        def __init__(self, dist_type=DistributionType.exponential, **kwargs):
            self._dist = None
            fit = kwargs.get('fit', False)
            if not fit:
                if dist_type == DistributionType.normal:
                    self._param = [kwargs.get('mean', 5), kwargs.get('sd', 1)]
                elif dist_type == DistributionType.exponential:
                    self._param = [kwargs.get('rate', 1)]
                elif dist_type == DistributionType.uniform:
                    self._param = [kwargs.get('low', 0), kwargs.get('high', 1)]
                elif dist_type == DistributionType.scheduled:
                    self._param = [kwargs.get('mintime', 0)]
                elif dist_type == DistributionType.triangular:
                    self._param = [kwargs.get('min', 0), kwargs.get('peak', 3), kwargs.get('max', 5)]
                else:  # immediate transition
                    self._param = [0]
                    dist_type = DistributionType.immediate
            else:  # fit distributions to data
                # Assuming that the values are stored in a 1-d vector called "values"
                self._param = kwargs.get('values', [0, 0, 1])
                if dist_type == DistributionType.empirical:
                    pass
                else:
                    self._dist = stats.gaussian_kde(self._param)
                    dist_type = DistributionType.kernel_density_estimate

            self._name = dist_type
            print("creating {} distribution with {} values".format(dist_type, self._param))

        def __repr__(self):
            return "{} distribution (params:{})".format(self._name, self._param)

        def sample(self, **kwargs):
            """
            Samples a random value from the distribution that is used.
            :return: a sample from the distribution
            """
            if TimeSimulator.DEBUG:
                print "dist name: {}".format(self._name)
            if self._name == DistributionType.normal:
                return random.gauss(self._param[0], self._param[1])
            elif self._name == DistributionType.exponential:
                return random.expovariate(self._param[0])
            elif self._name == DistributionType.immediate:
                return 0
            elif self._name == DistributionType.scheduled:
                current_time = kwargs.get('time', 0)
                return max(self._param[0], current_time) - current_time
            elif self._name == DistributionType.kernel_density_estimate:
                return abs(self._dist.resample(size=1)[0][0])
            elif self._name == DistributionType.triangular:
                return np.random.triangular(self._param[0], self._param[1], self._param[2])
            elif self._name == DistributionType.deterministic:
                return self._param[0]
            elif self._name == DistributionType.empirical:
                # draw one of the past samples uniformly
                return self._param[random.randint(0, len(self._param) - 1)]
            else:  # assume uniform distribution
                return random.uniform(self._param[0], self._param[1])

        def get_mean(self):
            if self._name == DistributionType.kernel_density_estimate or self._name == DistributionType.empirical:
                return np.mean(self._param)
            elif self._name == DistributionType.exponential:
                return 1/self._param[0]
            elif self._name == DistributionType.deterministic:
                return self._param[0]
            elif self._name == DistributionType.immediate:
                return 0
            else:
                print("please implement for the Mean {} - debug me!".format(self._name))
                return -1

        def get_CV(self):
            if self._name == DistributionType.kernel_density_estimate or self._name == DistributionType.empirical:
                if np.mean(self._param)==0:
                    print("zero mean for the CV calculation - debug me!")
                else:
                    return np.std(self._param)/np.mean(self._param)
            elif self._name == DistributionType.exponential:
                 return 1
            elif self._name == DistributionType.deterministic:
                return 0
            elif self._name == DistributionType.immediate:
                return 0
            else:
                print("please implement for the CV {} - debug me!".format(self._name))
                return -1

    class TimeSimulator(object):
        MAX_EVENTS_PER_RUN = 100000000
        MAX_ITERATIONS_WITHOUT_PROGRESS = 10000
        MAX_ITERATIONS_WITHOUT_PROGRESS_PER_TASK = 4000  # (half-hour checks result in 24 * 2 * 7 = 336 checks for an entire week)
        PART_OF_DAY_LOOKUP = {8: 'morning', 9: 'morning', 10: 'morning',
                              11: 'noon', 12: 'noon',
                              13: 'afternoon', 14: 'afternoon', 15: 'afternoon',
                              16: 'evening', 17: 'evening', 18: 'evening',
                              19: 'night', 20: 'night', 21: 'night', 22: 'night', 23: 'night', 0: 'night', 1: 'night',
                              2: 'night', 3: 'night', 4: 'night', 5: 'night', 6: 'night', 7: 'night'}
        NEXT_PART_OF_DAY = {'morning': 'noon', 'noon': 'afternoon', 'afternoon': 'evening', 'evening': 'night', 'night':'morning'}

        DEBUG = False

        def __init__(self, net):
            self.net = net
            self.initial = net.get_marking()
            self.resource_queues = self.extract_resource_queues(net)
            # self.case_queues = self.extract_resource_queues(net,)
            self.upstream_places = self.extract_upstream_places(net)
            self.current = self.initial
            self.current_time = 0
            self.snapshot_principle = False
            self.task_snapshot = {}
            self.histories = {}
            self._cases = {}
            self._produce_log = False

        def set_snapshot(self, snapshot):
            self.snapshot_principle = snapshot

        def set_produce_log(self, produce_log):
            self._produce_log = produce_log

        def add_log_entry(self, activity, instance, resources, firing_time, time, data):
            lifecycle = 'complete'
            if activity.endswith(Action.enter.name):
                activity = activity[:activity.index(Action.enter.name) - 1]
                lifecycle = "start"
            elif activity.endswith(Action.exit.name):
                activity = activity[:activity.index(Action.exit.name) - 1]
            elif activity.startswith('tarrival_'):
                activity = 'arrival'

            event = {'name': activity, 'timestamp': firing_time, 'resources': resources, 'lifecycle': lifecycle,
                     'priority': 0 if 'priority' not in data else data['priority']}
            if 'activity_count' in data:
                event['activity_count'] = data['activity_count']

            # end_event = {'name': activity, 'timestamp': time, 'resources': resources, 'lifecycle': 'complete'}
            if instance not in self._cases:
                self._cases[instance] = [event]
            else:
                self._cases[instance].append(event)
                # self._cases[instance].append(end_event)

        def get_log(self):
            flattened_cases = []
            for a_case in self._cases:
                flattened_cases.append(self._cases[a_case])
            return flattened_cases

        def pick_mode(self, modes, earliest=False):
            mode = None
            earliest_time = -1
            for m in modes:
                tokens = m.image()
                token_time = -1
                for token in tokens:
                    if is_instance(token):
                        token_time = token.get_time() if token_time == -1 else token.get_time()
                if earliest:
                    is_better = earliest_time == -1 or token_time < earliest_time
                else:
                    is_better = earliest_time == -1 or token_time > earliest_time
                if is_better:
                    earliest_time = token_time
                    mode = m
            return mode

        def extract_resource_queues(self, net):
            resource_queues = {}
            marking = net.get_marking()
            for place, tokens in marking.iteritems():
                is_resource = len(tokens) > 0 and not is_instance(tokens.iteritems().next()[0])
                if is_resource:
                    transitions = net.post(place)
                    predecessors = sum([list(net.pre(transition)) for transition in transitions], [])
                    inplaces = [pl for pl in predecessors if pl != place]
                    resource_queues[place] = inplaces
            return resource_queues

        def extract_upstream_places(self, net):
            transitions = net.transition()

            source_place = self.find_source(net)

            visited = {source_place.name: [source_place]}
            # path = {}

            nodes = set(net.node())

            while nodes:
                min_node = None
                for node in nodes:
                    if node.name in visited:
                        if min_node is None:
                            min_node = node
                        elif len(visited[node.name]) < len(visited[min_node.name]):
                            min_node = node

                if min_node is None:
                    break

                nodes.remove(min_node)
                currentpath = visited[min_node.name]

                for nextneighbor in net.post(min_node.name):
                    nextneighbornode = net.node(nextneighbor)
                    newpath = currentpath + [nextneighbornode]
                    if nextneighbor not in visited or len(newpath) < len(visited[nextneighbor]):
                        visited[nextneighbor] = newpath
                        # path[nextneighbor] = min_node
            upstream_places = defaultdict(list)
            for transition in transitions:
                if transition.name in visited:
                    for node in visited[transition.name]:
                        if str(type(node)) == '<class \'snakes.plugins.timecpn.Place\'>' and not is_resource_place(node):
                            upstream_places[transition].append(node)

            return upstream_places
            # #  , path
            #
            # for transition in transitions:
            #     pre_places = self.get_recursive_places(net, transition, [transition.name])
            #     upstream_places_without_queue = [place for place in pre_places if place not in net.pre(transition.name)]
            #     upstream_places[transition] = upstream_places_without_queue
            # return upstream_places

        def get_recursive_places(self, net, transition, transitions_visited):
            pre_places = [place for place in net.pre(str(transition)) if not is_resource_place(place)]
            places = set(pre_places)
            transitions = set()
            for place in pre_places:
                transitions |= set([transition for transition in net.pre(place) if transition not in transitions_visited])
            transitions_visited += transitions
            for trans in transitions:
                places |= self.get_recursive_places(net, trans, transitions_visited)
            return places


        def get_upstream_cases(self, trans, marking):
            case_count = 0
            for place in self.upstream_places[trans]:
                case_count += 0 if place.name not in marking else len(marking[place.name])
            return case_count

        def firings_count(self, token, label):
            case_history = self.histories[token.value] if token.value in self.histories else []
            count = 0
            for firing in case_history:
                if str(firing.get_activity()) == label:
                    count += 1
            return count

        def firings(self, pat, label, num):
            return self.firings_count(pat, label) == num

        def simulate_one(self, marking=None, global_time=0, queuing_policy=QueueingPolicy.earliest_due_date, rulename=None, experiment=False):
            """
            Simulates a schedule-driven queueing-enabled net until all scheduled tasks are done
            (we mostly use it to simulate one day at once)

            :param marking: initial marking
            :param global_time: the global time at start (a low value should be fine)
            :param queuing_policy: the queueing policy (random or EDD)
            :@param experiment: in experiment mode, the number of activity executions is drawn randomly for each case to avoid deadlocks
            :rtype : Marking
            :return: the marking after no more transition is available
            """
            self.net.globals.update(Evaluator(firings=self.firings))
            for trans in self.net.transition():
                trans.guard.globals.update(Evaluator(firings=self.firings))

            if marking is None:
                marking = self.initial
            self.current = marking.copy()
            self.current_time = global_time

            self.histories = {}

            times_to_check = SortedSet()

            times_to_check.add(global_time)

            for pName, multi_set in self.current.iteritems():
                for tok in iter(multi_set):
                    times_to_check.add(tok.get_time())
            if TimeSimulator.DEBUG:
                print "times to check: {}".format(times_to_check)

            self.net.set_marking(self.current)

            activeTransitions = set(self.net.transition())
            inactivatedTransitions = set([])
            batchTransitions = set([])

            ZERO = datetime.utcfromtimestamp(0)
            last_workload = 0
            num = 0
            number_of_resource_unavailabilities = 0
            iterations_without_change = 0
            while num < TimeSimulator.MAX_EVENTS_PER_RUN and iterations_without_change < TimeSimulator.MAX_ITERATIONS_WITHOUT_PROGRESS:
                if num % 100 == 0:
                    print("done {} iterations. Times to check: {}".format(num, len(times_to_check)))
                num += 1
                iterations_without_change += 1
                enabled_transitions = {}
                sum_weight = 0
                currentmarking = self.net.get_marking()
                for trans in batchTransitions:
                    exec_plan = trans.get_execution_plan()
                    if exec_plan is not None:
                        modes = trans.modes(time=global_time)
                        if modes:
                            (resources_available, times_to_check) = check_resources_availability(resource_tokens,
                                                                                                 global_time,
                                                                                                 global_datetime.strftime(
                                                                                                     '%a'),
                                                                                                 times_to_check,
                                                                                                 probability_based=hasattr(
                                                                                                     self.net,
                                                                                                     'probability_based'))
                            if resources_available:
                                enabled_transitions[trans] = modes

                if len(enabled_transitions) == 0:
                    for trans in activeTransitions:
                    # for trans in self.net.transition():
                        preplaces = self.net.pre(trans.name)
                        # self.
                        active = all([p in currentmarking for p in preplaces])
                        if not active:
                            inactivatedTransitions.add(trans)
                        else:
                            modes = trans.modes(time=global_time)
                            exec_plan = trans.get_execution_plan()

                            if modes:
                                resource_tokens = [(modes[0][key], key) for key in modes[0] if
                                                   not is_instance(modes[0][key])]

                                global_datetime = datetime.utcfromtimestamp(global_time)

                                (resources_available, times_to_check) = check_resources_availability(resource_tokens, global_time, global_datetime.strftime('%a'),times_to_check, probability_based=hasattr(self.net,'probability_based'))
                                if resources_available:
                                    if exec_plan is not None:
                                        # check if modes overlap
                                        i, j = get_stored_modes_key(modes, exec_plan)
                                        if i > -1:
                                            mode = modes[i]
                                            enabled_transitions[trans] = [mode]
                                            sum_weight += trans.get_weight()
                                        else:
                                            print("did not find corresponding mode for {} in {}".format(modes, exec_plan.get_plan))
                                    elif modes:
                                        # there can be many modes, but each of them should have a resource
                                        waiting_times = self.get_waiting_case_token_times(modes, global_time)
                                        longest_waiting_time = 0 if len(waiting_times) == 0 else max(waiting_times)
                                        mean_waiting_time = 0 if len(waiting_times) == 0 else mean(waiting_times)
                                        time_since_last_arrival = 0 if len(waiting_times) == 0 else min(waiting_times)
                                        context = {}

                                        context['n_in_queue'] = len(modes)
                                        context['n_running_cases'] = self.get_running_cases(self.net, currentmarking)
                                        context['hour_of_day'] = global_datetime.hour
                                        # if global_datetime.hour == 0:
                                        #     print "hour = 0!"
                                        context['part_of_day'] = TimeSimulator.PART_OF_DAY_LOOKUP[global_datetime.hour]
                                        context['day_of_week'] = global_datetime.strftime('%a')
                                        context['longest_waiting_time'] = longest_waiting_time / 60  # in minutes
                                        context['mean_waiting_time'] = mean_waiting_time / 60  # in minutes
                                        maximum_flow_time = self.get_maximum_flow_time(modes, global_time)
                                        context['maximum_flow_time'] = maximum_flow_time / 60  # in minutes
                                        context['time_since_last_arrival'] = time_since_last_arrival / 60  # in minutes


                                        trace_tokens = sum([[mode[key] for key in mode if is_instance(mode[key])] for mode in modes], [])

                                        res_place = [pla for pla in trans.pre if len(resource_tokens) > 0 and str(trans.pre[pla]) == resource_tokens[0][1]]
                                        self.current = self.net.get_marking()
                                        context['workload'] = 0 if len(res_place) == 0 else sum(
                                            [len(self.current[place]) if place in self.current else 0 for place in
                                             self.resource_queues[res_place[0]]])

                                        # get last case token to arrive at transition:
                                        last_arrived_token_mode = self.pick_mode(modes, earliest=False)
                                        last_arrived_tokens = [last_arrived_token_mode[tok] for tok in last_arrived_token_mode if
                                                              is_instance(last_arrived_token_mode[tok])]
                                        last_arrived_token = last_arrived_tokens[0] if len(last_arrived_tokens) > 0 else None

                                        context['attributes'] = last_arrived_token.get_data() if last_arrived_token else {}

                                        context['aggr_attribute_priority'] = sum([tok.get_data()['priority'] for tok in trace_tokens])
                                        if experiment:
                                            context['activity_execution_counter'] = last_arrived_token.get_data().get('activity_count',0) if last_arrived_token else 0
                                        else:
                                            context['activity_execution_counter'] = self.firings_count(last_arrived_token,str(trans)) if last_arrived_token else 0

                                        context['n_upstream_cases'] = self.get_upstream_cases(trans, self.net.get_marking()) - context['n_in_queue']
                                        # if context['n_upstream_cases'] > 0:
                                        #     print "DEBUG: more cases expected to arrive here."
                                        # check if batch strategy allows to fire this one
                                        strategy = trans.get_batch_strategy()
                                        start = strategy.get_activate(context)
                                        # if not start and strategy.logic is not True:
                                        #     print "not starting"
                                        if not start and strategy.has_timeout():
                                            # time_waiting, mean_waiting_time, maximum_flow_time, time_since_last_arrival, simulation_time
                                            time_waiting = longest_waiting_time
                                            if hasattr(trans, 'not_starting'):
                                                trans.not_starting += 1
                                            else:
                                                trans.not_starting = 1
                                            if trans.not_starting % 200 == 0:
                                                print("trans {} not starting for {} iterations".format(trans, trans.not_starting))
                                            if trans.not_starting < self.MAX_ITERATIONS_WITHOUT_PROGRESS_PER_TASK:
                                                next_time = strategy.get_next_time_to_check(trans, time_waiting, mean_waiting_time,
                                                                                                  maximum_flow_time,
                                                                                                  time_since_last_arrival,
                                                                                                  simulation_time=global_time)
                                                if next_time:
                                                    times_to_check.add(global_time + int(next_time))
                                                # else:
                                                #     print "no next time?"
                                        elif start and not strategy.is_always_starting():
                                            #print "starting batch for {}! waiting tokens: {}, longest time: {} min, mean time: {}, min time: {}, max_flow_time:{}".format(trans, len(modes),
                                            #                                                                         round(longest_waiting_time / 60.), round(mean_waiting_time/60.), round(time_since_last_arrival/60.), maximum_flow_time / 60)
                                            trans.not_starting = 0
                                            # print "starting batch for {}!\ncontext: {}".format(trans, context)
                                            # if last_workload == context['workload']:
                                            #     print "debug me"
                                            # last_workload = context['workload']
                                            pass
                                        trans.set_start(start)
                                        if start:
                                            enabled_transitions[trans] = modes
                                            sum_weight += trans.get_weight()
                                else:
                                    number_of_resource_unavailabilities += 1
                                    trans.set_start(False)
                    activeTransitions = activeTransitions.difference(inactivatedTransitions)
                    inactivatedTransitions.clear()

                if not enabled_transitions:
                    # increment time to next token's time!
                    last_time = global_time
                    if not times_to_check:
                        break
                    global_time = times_to_check.pop(0)
                    if TimeSimulator.DEBUG:
                        print("no transitions enabled at time {} checking model at time {}".format(last_time,
                                                                                                   global_time))
                        # self.net.draw("generated_during_simulation.png")
                else:
                    # pick according to transition weights
                    trans_to_fire = random.uniform(0, sum_weight)
                    if TimeSimulator.DEBUG:
                        print("sampled {} from {}".format(trans_to_fire, sum_weight))
                    cumulative_weight = 0
                    transition = None
                    for enabled, modes in enabled_transitions.iteritems():
                        cumulative_weight += enabled.get_weight()
                        if trans_to_fire < cumulative_weight:
                            transition = enabled
                            break
                    modes = enabled_transitions[transition]
                    mode = None
                    if queuing_policy == QueueingPolicy.random:
                        # pick randomly from modes:
                        mode = random.choice(modes)
                    elif queuing_policy == QueueingPolicy.earliest_due_date:
                        # pick the ones with the lowest time stamps on patients queuing

                        mode = self.pick_mode(modes, earliest=True)
                    exec_plan = transition.get_execution_plan()
                    if exec_plan is not None:
                        stored_modes = exec_plan.get_plan()
                        if TimeSimulator.DEBUG:
                            print("{} items in batch for transition {}.".format(len(stored_modes), transition))
                        stored_mode_key = get_stored_mode_key(mode, exec_plan)
                        if stored_mode_key > -1:
                            stored_modes.pop(stored_mode_key)
                            exec_plan.set_plan(stored_modes)
                        if len(stored_modes) == 0:
                            transition.set_execution_plan(None)  # finished batch!
                            batchTransitions.remove(transition)
                            if TimeSimulator.DEBUG:
                                print("finished batch for transition {}".format(transition))
                    elif transition.get_start() and len(modes) > 1:
                        # fprint "transition {} starts batch at {}".format(transition, TimeSimulator.PART_OF_DAY_LOOKUP[global_datetime.hour])
                        if TimeSimulator.DEBUG:
                            print("transition {} starts batch of size {} with longest wait: {} minutes.".format(
                                transition, len(modes),
                                round(max(self.get_waiting_case_token_times(modes, global_time)) / 60000)))
                        batchTransitions.add(transition)
                        exec_plan = ExecutionPlan(modes)
                        transition.set_execution_plan(exec_plan)
                        stored_mode_key = get_stored_mode_key(mode, exec_plan)
                        if stored_mode_key > -1:
                            stored_modes = exec_plan.get_plan()
                            stored_modes.pop(stored_mode_key)
                            exec_plan.set_plan(stored_modes)

                    firing_time = global_time
                    time = transition.fire(mode, time=global_time, snapshot=self.snapshot_principle)
                    postplaces = self.net.post(transition.name)
                    newlyEnabledTransitions = [self.net.transition(transitionName) for transitionName in self.net.post(postplaces)]
                    activeTransitions = activeTransitions.union(newlyEnabledTransitions)
                    self.update_history(mode, transition, global_time, (time-global_time))
                    if TimeSimulator.DEBUG:
                        print("fired transition: {} at time {} with duration {}".format(transition.name, global_time, (time - global_time)))
                    # self.net.draw("model_during_sim.png")
                    iterations_without_change = 0

                    if self._produce_log:
                        (instance, data) = get_instance(mode)
                        resources = get_resources(mode)
                        if not transition.is_invisible() and instance:
                            self.add_log_entry(str(transition), instance, resources, firing_time, time, data)

                    times_to_check.add(int(time))
                    if rulename:
                        self.net.draw("{}/generated_debug.png".format(rulename))

                    if TimeSimulator.DEBUG:
                        print("times to check: {}".format(times_to_check))

            print "times_to_check: {}".format(times_to_check)
            print "\n\nnumber_of_resource_unavailabilities: {}\n".format(number_of_resource_unavailabilities)
            return self.net.get_marking()

        def init_help(self):
            return {
                "#trace": {
                    "title": "Trace",
                    "content": "the states and transitions explored so far"
                },
                "#model": {
                    "title": "Model",
                    "content": "the model being simulated"
                },
                "#alive .ui #ui-quit": {
                    "title": "Stop",
                    "content": "stop the simulator (server side)"
                },
                "#alive .ui #ui-help": {
                    "title": "Help",
                    "content": "show this help"
                },
                "#alive .ui #ui-about": {
                    "title": "About",
                    "content": "show information about the simulator"
                },
            }

        def update_history(self, mode, trans, global_time, duration):
            # find instance token in mode:
            for key in mode:
                token = mode[key]
                if is_instance(token): # add firing event
                    if token.value not in self.histories:
                        self.histories[token.value] = FiringHistory(token)
                    self.histories[token.value].append(FiringEvent(trans.name, global_time, duration))

        def get_histories(self):
            return self.histories

        def get_maximum_flow_time(self, modes, global_time):
            time = global_time
            for mode in modes:
                stored_tokens = mode.image()
                for token in stored_tokens:
                    if is_instance(token):
                        if token.value in self.histories:
                            case_start_time = self.histories[token.value].get_case_start_time()
                        else:
                            case_start_time = time
                        if case_start_time >= 0 and case_start_time < time:
                            time = case_start_time
            return global_time - time

        def get_waiting_case_token_times(self, modes, global_time):
            times = []
            for mode in modes:
                stored_tokens = mode.image()
                for token in stored_tokens:
                    if is_instance(token):
                        times.append(global_time - token.get_time())
            return times

        def find_source(self, net):
            for place in net.place():
                if len(net.pre(place.name)) == 0 or place.name == 'parrival_service':
                    return place

        def get_running_cases(self, net, marking):
            n_running = 0
            # TODO: check for parallelism
            for place in net.place():
                if place.name != 'arrivals' and place.name != 'exists' and place.name in marking and not is_resource_place(place) :
                    n_running += len(marking[place.name])
            return n_running



    def is_resource_place(place):
        return str(place).startswith('p_res_')

    def online_variance(data):
        n = 0
        mean = 0.0
        M2 = 0.0

        for x in data:
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)

        if n < 2:
            return float('nan')
        else:
            return M2 / (n - 1)

    def is_instance(token):
        """
        Checks whether the token represents a patient/instance
        :param token: the token specifying a resource (patients start with "pat", instances start with "trace")
        :return: boolean
        """
        #return str(token).startswith("pat") or str(token).startswith('trace')
        if isinstance(token, tuple):
            return repr(token[0]._type) == repr(TokenType.instance)
        return repr(token._type)==repr(TokenType.instance)

    def get_stored_modes_key(modes, exec_plan):
        for i, mode in enumerate(modes):
            j = get_stored_mode_key(mode, exec_plan)
            if j > -1:
                return i, j
        return -1, -1

    def get_stored_mode_key(mode, exec_plan):
        tokens = mode.image()
        instance_to_fire = None
        for token in tokens:
            if is_instance(token):
                instance_to_fire = get_name(token)
        stored_mode_key = -1
        for i, stored_mode in enumerate(exec_plan.get_plan()):
            stored_tokens = stored_mode.image()
            for token in stored_tokens:
                if is_instance(token):
                    if instance_to_fire == get_name(token):
                        stored_mode_key = i
        return stored_mode_key

    def get_instance(mode):
        tokens = mode.image()
        for token in tokens:
            if is_instance(token):
                return get_name(token), token.get_data()
        return None, None

    def get_resources(mode):
        tokens = mode.image()
        resources = []
        for token in tokens:
            if not is_instance(token):
                resources.append(get_name(token))
        return resources

    def get_name(token):
        name = str(token)
        if '@' in name:
            index = name.find('@')
            return name[:index]
        return name

    return module.PetriNet, EnsureFinish, Action, Place, Token, TokenType, Transition, FiringEvent, QueueingPolicy, DistributionType, Distribution, \
           TimeSimulator, BatchStrategy, BatchAnnotator, is_instance, extractVarNamesFromLogic, instantiateRandomDict, normalize_weekday
