# coding: utf-8
import json
import random
import string
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime

import pmlab.log

import snakes.plugins

time_cpn = snakes.plugins.load(['timecpn', 'unfolder'], 'snakes.nets', 'time_cpn')
from time_cpn import PetriNet, Place, Transition, State, Action, Token, TokenType, Distribution, DistributionType, Unfolder, \
    Variable, Expression, Tuple, TimeSimulator, QueueingPolicy, BatchStrategy, instantiateRandomDict, extractVarNamesFromLogic

rules = {}

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


def loadRules(mypath):
    rules = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)
    for myfilepath in onlyfiles:
        print "opening file " + str(myfilepath)
        with open(mypath + myfilepath) as f:
            rules = rules + [json_load_byteified(f)]
    return rules


def getRules():
    if len(rules) == 0:
        # load rules:
        path = './'
        rules['plain'] = loadRules(path + 'rules/')
        rules['composite'] = loadRules(path + 'rules/composite/')
    return rules


def getRandomRule(also_composite=False):
    rules = getRules()
    ruleset = rules['plain']
    if also_composite:
        if random.choice([True, False]):
            ruleset = rules['composite']
    return random.choice(ruleset)


def getRandomDictForRule(rule):
    randomVals = {}
    # the convention is that the first expression is the rule, and the others define the values.
    for valuerule in rule[1:]:
        randomVals.update(instantiateRandomDict(valuerule))
    return randomVals


def getLogicForRule(rule):
    return rule[0]


# myrule = getRandomRule(also_composite=True)
# mylogic = getLogicForRule(myrule)
# mydict = getRandomDictForRule(myrule)
# print mylogic
# print mydict


def save_csv(filename, log, sep=';'):
    own_fid = False
    if isinstance(filename, basestring):  # a filename
        file = open(filename, 'w')
        own_fid = True
    else:
        file = filename
        myfilename = file.name
    cases = log.get_cases(True)
    print >> file, sep.join(['timestamp', 'time', 'case_id', 'activity', 'resource', 'event_type', 'cattr_priority', 'activity_count'])
    for i, case in enumerate(cases):
        for act in case:
            res_string = 'NA'
            lifecycle = 'NA'
            priority = 'NA'
            activity_count = 'NA'
            if 'resources' in act:
                res_string = ",".join(act['resources'])
            if 'lifecycle' in act:
                lifecycle = act['lifecycle']
            if 'priority' in act:
                priority = str(act['priority'])
            if 'activity_count' in act:
                activity_count = str(act['activity_count'])
            mytime = datetime.utcfromtimestamp(round(act['timestamp']))
            print >> file, sep.join([str(round(act['timestamp'])), str(mytime), "tr" + str(i), act['name'], res_string, lifecycle, priority, activity_count])


# getRules()['plain']


def create_new_instance(init_token):
    data = {'priority':0}
    if init_token.get_data():
        case_id = init_token.get_data()['cases']
        priority_chance = init_token.get_data()['priority_chance']
        activity_chance = init_token.get_data().get('activity_chance', 0.5)
        if random.random() < priority_chance:
            data['priority'] = 1
        more_activities = True
        activity_count = 0
        while more_activities:
            if random.random() < activity_chance:
                activity_count += 1
            else:
                more_activities = False
        data['activity_count'] = activity_count
    return Token("trace{}".format(case_id), data=data, time=init_token.get_time(), type=TokenType.instance)


def add_one(init_token):
    tok = Token(init_token.value, data=init_token.get_data().copy(), time=init_token.get_time())
    tok.get_data()['cases'] += 1
    tok.add_one = init_token.add_one
    tok.create_new_instance = init_token.create_new_instance
    return tok


class Creator(object):
    SEQUENTIAL = 1
    RUNNING_EXAMPLE = 2
    RANDOM = 3
    FULL_SET_OF_SEQUENTIAL_BATCHES = 4

    ARRIVAL = "arrival"

    def __init__(self, n_cases=1000, inter_arrival=10. / 3600000, n_activities=3, type=SEQUENTIAL, sim_start_time=0,
                 high_priority_chance=0.1, rule=getRandomRule(), add_extra_start=False):
        self.net = PetriNet('simple_batch')
        arrival_process = "arrival_process"
        self._resources = {}
        self.task_queues = {}
        self.create_cases(n_cases, arrival_process, inter_arrival, sim_start_time, high_priority_chance)
        self._allTheLetters = string.lowercase + string.uppercase

        previous = arrival_process
        # add explicit immediate arrival transition


        distribution = Distribution(dist_type=DistributionType.immediate)
        distribution_exit = Distribution(dist_type=DistributionType.immediate)
        logic = True
        if add_extra_start:
            logic = {'>=':[{'var':'n_in_queue'},10]}
            distribution_exit = Distribution(rate=3./ 3600.)

        activity = Creator.ARRIVAL
        self.add_activity(activity, dist=distribution, resources=[],
                          batchlogic=logic, batchdata={}, invisible=(False,True))  #only add complete of arrival
        self.wire_sequentially(previous, activity, dist=distribution_exit)
        previous = activity

        if type == Creator.SEQUENTIAL:
            for i in range(n_activities):
                activity = self.get_name(i)
                # rule = getRandomRule()
                rate = 10.
                if i % 3 == 0:  # a
                    rate = 15.
                elif i % 3 == 1:  # b
                    def my_activation_rule(self, **kwargs):
                        time = kwargs.get('time', 0)
                        numberwaiting = kwargs.get('numberwaiting', 1)
                        return numberwaiting >= self.params['n'] or time > self.params['max_wait']

                    # b_strategy = BatchStrategy(n=10, max_wait=50 * 60 * 1000, strategy=BatchStrategy.wait_till_n_cases)
                    # b_strategy.get_activate = types.MethodType(my_activation_rule, b_strategy)
                    rate = 12.
                elif i % 3 == 2:
                    def my_activation_rule2(self, **kwargs):
                        time = kwargs.get('time', 0)
                        numberwaiting = kwargs.get('numberwaiting', 1)
                        return numberwaiting >= self.params['n'] or time > self.params['max_wait']

                    # b_strategy = BatchStrategy(n=3, max_wait=10 * 60 * 1000, strategy=BatchStrategy.wait_till_n_cases)
                    # b_strategy.get_activate = types.MethodType(my_activation_rule2, b_strategy)
                    rate = 50.

                self.add_activity(activity, dist=Distribution(rate=rate / 3600.), resources=['r' + str(i)],
                                  batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))  # 1 per minute
                self.wire_sequentially(previous, activity)
                previous = activity
        elif type == Creator.RUNNING_EXAMPLE:
            rule = getRandomRule()
            activity = "A"
            self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=1, peak=4, max=7),
                              resources=['r1'], batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
            self.wire_sequentially(previous, activity)
            previous = activity
            activity = "B"
            rule = getRandomRule()
            self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=2, peak=5, max=7),
                              resources=['r2'], batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
            self.wire_sequentially(previous, activity)
            previous = activity
            activity = "C"
            rule = getRandomRule()
            self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=5, peak=8, max=11),
                              resources=['r3'], batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
            self.wire_sequentially(previous, activity)

            activity = "E"
            rule = getRandomRule()
            self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=3, peak=7, max=11),
                              resources=['r5'], batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
            self.wire_sequentially(previous, activity)
            previous = "C"
            activity = "D"
            rule = getRandomRule()
            self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=4, peak=6, max=8),
                              resources=['r4'], batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
            self.wire_sequentially(previous, "D")
            self.wire_sequentially("D", "D", 2)
            activity = "F"
            rule = getRandomRule()
            self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=2, peak=4, max=6),
                              resources=['r6'], batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
            self.wire_sequentially("D", "F", 8)
            self.wire_sequentially("E", "F")
            previous = "F"
            activity = "G"
            rule = getRandomRule()
            self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=1, peak=3, max=5),
                              resources=['r7'], batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
            self.wire_sequentially(previous, activity)
        elif type == Creator.FULL_SET_OF_SEQUENTIAL_BATCHES:
            all_rules = getRules()['plain']
            all_rules += getRules()['composite']

            for i, rule in enumerate(all_rules):
                activity = 'rule_' + str(i)
                self.add_activity(activity, Distribution(dist_type=DistributionType.triangular, min=1, peak=3, max=5),
                                  resources=['r' + str(i)], batchlogic=getLogicForRule(rule),
                                  batchdata=getRandomDictForRule(rule))
                self.wire_sequentially(previous, activity)
                previous = activity
        else:
            raise ValueError("type " + str(type) + " unsupported.")

    def get_name(self, i):
        j = i / 52
        val = i
        suffix = ''
        if j >= 1:
            val = j
            suffix = str(i)
        return self._allTheLetters[val] + suffix

    def add_activity(self, activity_name, dist=Distribution(), resources=["r1"], batchlogic=True, batchdata={},
                     sim_start_time=0, invisible=(False,False)):
        p_names = {State.queued.value: Unfolder.get_task_place_name(activity_name, State.queued),
                   State.service.value: Unfolder.get_task_place_name(activity_name, State.service),
                   State.finished.value: Unfolder.get_task_place_name(activity_name, State.finished)}

        # ensure that resource places are there!
        for res in resources:
            if res not in self._resources:
                res_place = Place(Unfolder.get_resource_place_name(res), [])
                tok = Token("res_{}".format(res), time=sim_start_time, type = TokenType.resource)
                res_place.add(tok)
                self.net.add_place(res_place)
                self._resources[res] = res_place

        p_queue = Place(p_names[State.queued.value], [])
        self.net.add_place(p_queue)
        p_service = Place(p_names[State.service.value], [])
        self.net.add_place(p_service)
        p_finish = Place(p_names[State.finished.value], [])
        self.net.add_place(p_finish)

        # create transitions for task:
        # enter:
        t_names = {Action.enter: Unfolder.get_transition_name(activity_name, Action.enter),
                   Action.exit: Unfolder.get_transition_name(activity_name, Action.exit)}

        self.net.add_transition(
            Transition(t_names[Action.enter], guard=Expression("True"), dist=dist, batchlogic=batchlogic,
                       batchdata=batchdata, invisible=invisible[0]))

        self.net.add_input(p_names[State.queued.value], t_names[Action.enter], Variable('trace'))
        tuple_parts = [Expression('trace')]
        variables = [Variable('trace')]
        counter = 0

        for res in resources:
            counter += 1
            res_name = 'res{}'.format(counter)
            self.net.add_input(Unfolder.get_resource_place_name(res), t_names[Action.enter], Variable(res_name))
            tuple_parts.append(Expression(res_name))
            variables.append(Variable(res_name))

        self.net.add_output(p_names[State.service.value], t_names[Action.enter], Tuple(tuple_parts))
        # exit:
        self.net.add_transition(
            Transition(t_names[Action.exit], guard=Expression("True"), dist=Distribution("immediate"), invisible=invisible[0]))
        self.net.add_input(p_names[State.service.value], t_names[Action.exit], Tuple(variables))
        self.net.add_output(p_names[State.finished.value], t_names[Action.exit], Expression('trace'))

        counter = 0
        for res in resources:
            counter += 1
            res_name = 'res{}'.format(counter)
            self.net.add_output(Unfolder.get_resource_place_name(res), t_names[Action.exit], Variable(res_name))

        self.task_queues[activity_name] = {State.queued.value: p_queue,
                                           State.service.value: p_service,
                                           State.finished.value: p_finish,
                                           Action.enter: t_names[Action.enter],
                                           Action.exit: t_names[Action.exit]}

    def wire_sequentially(self, predecessor, successor, weight=1, invisible=True, dist=Distribution("immediate"), guard=Expression("True")):
        name = Unfolder.get_task_transition_name(predecessor, successor)
        trans = Transition(name, guard=guard, dist=dist, invisible=invisible,
                           weight=weight)
        self.net.add_transition(trans)
        self.net.add_input(self.task_queues[predecessor][State.finished.value].name, name, Variable('trace'))
        self.net.add_output(self.task_queues[successor][State.queued.value].name, name, Expression('trace'))

    def create_cases(self, n_cases, arrival_process, inter_arrival_rate, sim_start_time, high_priority_chance=0.1):
        self.add_activity(arrival_process, Distribution(dist_type=DistributionType.exponential, rate=inter_arrival_rate),
                          resources=["ghost"], sim_start_time=sim_start_time, invisible=(True,True))
        p_init = self.task_queues[arrival_process][State.queued.value]
        trans_arrival = self.net.transition(self.task_queues[arrival_process][Action.enter])
        trans_arrival.guard = Expression('trace.get_data()["cases"]<{}'.format(n_cases))
        data = {'priority': 0, 'cases': 0, 'priority_chance': high_priority_chance}

        tok = Token(0, time=sim_start_time, data=data, type=TokenType.instance)
        tok.create_new_instance = create_new_instance
        tok.add_one = add_one
        trans_exit = self.task_queues[arrival_process][Action.exit]
        self.net.add_output(p_init.name, trans_exit, Expression('trace.add_one(trace)'))
        self.net.remove_output(self.task_queues[arrival_process][State.finished.value].name, self.task_queues[arrival_process][Action.exit])
        self.net.add_output(self.task_queues[arrival_process][State.finished.value].name, self.task_queues[arrival_process][Action.exit], Expression('trace.create_new_instance(trace)'))
        p_init.add(tok)

        # for i in range(n_cases):
        #     data = {'priority': 0}
        #     if random.random() < high_priority_chance:
        #         data['priority'] = 1
        #     tok = TimedToken("trace{}".format(i), time=sim_start_time, data=data)
        #     p_init.add(tok)

    def get_net(self):
        return self.net


if __name__ == '__main__':
    selective_tests = True
    doOnly = 21
    i = 0
    for type in ["plain","composite"]:
        for rule in getRules()[type]:

            #rule = getRules()['composite'][8]

            i += 1
            if selective_tests and doOnly == i:
                varnames = [name for name in extractVarNamesFromLogic(rule[0]) if not name.endswith("value")]

                folder = str(i)+"_"+"&".join(varnames)
                if not os.path.exists(folder):
                    os.makedirs(folder)

                extra_start = False  # delay tokens before arrival to be included in the upstream part of the model
                if varnames[0] == 'n_upstream_cases':
                    extra_start = True

                creator = Creator(n_cases=1000, inter_arrival=8. / (60 * 60*24), n_activities=3, type=Creator.SEQUENTIAL, rule=rule, add_extra_start=extra_start, high_priority_chance=0.2)
                net = creator.get_net()
                net.draw(folder+"/generated.png")
                sim = TimeSimulator(net)
                # TimeSimulator.DEBUG = True
                sim.set_produce_log(True)
                sim.simulate_one(queuing_policy=QueueingPolicy.earliest_due_date, experiment=True)
                # net.draw(folder+"/finished.png")
                traces = sim.get_log()
                log = pmlab.log.EnhancedLog(filename='log.xes', format='xes', cases=traces)
                # log.save(folder+"/"+'mylog.xes', format='csv')
                save_csv(folder+"/"+'log.csv', log)

                rules_text = ""
                for trans in net.transition():
                    if trans.get_batch_strategy().logic is not True:
                        rules_text += trans.name+":\n"+str(trans.get_batch_strategy()) + "\n\n"
                with open(folder+"/rules.txt",mode="w") as f:
                    f.write(rules_text)


