import unittest

import snakes.plugins
import os

time_cpn = snakes.plugins.load(['timecpn', 'unfolder'], 'snakes.nets', 'time_cpn')
from time_cpn import PetriNet, Place, Transition, State, Action, Token, Distribution, DistributionType, Unfolder, \
    Variable, Expression, Tuple, FiringEvent, TimeSimulator, QueueingPolicy, BatchStrategy, instantiateRandomDict

import pmlab.log

from createModel import Creator, save_csv, getRules, getLogicForRule, getRandomDictForRule

class TestRules(unittest.TestCase):
    sim_start_time = 0
    n_cases = 500
    inter_arrival = 4./(60*60*24) # 4 people per day
    high_priority_chance = 0.1
    draw_model = False

    def setUp(self):
        self.creator = Creator(n_cases=TestRules.n_cases, inter_arrival=TestRules.inter_arrival, type=Creator.SEQUENTIAL, n_activities=1, high_priority_chance=TestRules.high_priority_chance)

        self.net = self.creator.get_net()
        self.activity_name = 'a'
        self.sim = TimeSimulator(self.net)
        # TimeSimulator.DEBUG = True
        self.sim.set_produce_log(True)

        # self.net = PetriNet('simple_batch')
        # # add resource
        # res = 1
        # res_place = Place(Unfolder.get_resource_place_name(res), [])
        # tok = TimedToken("res_{}".format(res), time=TestRules.sim_start_time)
        # res_place.add(tok)
        # self.net.add_place(res_place)
        # self._resources[res] = res_place
        #
        # # add arrival activity
        # self.ARRIVAL = "arrival"
        # self._resources = {}
        # self.task_queues = {}
        # self.create_cases(TestRules.n_cases, TestRules.inter_arrival, TestRules.sim_start_time, TestRules.high_priority_chance)
        #
        # # add one activiy A

    def test_n_in_queue(self):
        rulename = "01_n_in_queue"
        self.run_with_rule(rulename, 0)

    def test_hour_of_day(self):
        rulename = "02_hour_of_day"
        self.run_with_rule(rulename, 1)

    def test_part_of_day(self):
        rulename = "03_part_of_day"
        self.run_with_rule(rulename, 2)

    def test_04_day_of_week(self):
        rulename = "04_day_of_week"
        self.run_with_rule(rulename, 3)

    def test_05_longest_waiting_time(self):
        rulename = "05_longest_waiting_time"
        self.run_with_rule(rulename, 4)

    def test_06_mean_waiting_time(self):
        rulename = "06_mean_waiting_time"
        self.run_with_rule(rulename,5)

    def test_07_maximum_flow_time(self):
        rulename = "07_maximum_flow_time"
        self.run_with_rule(rulename,6)

    def test_08_time_since_last_arrival(self):
        rulename = "08_time_since_last_arrival"
        self.run_with_rule(rulename, 7)

    def test_09_workload(self):
        rulename = "09_workload"
        self.run_with_rule(rulename, 8)

    def test_10_priority(self):
        rulename = "10_priority"
        self.run_with_rule(rulename, 9)

    def test_11_aggregated_priority(self):
        rulename = "11_aggregated_priority"
        self.run_with_rule(rulename, 10)

    def test_12_activity_execution_counter(self):
        rulename = "12_activity_execution_counter"
        p_init = self.net.place('parrival_finished') # self.creator.task_queues[self.creator.ARRIVAL][State.queued.value])
        data = {'priority': 0}
        tok = Token("trace{}".format(50), time=0, data=data)
        p_init.add(tok)
        tok.trace_firing(FiringEvent('ta_enter', 0, 1))
        tok.trace_firing(FiringEvent('ta_enter', 1, 1))
        tok.trace_firing(FiringEvent('ta_enter', 2, 1))
        tok.trace_firing(FiringEvent('ta_enter', 3, 1))
        tok.set_time(4)

        self.run_with_rule(rulename, 11)

    def test_13_n_upstream_cases(self):
        rulename = "13_n_upstream_cases"
        self.run_with_rule(rulename, 12)

    def test_14_n_in_queue_longest_waiting_time(self):
        rulename = "14_n_in_queue_longest_waiting_time"
        self.run_with_rule(rulename, 13)

    def test_15_part_of_day_day_of_week_n_in_queue(self):
        rulename = "15_part_of_day_workload_n_in_queue"
        self.run_with_rule(rulename, 14)

    def test_16_mean_waiting_time_aggr_attributes_priority(self):
        rulename = "16_mean_waiting_time_aggr_attribute_priority"
        self.run_with_rule(rulename, 15)

    def test_17_day_of_week_part_of_day(self):
        rulename = "17_day_of_week_part_of_day"
        self.run_with_rule(rulename, 16)

    def test_18_n_in_queue_time_since_last_arrival_workload(self):
        rulename = "18_n_in_queue_time_since_last_arrival_workload"
        self.run_with_rule(rulename, 17)

    def test_20_longest_waiting_time_mean_waiting_time_time_since_last_arrival_n_in_queue(self):
        rulename = "20_longest_waiting_time_mean_waiting_time_time_since_last_arrival_n_in_queue"
        self.run_with_rule(rulename, 19)

    def test_21_n_upstream_cases_longest_waiting_time(self):
        rulename = "21_n_upstream_cases_longest_waiting_time"
        self.run_with_rule(rulename, 20)

    def test_22_hour_of_day_part_of_day_n_in_queue(self):
        rulename = "22_hour_of_day_part_of_day_n_in_queue"
        self.run_with_rule(rulename, 21)

    def run_with_rule(self, rulename, rule_id):
        if not os.path.exists(rulename):
            os.makedirs(rulename)
        if self.draw_model:
            self.net.draw("{}/generated_running.png".format(rulename))
        plainRuleCount = len(getRules()['plain'])
        if rule_id < plainRuleCount:
            rule = getRules()['plain'][rule_id]
        else:
            rule = getRules()['composite'][rule_id - plainRuleCount]
        b_strat = BatchStrategy(batchlogic=getLogicForRule(rule), batchdata=getRandomDictForRule(rule))
        with open("{}/rule.txt".format(rulename), "w") as text_file:
            text_file.write(str(b_strat))
        self.net.transition(Unfolder.get_transition_name(self.activity_name, Action.enter)).set_batch_strategy(b_strat)
        self.net.transition(Unfolder.get_task_transition_name(self.creator.ARRIVAL, "a")).set_invisible(False)
        self.sim.simulate_one(queuing_policy=QueueingPolicy.earliest_due_date, marking=self.net.get_marking()) #, rulename=rulename)
        if self.draw_model:
            self.net.draw("{}/generated_finish.png".format(rulename))
        traces = self.sim.get_log()
        self.log = pmlab.log.EnhancedLog(filename='schedule2.xes', format='xes', cases=traces)
        # log.save('mylog.xes', format='csv')
        save_csv("{}/mylog.csv".format(rulename), self.log)



if __name__ == '__main__':
    unittest.main()

