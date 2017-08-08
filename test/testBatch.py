# coding: utf-8

import snakes.plugins

time_cpn = snakes.plugins.load(['timecpn', 'unfolder'], 'snakes.nets', 'time_cpn')
from time_cpn import extractVarNamesFromLogic, BatchAnnotator

logic = {">=": [{"var": "n_in_queue"}, {"var": "value"}]}

logic = {"and": [
    {"or": [
        {">=": [{"var": "longest_waiting_time"}, {"var": "value1"}]},
        {">=": [{"var": "mean_waiting_time"}, {"var": "value2"}]}
    ]},
    {"or": [
        {">=": [{"var": "time_since_last_arrival"}, {"var": "value3"}]},
        {">=": [{"var": "n_in_queue"}, {"var": "value4"}]}
    ]}
]}

annotator = BatchAnnotator("data/decision_rules_manufacturing_process_full_log.txt")
annotator.rules

varnames = extractVarNamesFromLogic(logic)
for varname in varnames:
    print varname

