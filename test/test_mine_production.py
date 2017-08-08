import sys
import os
path = './'
sys.path.append(os.path.abspath(path))
import pmlab
import pmlab.log
import pmlab.log.reencoders
import copy

from collections import defaultdict
from datetime import datetime, timedelta

from enum import Enum

#from snakes.nets import *
import snakes.plugins

timecpn = snakes.plugins.load(['gv', 'timecpn', 'unfolder'], 'snakes.nets', 'timecpn')
from timecpn import EnsureFinish, PetriNet, Distribution, DistributionType, TimeSimulator, BatchAnnotator, normalize_weekday

import snakes.plugins.gv as gv

import snakes.plugins.unfolder as unfolder
from datetime import datetime
import time

import csv

def perform_run(models, run, f, firstCall=False):
    if firstCall:
        f.write("model_name; case; act_start; act_end; pred_start; pred_end; finished; run_num\n")
    for (name, model) in models:
        m = model.get_marking()
        initial_marking = copy.deepcopy(m)
        sim = TimeSimulator(net=model)
        # if name == "QCSPN rp,rsr,fq":
        #     sim.snapshot_principle = True
        marking = sim.simulate_one(marking=copy.deepcopy(initial_marking))
        histories = sim.get_histories()
        # model.draw("model_{}_after_sim.png".format(name))
        finishedCases = [token.value for token in marking['exits']] if 'exits' in marking else []
        for key in histories:
            finishedCase = key in finishedCases
            f.write(histories[key].export_start_end(name, run=run, finished=1 if finishedCase else 0))
            f.write("\n")

        # written_tokens = set([])
        # for m_set in marking.values():
        #     for token in m_set.items():
        #         if snakes.plugins.timecpn.is_instance(token) and token.value not in written_tokens:
        #             written_tokens.add(token.value)
        #             # print "{}\n{}\n".format(token, token.export_history())
        #             f.write(token.export_start_end(name, run=run))
        #             f.write("\n")
        #             if sim.snapshot_principle:
        #                 f.write(token.export_snapshot_predictor("QCSPN rp,rsr,fq,snapshot", run=run))
        #                 f.write("\n")
        #                 f.write(token.export_queueing_predictor("QCSPN rp,rsr,fq,queueing", run=run))
        #                 f.write("\n")
        #                 f.write(token.export_mixed_predictor("QCSPN rp,rsr,fq,mixed", run=run))
        #                 f.write("\n")

        # restore initial marking for next iterations
        model.set_marking(initial_marking)
        f.flush()

def loadLog(folder, filename):
    with open(folder + filename) as csvfile:
        logreader = csv.DictReader(csvfile, delimiter=",")

        traces = []
        tracesMap = defaultdict(list)

        for row in logreader:
            resourceString = row['resource'] if 'resource' in row else row.get('Resource', '')
            resources = resourceString.split(" - ")[0]
            if resources.find("- ") > 0:
                resources = resourceString.split("- ")[0]

            if resources == '':
                resources = None
                # 2012/01/29 23:24:00.000
            if 'Start Timestamp' in row:
                start_ts = time.mktime(datetime.strptime(row['Start Timestamp'], '%Y/%m/%d %H:%M:%S.000').timetuple())
                end_ts = time.mktime(datetime.strptime(row['Complete Timestamp'], '%Y/%m/%d %H:%M:%S.000').timetuple())
            else:
                start_ts = time.mktime(datetime.strptime(row['start_time'], '%Y/%m/%d %H:%M:%S.000').timetuple())
                end_ts = time.mktime(datetime.strptime(row['complete_time'], '%Y/%m/%d %H:%M:%S.000').timetuple())
            caseId = row['Case ID'] if 'Case ID' in row else row['caseID']
            activity = row['Activity'] if 'Activity' in row else row['activity']
            tracesMap[caseId].append(
                {'traceId': caseId,
                 'name': activity.split(" - ")[0],
                 'timestamp': int(start_ts),
                 'duration': 60,
                 'resource': resources,
                 'start_time': int(start_ts),
                 'end_time': int(end_ts)
                 })

        for key in tracesMap.keys():
            # if key == 'Case 77':
            traces.append(tracesMap.get(key))
            # if len(traces) >= 40:
            #     break

        log = pmlab.log.EnhancedLog(filename=filename[:-3] + 'xes', cases=traces)
        print "done."
    return log


def loadResources(filename):
    resourceMap = {}
    with open(filename, mode='r') as f:
        logreader = csv.DictReader(f, delimiter=";")
        for row in logreader:
            resource = row["resource"]
            if resource not in resourceMap:
                resourceMap[resource] = {}
            weekday = normalize_weekday(row["weekday_start"])
            n = row["n_act_instances"]
            resourceMap[resource][weekday] = int(n)
    return resourceMap



def mine_model(filename="Production_2.csv", traininglog="Production_1.csv", resources = None, rules="data/decision_rules_manufacturing_process_part_1.txt", folder="data/", num_runs=3):
    rules = folder+rules
    log = loadLog(folder, filename)
    training_log = loadLog(folder, traininglog)

    [normalized_log, resource_capacity] = unfolder.normalize_resources(log)
    [normalized_training_log, resource_capacity] = unfolder.normalize_resources(training_log)

    unfold = unfolder.Unfolder()

    net = unfold.unfold(normalized_log, fuse_queues=True, scheduled=False)

    enricher = unfolder.Enricher(DistributionType.kernel_density_estimate)
    net = enricher.enrich(net, training_log, resources)

    if resources is None:
        resources = loadResources(folder+"resource_weekdays_n_act_inst.csv")

    # net.draw('model_init.png')
    net_batched = net.copy(net.name + '_batch')
    net_batched = enricher.enrich(net_batched, normalized_training_log)
    init_marking = net_batched.get_marking()
    net_batched.set_marking(copy.deepcopy(init_marking))
    annotator = BatchAnnotator(rules)
    net_batched = annotator.annotate_net(net_batched)

    net_batched_resource = net.copy(net.name + '_batch_resource')
    net_batched_resource = enricher.enrich(net_batched_resource, normalized_training_log, resources=resources)
    init_marking = net_batched_resource.get_marking()
    net_batched_resource.set_marking(copy.deepcopy(init_marking))
    annotator = BatchAnnotator(rules)
    net_batched_resource = annotator.annotate_net(net_batched_resource)


    net_batched_resource_prob = net.copy(net.name + '_batch_res_prob')
    net_batched_resource_prob = enricher.enrich(net_batched_resource, normalized_training_log, resources=resources)
    net_batched_resource_prob.probabilities = True
    init_marking = net_batched_resource_prob.get_marking()
    net_batched_resource_prob.set_marking(copy.deepcopy(init_marking))
    annotator = BatchAnnotator(rules)
    net_batched_resource_prob = annotator.annotate_net(net_batched_resource_prob)


    # net_batched.draw('model_init.png')
    net_batched_finish = net.copy(net.name + '_batch_ensure_finish')
    net_batched_finish = enricher.enrich(net_batched_finish, normalized_training_log)
    annotator = BatchAnnotator(rules, ensure_finish=EnsureFinish.NO_UPSTREAM_CASES)
    net_batched_finish = annotator.annotate_net(net_batched_finish)
    init_marking = net_batched_finish.get_marking()
    net_batched_finish.set_marking(copy.deepcopy(init_marking))

    # net_batched.draw('model_init.png')
    net_batched_finish_time = net.copy(net.name + '_batch_ensure_time')
    net_batched_finish_time = enricher.enrich(net_batched_finish_time, normalized_training_log)
    annotator = BatchAnnotator(rules, ensure_finish=EnsureFinish.LONG_WAITING_TIME)
    net_batched_finish_time = annotator.annotate_net(net_batched_finish_time)
    init_marking = net_batched_finish_time.get_marking()
    net_batched_finish_time.set_marking(copy.deepcopy(init_marking))

    net_batched_resource_time = net.copy(net.name + '_batch_resource_time')
    net_batched_resource_time = enricher.enrich(net_batched_resource_time, normalized_training_log, resources=resources)
    init_marking = net_batched_resource_time.get_marking()
    net_batched_resource_time.set_marking(copy.deepcopy(init_marking))
    annotator = BatchAnnotator(rules, ensure_finish=EnsureFinish.LONG_WAITING_TIME)
    net_batched_resource_time = annotator.annotate_net(net_batched_resource_time)

    # TimeSimulator.DEBUG = True

    f = open('output2_'+filename[:-4]+'.csv', 'w')
    models = [('without_batch', net),
              ('batched', net_batched),
              ('batched_resource_prob',net_batched_resource_prob),
              #('batched_finish', net_batched_finish),
              ('batched_finish_time', net_batched_finish_time),
              ('batched_resource', net_batched_resource),
              ('batched_resource_time', net_batched_resource_time)]
    # models = [('batched_finish_time', net_batched_finish_time)]
    firstCall=True
    for run in range(num_runs):
        print("---------  starting run {}  ------------".format(run))
        start_time = time.time()
        perform_run(models, run, f, firstCall=firstCall)
        if firstCall:
            firstCall = False
        print("---- run {} completed in %s seconds ----".format(run) % (time.time() - start_time))


os.chdir(path+"/test")

resources = loadResources(path+"data/resource_weekdays_n_act_inst.csv")

mine_model(filename="Production_2.csv", resources=resources, traininglog="Production_1.csv", rules="decision_rules_manufacturing_process_part_1.txt", folder=path+"data/")
mine_model(filename="Production_1.csv", resources=resources, traininglog="Production_2.csv", rules="decision_rules_manufacturing_process_part_2.txt", folder=path+"data/")

mine_model(filename="Production_Data.csv", resources=resources, traininglog="Production_Data.csv", rules="decision_rules_manufacturing_process_v4_full_log.txt", folder=path+"data/")

