# coding: utf-8

import csv
from collections import defaultdict

infile = "data/normalized.csv"
outfile = "data/normalized_with_arrival.csv"

def correct_timestamp(x):
    x['timestamp'] = x['timestamp'][:-4]
    return x

if __name__ == '__main__':
    with open(infile, 'rb') as f:
        reader = csv.DictReader(f, delimiter=',')

        # sort by timestamp and case:
        cases = defaultdict(list)
        for line in reader:
            cases[line['Case ID']].append(line)

        with open(outfile, 'wb') as fw:
            writer = csv.DictWriter(fw, reader.fieldnames, delimiter=',', quotechar='"')
            writer.writeheader()
            for c in cases:
                case_entries = sorted(cases[c], key=lambda k: k['timestamp'])
                # case_entries = map(correct_timestamp, case_entries)
                last_entry = None
                for case_entry in case_entries:
                    if case_entry['lifecycle:transition'] == 'start':
                        last_time = last_entry['timestamp'] if last_entry else case_entry['timestamp']
                        last_time_string = last_entry['Complete Timestamp'] if last_entry else case_entry['Complete Timestamp']
                        arrival_entry = case_entry.copy()
                        arrival_entry['timestamp'] = last_time
                        arrival_entry['Complete Timestamp'] = last_time_string
                        arrival_entry['lifecycle:transition'] = 'arrival'
                        writer.writerow(arrival_entry)
                    writer.writerow(case_entry)
                    last_entry = case_entry
