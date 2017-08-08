# coding: utf-8

import csv


infile = "data/normalized_with_arrival.csv"
outfile = "data/normalized_trimmed_with_arrival.csv"

if __name__ == '__main__':
    with open(infile, 'rb') as f:
        reader = csv.reader(f)

        header = []
        entries = []
        for index, row in enumerate(reader):
            if index == 0:
                header = row
                header.append("Activity_Only")
                header.append("Resource_Only")
            if index > 0:
                entry = row[:]
                trimmed_activity = entry[1].split(" - ")
                trimmed_resource = entry[2].split(" - ")
                trimmed_activity = trimmed_activity[0]
                trimmed_resource = trimmed_resource[0]
                entry.append(trimmed_activity)
                entry.append(trimmed_resource)
                entries.append(entry)

        with open(outfile, 'wb') as fw:
            writer = csv.writer(fw, delimiter=',', quotechar='"')
            writer.writerow(header)
            for entry in entries:
                writer.writerow(entry)

