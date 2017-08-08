# coding: utf-8

import csv

from datetime import datetime, timedelta
import time


infile = "data/Production_Data.csv"
outfile = "data/normalized.csv"

if __name__ == '__main__':
    with open(infile, 'rb') as f:
        reader = csv.reader(f)

        header = []
        entries = []
        for index, row in enumerate(reader):
            if index == 0:
                header = row
                header += ['timestamp']
            if index > 0:
                entry_start = row[:]
                evt_time = datetime.strptime(row[3][:-4], "%Y/%m/%d %H:%M:%S")
                start_time = evt_time-timedelta(hours=23)

                entry_start[9] = 'start'
                entry_start[3] = start_time.strftime("%Y/%m/%d %H:%M:%S.000")
                entry_start += [time.mktime(start_time.timetuple())]

                entry_complete = row[:]
                entry_complete += [time.mktime(evt_time.timetuple())]

                entries.append(entry_start)
                entries.append(entry_complete)

        with open(outfile, 'wb') as fw:
            writer = csv.writer(fw, delimiter=',', quotechar='"')
            writer.writerow(header)
            for entry in entries:
                writer.writerow(entry)

