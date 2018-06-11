#!/usr/bin/env python3

import sys
import glob
import collections
import time
import csv
import os
import datetime
import gzip
import re
import argparse
from multiprocessing import Process

csv.field_size_limit(sys.maxsize)

STATEMENTS = ['select', 'SELECT', 'INSERT', 'insert', 'UPDATE', 'update', 'delete', 'DELETE']
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_STAMP_STEP = datetime.timedelta(minutes=1)

def MakeCSVFiles(workload_dict, min_timestamp, max_timestamp, output_dir):
    print("Generating CSV files...")
    print(output_dir)

    # Create the result folder if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # delete any old existing files
    for old_file in os.listdir(output_dir):
        os.remove(output_dir + old_file)

    template_count = 0
    for template in workload_dict:
        template_timestamps = workload_dict[
            template]  # time stamps for ith cluster
        num_queries_for_template = sum(template_timestamps.values())

        # write to csv file
        with open(output_dir + 'template' + str(template_count) +
                  ".csv", 'w') as csvfile:
            template_writer = csv.writer(csvfile, dialect='excel')
            template_writer.writerow([num_queries_for_template, template])
            for entry in sorted(template_timestamps):
                template_writer.writerow([entry, template_timestamps[entry]])
        csvfile.close()
        template_count += 1
    
    print("Template count: " + str(template_count))

def AddEntry(template, reader, min_timestamp, max_timestamp, templated_workload):

    # Finer process the template a bit to reduce the total template numbers
    template = re.sub(r"&&&", r"#", template)
    template = re.sub(r"@@@", r"#", template)
    template = re.sub(r"[nN]ull", r"#", template)
    template = re.sub(r"NULL", r"#", template)
    template = re.sub(r"\s+", r" ", template)
    template = re.sub(r"\( ", r"(", template)
    template = re.sub(r" \)", r")", template)
    template = re.sub(r"([^ ])\(", r"\1 (", template)
    template = re.sub(r"\)([^ ])", r") \1", template)
    template = re.sub(r" IN \([^\(]*?\)", r" IN ()", template)
    template = re.sub(r" in \([^\(]*?\)", r" IN ()", template)
    template = re.sub(r"([=<>,!\?])([^ ])", r"\1 \2", template)
    template = re.sub(r"([^ ])=", r"\1 =", template)

    #if (template.find("gradAdmissions2#Test") > 0 and template.find("INSERT") >= 0 and
    if (template.find("INSERT") >= 0 and
            template.find("VALUES") > 0):
        template = template[: template.find("VALUES") + 6]

    for line in reader:
        time_stamp = datetime.datetime.strptime(line[0], DATETIME_FORMAT)
        count = int(line[1])

        if not template in templated_workload:
            # add template
            templated_workload[template] = dict()

        if time_stamp in templated_workload[template]:
            templated_workload[template][time_stamp] += count
        else:
            templated_workload[template][time_stamp] = count

        min_timestamp = min(min_timestamp, time_stamp)
        max_timestamp = max(max_timestamp, time_stamp)

    return (templated_workload, min_timestamp, max_timestamp)


def Combine(input_dir, output_dir):

    templated_workload = dict()

    min_timestamp = datetime.datetime.max
    max_timestamp = datetime.datetime.min

    target = os.path.join(input_dir, "*/*template*.csv")
    print(target)
    files = sorted([ x for x in glob.glob(target) ])
    cnt = 0
    for x in files:
        print(x)
        with open(x, 'r') as f:
            reader = csv.reader(f)
            queries, template = next(reader)
            #statement = template.split(' ',1)[0]
            #if not statement in STATEMENTS:
            #    continue

            templated_workload, min_timestamp, max_timestamp = AddEntry(template, reader,
                    min_timestamp, max_timestamp, templated_workload)

        cnt += 1
        #if cnt == 1000:
        #    break

    print(min_timestamp)
    print(max_timestamp)
    with open('templates.txt', 'w') as template_file:
        [ template_file.write(t + "\n") for t in sorted(templated_workload.keys()) ]

    MakeCSVFiles(templated_workload, min_timestamp, max_timestamp, output_dir)




# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='Templated query csv combiner')
    aparser.add_argument('--input_dir', help='Input Data Directory')
    aparser.add_argument('--output_dir', help='Output Data Directory')
    args = vars(aparser.parse_args())

    Combine(args['input_dir'], args['output_dir'] + '/')
