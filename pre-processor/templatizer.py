#!/usr/bin/env python3.5

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

TIME_STAMP_STEP = datetime.timedelta(minutes=1)
STATEMENTS = ['select', 'SELECT', 'INSERT', 'insert', 'UPDATE', 'update', 'delete', 'DELETE']

# ==============================================
# PROJECT CONFIGURATIONS
# ==============================================

PROJECTS = {
    "tiramisu": {
        "name": "tiramisu",
        "files": "dbp*postgresql-*.anonymized.sample.gz",
        "mysql": False,
        "query_index": 3,
        "time_stamp_format": "%Y-%m-%d %H:%M:%S"
    },
    "admissions": {
        "name": "admissions",
        "files": "magneto.log.*.anonymized.gz",
        "mysql": True,
        "type_index": 3,
        "query_index": 4,
        "time_stamp_format": "%Y-%m-%d %H:%M:%S"
    },
    "oli": {
        "name": "oli",
        "files": "db*logfile*.anonymized.gz",
        "mysql": True,
        "type_index": 2,
        "query_index": 3,
        "time_stamp_format": "%y%m%d %H:%M:%S"
    }
}


def ProcessData(path, output_dir, num_logs, config):
    # input: string of path to csv file
    # output: array of tuples
    #           tuple setup: (time_stamp, query)
    #           type: (datetime object, string)

    # Define time tracker
    #over_all_start = time.time()
    #start = time.time()

    print("Start processing: " + path)

    data = []
    processed_queries = 0
    templated_workload = dict()

    min_timestamp = datetime.datetime.max
    max_timestamp = datetime.datetime.min

    try:
        f = gzip.open(path, mode='rt')
        reader = csv.reader(f, delimiter=',')
        
        for query_info in reader:
            processed_queries += 1

            if (not num_logs is None) and processed_queries > num_logs:
                break

            if config['name'] == 'tiramisu':
                time_stamp = query_info[0]
                time_stamp = time_stamp[: -8] # remove milliseconds and the time zone

            else:
                if query_info[config['type_index']] != 'Query':  # skip if not a query
                    continue

                # create timestamp
                if config['name'] == 'admissions':
                    day = query_info[0]
                    time = query_info[1].split(".")[0]  # removes the milliseconds
                    time_stamp = day + " " + time

                if config['name'] == 'oli':
                    time_stamp = query_info[0]
                    if time_stamp[7] == ' ':
                        time_stamp = time_stamp[0: 7] + '0' + time_stamp[8: -1]
            #IF

            time_stamp = datetime.datetime.strptime(
                time_stamp, config['time_stamp_format'])
            time_stamp = time_stamp.replace(second=0)  # accurate to the minute
            # Format query
            query = query_info[config['query_index']]

            for stmt in STATEMENTS:
                idx = query.find(stmt)
                if idx >= 0:
                    break

            if idx < 0:
                continue
            
            min_timestamp = min(min_timestamp, time_stamp)
            max_timestamp = max(max_timestamp, time_stamp)

            # Update query templates
            GetTemplate(query[idx:], time_stamp, templated_workload)

    except Exception as e:
        print("It might be an incomplete file. But we continue anyway.")
        print(e)


    MakeCSVFiles(templated_workload, min_timestamp, max_timestamp, output_dir + '/' +
            path.split('/')[-1].split('.gz')[0] + '/')

    #end = time.time()
    #print("Preprocess and template extraction time for %s: %s" % (path, str(end - start)))

def GetTemplate(query, time_stamp, templated_workload):
    # CHANGE: Returns a dictionary, where keys are templates, and they map to
    # a map of timestamps map to query counts with that timestamp

    STRING_REGEX = r'([^\\])\'((\')|(.*?([^\\])\'))'
    DOUBLE_QUOTE_STRING_REGEX = r'([^\\])"((")|(.*?([^\\])"))'

    INT_REGEX = r'([^a-zA-Z])-?\d+(\.\d+)?' # To prevent us from capturing table name like "a1"

    HASH_REGEX = r'(\'\d+\\.*?\')'

    template = re.sub(HASH_REGEX, r"@@@", query)
    template = re.sub(STRING_REGEX, r"\1&&&", template)
    template = re.sub(DOUBLE_QUOTE_STRING_REGEX, r"\1&&&", template)
    template = re.sub(INT_REGEX, r"\1#", template)

    if template in templated_workload:
        # add timestamp
        if time_stamp in templated_workload[template]:
            templated_workload[template][time_stamp] += 1
        else:
            templated_workload[template][time_stamp] = 1
    else:
        templated_workload[template] = dict()
        templated_workload[template][time_stamp] = 1

    return templated_workload


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
        #print(template)
        template_timestamps = workload_dict[
            template]  # time stamps for ith cluster
        #time_stamp_dict = collections.OrderedDict()
        num_queries_for_template = sum(template_timestamps.values())

        # loops over timestamps stepping by TIME_STAMP_STEP
        #for i in range(
        #        int((max_timestamp - min_timestamp) / TIME_STAMP_STEP) + 1):
        #    time_stamp = min_timestamp + (i * TIME_STAMP_STEP)
        #    if time_stamp in template_timestamps:
        #        count = template_timestamps[time_stamp]
        #    else:
        #        count = 0

        #    time_stamp_dict[time_stamp] = count

        # write to csv file
        with open(output_dir + 'template' + str(template_count) +
                  ".csv", 'w') as csvfile:
            template_writer = csv.writer(csvfile, dialect='excel')
            template_writer.writerow([num_queries_for_template, template])
            for entry in sorted(template_timestamps.keys()):
                template_writer.writerow([entry, template_timestamps[entry]])
            #for entry in time_stamp_dict:
            #    template_writer.writerow([entry, time_stamp_dict[entry]])
        csvfile.close()
        template_count += 1
    
    print("Template count: " + str(template_count))

def ProcessAnonymizedLogs(input_dir, output_dir, max_log, config):
    target = os.path.join(input_dir, config['files'])
    files = sorted([ x for x in glob.glob(target) ])

    proc = []
    for i, log_file in enumerate(files):
        #if i < 45:
        #    continue
        print(i, log_file)

        #continue

        # Process log
        p = Process(target = ProcessData, args = (log_file, output_dir, max_log, config))
        p.start()
        proc.append(p)

    for p in proc:
        p.join()


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='Templatize SQL Queries')
    aparser.add_argument('project', choices=PROJECTS.keys(), help='Data source type')
    aparser.add_argument('--dir', help='Input Data Directory')
    aparser.add_argument('--output', help='Output data directory')
    aparser.add_argument('--max_log', type=int, help='Maximum number of logs to process in a'
            'data file. Process the whole file if not provided')
    args = vars(aparser.parse_args())

    ProcessAnonymizedLogs(args['dir'], args['output'], args['max_log'], PROJECTS[args['project']])
    

