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

SAMPLE_STEP = 50

OUTPUT = csv.writer(sys.stdout, quoting=csv.QUOTE_ALL)

def ProcessData(path, num_logs):
    data = []
    processed_queries = 0
    templated_workload = dict()

    min_timestamp = datetime.datetime.max
    max_timestamp = datetime.datetime.min

    #try:
    f = gzip.open(path, mode='rt')
    reader = csv.reader(f, delimiter=',')
    
    for i, query_info in enumerate(reader):
        processed_queries += 1

        if (not num_logs is None) and processed_queries > num_logs:
            break

        if i % SAMPLE_STEP == 0:
            OUTPUT.writerow(query_info)

# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='Templatize SQL Queries')
    aparser.add_argument('input', help='Input file')
    aparser.add_argument('--max_log', type=int, help='Maximum number of logs to process in a'
            'data file. Process the whole file if not provided')
    args = vars(aparser.parse_args())

    ProcessData(args['input'], args['max_log'])
    

