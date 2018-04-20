#!/usr/bin/env python3

import os
import sys
from datetime import datetime
import datetime as dt
import argparse
import csv
import numpy as np
import time
import itertools
import random
import pickle
import re
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mpdates
import matplotlib as mpl

from sortedcontainers import SortedDict

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from logical_clustering_utility.schemaParser import extract_tables_and_columns
from logical_clustering_utility.buildVectors import create_vectors

csv.field_size_limit(sys.maxsize)


OUTPUT_DIR = 'online-logical-clustering-results/'

YLABEL = r"# Queries / min"
STATEMENTS = ['select', 'SELECT', 'INSERT', 'insert', 'UPDATE', 'update', 'delete', 'DELETE']

# "2016-10-31","17:50:21.344030"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S" # Strip milliseconds ".%f"

TESTING = False

USE_KNN = True
KNN_ALG = "kd_tree"

def LoadData(input_path):
    total_queries = dict()
    templates = []
    min_date = datetime.max
    max_date = datetime.min
    data = dict()

    cnt = 0
    for csv_file in sorted(os.listdir(input_path)):
        print(csv_file)
        with open(input_path + "/" + csv_file, 'r') as f:
            reader = csv.reader(f)
            queries, template = next(reader)

            # To make the matplotlib work...
            template = template.replace('$', '')

            # Assume we already filtered out other types of queries when combining template csvs
            #statement = template.split(' ',1)[0]
            #if not statement in STATEMENTS:
            #    continue

            #print queries, template
            total_queries[template] = int(queries)
            #print queries

            templates.append(template)

            # add template
            data[template] = SortedDict()

            for line in reader:
                time_stamp = datetime.strptime(line[0], DATETIME_FORMAT)
                count = int(line[1])

                data[template][time_stamp] = count

                min_date = min(min_date, time_stamp)
                max_date = max(max_date, time_stamp)

        cnt += 1

        if TESTING:
            if cnt == 10:
                break

    templates = sorted(templates)

    return min_date, max_date, data, total_queries, templates

def Similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-6)

def AdjustCluster(min_date, current_date, next_date, data, last_ass, next_cluster, centers,
        cluster_totals, total_queries, cluster_sizes, rho, vector_dict):
    new_ass = last_ass.copy()

    print("Building kdtree for single point assignment")
    clusters = sorted(centers.keys())

    samples = list()

    for cluster in clusters:
        sample = centers[cluster]
        samples.append(sample)

    if len(samples) == 0:
        nbrs = None
    else:
        normalized_samples = normalize(np.array(samples), copy = False)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm=KNN_ALG, metric='l2')
        nbrs.fit(normalized_samples)

    print("Finish building kdtree for single point assignment")
    

    cnt = 0
    for t in sorted(data.keys()):
        cnt += 1
        # Test whether this template still belongs to the original cluster
        if new_ass[t] != -1:
            center = centers[new_ass[t]]
            #print(cnt, new_ass[t], Similarity(data[t], center, index))
            if cluster_sizes[new_ass[t]] == 1 or Similarity(vector_dict[t], center) > rho:
                continue

        # the template is eliminated from the original cluster
        if new_ass[t] != -1:
            cluster = new_ass[t]
            cluster_sizes[cluster] -= 1
            centers[cluster] -= vector_dict[t]
            cluster_totals[cluster] -= total_queries[t]
            print("%s: template %s quit from cluster %d with total %d" % (next_date, cnt, cluster,
                total_queries[t]))

        
        # Whether this template has "arrived" yet?
        if new_ass[t] == -1 and len(list(data[t].irange(current_date, next_date))) == 0:
            continue

        new_cluster = None
        if nbrs != None:
            # whether this template is similar to the center of an existing cluster
            nbr = nbrs.kneighbors(normalize([vector_dict[t]]), return_distance = False)[0][0]
            if Similarity(vector_dict[t], centers[clusters[nbr]]) > rho:
                new_cluster = clusters[nbr]

        if new_cluster != None:
            if new_ass[t] == -1:
                print("%s: template %s joined cluster %d with total %d" % (next_date, cnt,
                    new_cluster, total_queries[t]))
            else:
                print("%s: template %s reassigned to cluster %d with total %d" % (next_date,
                    cnt, new_cluster, total_queries[t]))

            new_ass[t] = new_cluster
            centers[cluster] += vector_dict[t]
            cluster_totals[cluster] += total_queries[t]
            cluster_sizes[new_cluster] += 1
            continue

        if new_ass[t] == -1:
            print("%s: template %s created cluster as %d with total %d" % (next_date, cnt,
                next_cluster, total_queries[t]))
        else:
            print("%s: template %s recreated cluster as %d with total %d" % (next_date, cnt,
                next_cluster, total_queries[t]))

        new_ass[t] = next_cluster
        centers[next_cluster] = vector_dict[t]
        cluster_sizes[next_cluster] = 1
        cluster_totals[next_cluster] = total_queries[t]

        next_cluster += 1

    clusters = list(centers.keys())
    # a union-find set to track the root cluster for clusters that have been merged
    root = [-1] * len(clusters)

    print("Building kdtree for cluster merging")
    samples = list()

    for cluster in clusters:
        sample = centers[cluster]
        samples.append(sample)

    if len(samples) == 0:
        nbrs = None
    else:
        normalized_samples = normalize(np.array(samples), copy = False)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm=KNN_ALG, metric='l2')
        nbrs.fit(normalized_samples)
    print("Finish building kdtree for cluster merging")

    for i in range(len(clusters)):
        c1 = clusters[i]
        c = None

        if nbrs != None:
            nbr = nbrs.kneighbors([centers[c1]], return_distance = False)[0]

            if clusters[nbr[0]] == c1:
                nbr = nbr[1]
            else:
                nbr = nbr[0]

            while root[nbr] != -1:
                nbr = root[nbr]

            if c1 != clusters[nbr] and Similarity(centers[c1], centers[clusters[nbr]]) > rho:
                c = clusters[nbr]

        if c != None:
            centers[c] += centers[c1]
            cluster_sizes[c] += cluster_sizes[c1]

            del centers[c1]
            del cluster_sizes[c1]

            if nbrs != None:
                root[i] = nbr

            for t in data.keys():
                if new_ass[t] == c1:
                    new_ass[t] = c
                    print("%d assigned to %d with total %d" % (c1, c, total_queries[t]))

            print("%s: cluster %d merged into cluster %d" % (next_date, c1, c))

    return new_ass, next_cluster


def OnlineClustering(min_date, max_date, data, total_queries, rho, vector_dict):
    print(rho)
    cluster_gap = 1440

    n = (max_date - min_date).seconds // 60 + (max_date - min_date).days * 1440 + 1 
    num_gaps = n // cluster_gap

    centers = dict()
    cluster_totals = dict()
    cluster_sizes = dict()

    assignments = []
    ass = dict()
    for t in data.keys():
        ass[t] = -1
    assignments.append((min_date, ass))

    current_date = min_date
    next_cluster = 0
    for i in range(num_gaps):
        next_date = current_date + dt.timedelta(minutes = cluster_gap)
        assign, next_cluster = AdjustCluster(min_date, current_date, next_date, data, assignments[-1][1],
                next_cluster, centers, cluster_totals, total_queries, cluster_sizes, rho,
                vector_dict)
        assignments.append((next_date, assign))

        current_date = next_date


    return next_cluster, assignments, cluster_totals


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='Logical clusreting')
    aparser.add_argument('--dir', default="combined-results", help='The directory that contains the time series'
            'csv files')
    aparser.add_argument('--schema_path', help='The path of the schema file')
    aparser.add_argument('--project', help='The name of the workload')
    aparser.add_argument('--rho', default=0.8, help='The threshold to determine'
        'whether a query template belongs to a cluster')
    args = vars(aparser.parse_args())

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    min_date, max_date, data, total_queries, templates = LoadData(args['dir'])

    # Pre-processing: template extraction + schema parsing + preprocessing
    schema_file = open(args['schema_path'], 'r')
    schema_dict = extract_tables_and_columns(schema_file)

    # Get logical vectors for query templates
    vector_dict = create_vectors(templates, schema_dict)

    num_clusters, assignment_dict, cluster_totals = OnlineClustering(min_date, max_date, data,
            total_queries, float(args['rho']), vector_dict)

    with open(OUTPUT_DIR + "{}-{}-assignments.pickle".format(args['project'], args['rho']),
            'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump((num_clusters, assignment_dict, cluster_totals), f)

    print(num_clusters)
    print(cluster_totals)
    print(sum(cluster_totals.values()))
    print(sum(total_queries.values()))
