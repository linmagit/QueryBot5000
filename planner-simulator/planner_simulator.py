#!/usr/bin/env python3

import os
import csv
import fnmatch
import re
import sys
import pickle
import argparse
import sqlparse
import shutil
import datetime as dt
from datetime import datetime, timedelta

from sortedcontainers import SortedDict

from schemaParser import extract_tables_and_columns

class Simulator:
    """Index suggestion algorithm that simulates a basic "what-if" API

    We recommend indexes based on the workload forecasting results at the
    current time stamp. We only recommend single-column indexes here.
    The expected arrival rate of each query template is calculated by the
    predicted arrival rate of the cluster it belongs to and the ratio between
    the volume of this template and the total volume of the cluster.
    The benefit of the index is estimated also by the cardinality of the column
    and whether the query can already use another index.
    """
    def __init__(self, schema_file, original_path, predicted_path, assignment_path,
            top_cluster_path, max_cluster_num, aggregate, column_card, static_suggest):
        # params
        self.aggregate = aggregate
        self.max_cluster_num = max_cluster_num
        self.column_card = column_card
        self.static_suggest = static_suggest

        with open(assignment_path, 'rb') as f:
            _, self.assignment_dict, _ = pickle.load(f)

        with open(top_cluster_path, 'rb') as f:
            self.top_cluster, _ = pickle.load(f)

        # Pre-processing: template extraction + schema parsing + preprocessing
        sql_schema = open(schema_file, 'r')
        self.schema_dict = extract_tables_and_columns(sql_schema)

        self.data, self.total_queries, self.template_map = LoadOriginalData(original_path)

        self.predicted_data = LoadMultiplePredictedData(predicted_path)

        tables = self.schema_dict.keys()

        self.templates_dict = GetAccessDict(sorted(self.template_map.values()), tables, self.schema_dict)

        self.last_date = None
        # Clear the total_queries and calculate it online
        for template in self.total_queries:
            self.total_queries[template] = 0


    def SuggestIndex(self, start_date, duration, index_set):
        predicted_dict = BuildDictionary(self.schema_dict)
        predicted_data = self.predicted_data
        total_queries = self.total_queries
        aggregate = self.aggregate
        column_card = self.column_card
        template_map = self.template_map
        static_suggest = self.static_suggest
        data = self.data

        print(list(total_queries.values())[:30])

        if static_suggest is True:
            # Suggest index based on all the workload trace until now if static_suggest is True

            if self.last_date is None:
                for template in total_queries:
                    for date in data[template].irange(None, start_date):
                        total_queries[template] += data[template][date]
            self.last_date = start_date

            for template in total_queries:
                if self.total_queries[template] < 100:
                    continue

                template_dict = self.templates_dict[template_map[template]]
                weight = 10000
                for pair in template_dict:
                    if pair in index_set:
                        weight = 1

                for pair in template_dict:
                    predicted_dict[pair] += total_queries[template] * column_card[pair] * weight
        else:
            if self.last_date is None:
                for template in total_queries:
                    for date in data[template].irange(start_date - timedelta(weeks = 2), start_date):
                        total_queries[template] += data[template][date]
            else:
                for template in total_queries:
                    for date in data[template].irange(self.last_date, start_date):
                        total_queries[template] += data[template][date]
                for template in total_queries:
                    for date in data[template].irange(self.last_date - timedelta(weeks = 2), start_date -
                            timedelta(weeks = 2)):
                        total_queries[template] -= data[template][date]

            self.last_date = start_date


            for date, ass in self.assignment_dict:
                if date > start_date:
                    print(date)
                    break

                assignments = ass

            for date, cluster in self.top_cluster:
                if date > start_date:
                    print(date)
                    break

                cluster_list = cluster

            print("Clusters: ", cluster_list)
            clusters = next(zip(*cluster_list))[:self.max_cluster_num]
            print("Clusters: ", clusters, type(clusters[0]))

            total_queries_per_cluster = dict()
            for cluster in clusters:
                total_queries_per_cluster[cluster] = 0

            for template, cluster in assignments.items():
                if cluster not in clusters:
                    continue
                total_queries_per_cluster[cluster] += self.total_queries[template]

            cnt = 0
            cnt2 = 0
            for template, cluster in assignments.items():
                template_dict = self.templates_dict[template_map[template]]

                cnt2 += 1
                if self.total_queries[template] < 100 or cluster not in clusters:
                    continue
                cnt += 1
                print(type(cluster), cluster, total_queries[template], template[:50])

                weight = 10000
                for pair in template_dict:
                    if pair in index_set:
                        weight = 1

                print(template)
                print(weight, "\n")


                for j in range(0, 60, aggregate):
                    if j >= duration:
                        break

                    for pair in template_dict:
                        predict_date = next(self.predicted_data[0][cluster].irange(maximum = start_date +
                            timedelta(minutes = j), reverse = True))

                        predicted_dict[pair] += (predicted_data[0][cluster][predict_date] *
                            total_queries[template] / total_queries_per_cluster[cluster] * 100 *
                            column_card[pair] * weight)


                for j in range(60, 1440, aggregate):
                    if j >= duration:
                        break

                    for pair in template_dict:
                        predict_date = next(self.predicted_data[1][cluster].irange(maximum = start_date +
                            timedelta(minutes = j), reverse = True))

                        predicted_dict[pair] += (predicted_data[1][cluster][predict_date] *
                            total_queries[template] / total_queries_per_cluster[cluster] * 10 *
                            column_card[pair] * weight)

                for j in range(1440, 10080, aggregate):
                    if j >= duration:
                        break

                    for pair in template_dict:
                        predict_date = next(self.predicted_data[2][cluster].irange(maximum = start_date +
                            timedelta(minutes = j), reverse = True))

                        predicted_dict[pair] += (predicted_data[2][cluster][predict_date] *
                            total_queries[template] / total_queries_per_cluster[cluster] * 1 *
                            column_card[pair] * weight)

            print("Valid queries: ", cnt, cnt2)
            print(clusters)

        predicted_sorted_columns = sorted(predicted_dict.items(), key=lambda x: x[1], reverse = True)

        for pair in predicted_sorted_columns:
            print(pair)
            #if pair[1] == 0:
            #    break
            if not pair[0] in index_set:
                return pair[0]

        return None

def LoadOriginalData(input_path):
    datetime_format = "%Y-%m-%d %H:%M:%S" # Strip milliseconds ".%f"

    total_queries = dict()
    min_date = datetime.max
    max_date = datetime.min
    data = dict()

    # This is to keep track of our modification to the template. We have to keep the original
    # templates to restore the order in the clustering assignments.
    modified_template_map = dict()

    for csv_file in sorted(os.listdir(input_path)):
        print(csv_file)
        with open(input_path + "/" + csv_file, 'r') as f:
            reader = csv.reader(f)
            queries, template = next(reader)

            # To make the matplotlib work...
            template = template.replace('$', '')

            modified_template = template
            # replace '#' with 'param_holder' for sql parsing
            modified_template = modified_template.replace('#', "param_holder")
            # convert to lower case for matching convenience
            #modified_template = modified_template.lower()
            modified_template_map[template] = modified_template

            #print queries, template
            total_queries[template] = int(queries)

            # add template
            data[template] = SortedDict()

            #continue
            
            for line in reader:
                time_stamp = datetime.strptime(line[0], datetime_format)
                count = int(line[1])

                data[template][time_stamp] = count

                min_date = min(min_date, time_stamp)
                max_date = max(max_date, time_stamp)

    return data, total_queries, modified_template_map


def LoadData(file_path, aggregate):
    trajs = dict()

    datetime_format = "%Y-%m-%d %H:%M:%S" # Strip milliseconds ".%f"
    for csv_file in sorted(os.listdir(file_path)):
        print(csv_file)

        cluster = int(os.path.splitext(csv_file)[0])
        trajs[cluster] = SortedDict()

        with open(file_path + "/" + csv_file, 'r') as f:
            reader = csv.reader(f)

            traj = list()
            date = list()
            cnt = 0
            tot = 0

            for line in reader:
                cnt += 1
                tot += float(line[1])

                if cnt % aggregate == 0:
                    time_stamp = datetime.strptime(line[0], datetime_format)
                    trajs[cluster][time_stamp] = tot

                    tot = 0

    return trajs

def LoadMultiplePredictedData(paths):
    predicted_data = []

    for path in paths:
        predicted_data.append(LoadData(path, 1))

    return predicted_data


def BuildDictionary(schema_dict):
    d = dict()

    for table, column_dict in schema_dict.items():
        for column in column_dict:
            d[(table, column)] = 0

    return d

def GetAccessDict(templates, tables, schema_dict):
    templates_dict = dict()

    for template in templates:
        #print("processing template: %s" % template)
        sql = sqlparse.parse(template)[0]
        token_list = [str(x) for x in sql.flatten()]

        token_set = set()

        table_map = dict()
        before_from = True
        #print()
        #print(template)

        # Find the alias for table names
        keywords = ['upper', 'lower', 'left', 'right', 'join', 'as', ',', 'group', 'order', 'set']
        word_regex = re.compile('[\w]+')
        for i, token in enumerate(token_list):
            # Only look between from and where
            if token.lower() in ["from", "update"]:
                before_from = False
            if before_from:
                continue
            if token.lower() in ["set", 'where']:
                before_from = True

            table_name = None
            if token in tables:
                table_name = token
            if i > 1:
                combined_token = token_list[i - 2] + '.' + token
                if combined_token in tables:
                    table_name = combined_token

            if table_name != None:
                if i < len(token_list) - 2:
                    if token_list[i + 1] == ' ' and token_list[i + 2].lower() not in keywords:
                        if word_regex.match(token):
                            table_map[token_list[i + 2]] = table_name

                if i < len(token_list) - 4:
                    if token_list[i + 2].lower() == 'as':
                        table_map[token_list[i + 4]] = table_name

                table_map[table_name] = table_name

        #print("table_map: ", table_map)

        current_table = None
        within_where_clause = False
        for token in token_list:
            # The table does not exist in the schema file...
            # That happends with the test table
            if len(table_map) == 0:
                break

            # only consider columns within where clause
            if token.lower() in ['select', 'order', 'group', 'returning']:
                within_where_clause = False

            if within_where_clause:

                if token == 'param_holder':
                    continue

                if word_regex.match(token):
                    if token in table_map:
                        current_table = table_map[token]
                        continue

                    if current_table == None:
                        current_table = next(iter(table_map.values()))

                    if token in schema_dict[current_table]:
                        token_set.add((current_table, token))
                    current_table = None

            # only consider columns within where clause
            if token == 'where' or token == 'WHERE':
                within_where_clause = True

        templates_dict[template] = token_set
        #print(token_set)

    return templates_dict


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    SimulatorObject = Simulator("../workload-simulator/simulatorFiles/combined-results",
                                ["../workload-simulator/simulatorFiles/online-prediction/ar-60","../workload-simulator/simulatorFiles/online-prediction/ar-1440","../workload-simulator/simulatorFiles/online-prediction/ar-10080"],
                                "../workload-simulator/simulatorFiles/cluster-coverage/admission-assignments.pickle",
                                "../workload-simulator/simulatorFiles/cluster-coverage/admission-coverage.pickle",
                                3,
                                60
                                )
    index = SimulatorObject.SuggestIndex(datetime(2017, 1, 1), 300, [])
    print(index)


