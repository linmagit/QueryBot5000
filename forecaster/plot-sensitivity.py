#!/usr/bin/env python3

import fnmatch
import csv
import sys
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mpdates
import matplotlib as mpl

from datetime import datetime, timedelta

import seaborn as sns
from sortedcontainers import SortedDict

DATA_DICT = {'admission': "~/peloton-tf/time-series-clustering/admission-combined-results/",
        'oli': "~/peloton-tf/time-series-clustering/oli-combined-results/",
        'tiramisu': '~/peloton-tf/time-series-clustering/tiramisu-combined-results/',
        }

INPUT_DIR = "../prediction-sensitivity-result/"
GRAPH_DIR = "../plot/sensitivity/"
ASSIGNMENT_DIR = "~/peloton-tf/time-series-clustering/online-clustering-results/"
HORIZON = "60"
AGGREGATE = 60
METHOD = "ar"
PROJECTS = ['admission', 'tiramisu', 'oli']
RHOS = ['0.5', '0.6', '0.7', '0.8', '0.9']
#RHOS = ['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95']

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S" # Strip milliseconds ".%f"


def SetupMplParams():
    color = sns.color_palette("hls", 4)
    mpl.rcParams.update(mpl.rcParamsDefault)

    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = [
            #r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
            #r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
            r'\usepackage{helvet}',    # set the normal font here
            r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
            r'\sansmath' # <- tricky! -- # gotta actually # tell tex to use!
            ]  
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    return color

def PlotLineChart(xaxis, data, name, xlabel, ylabel, color):
    fig = plt.figure()
    fig.set_size_inches(12, 2)
    ax = fig.add_subplot(111)
    loc = plticker.MultipleLocator(1) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    loc = plticker.MaxNLocator(5) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.grid()

    n = len(xaxis)
    ax.set_ylabel(ylabel,fontsize=18,weight='bold')
    ax.set_xlabel(xlabel,fontsize=18,weight='bold')
    #ax.set_xlim([xaxis[0], xaxis[-1]])
    #ax.set_ylim([0.6, 1.0])

    ax.set_xticks(range(n))
    ax.set_xticklabels(xaxis)

    ax.plot(range(n), data[0][:n], marker = 'o', color = color[1], label = 'Admissions', linewidth = 3)
    ax.plot(range(n), data[1][:n], marker = '^', color = color[3], label = 'BusTracker', linewidth = 3)
    ax.plot(range(n), data[2][:n], marker = 's', color = color[0], label = 'MOOC', linewidth = 3)
    #ax.legend(bbox_to_anchor = [1, 0.3], loc = 'lower right')
    #ax.legend(bbox_to_anchor = [0.5, 1], loc = 'lower center', ncol = 6)
    plt.savefig("%s%s.pdf" % (GRAPH_DIR, name), bbox_inches='tight')

    # produce a legend for the objects in the other figure
    figLegend = plt.figure(figsize = (4.5,0.4))
    plt.figlegend(*ax.get_legend_handles_labels(), loc = 'center', ncol=6, fontsize=20)
    figLegend.savefig(GRAPH_DIR + "legend.pdf", bbox_inches='tight')
    plt.close(figLegend)

    plt.close(fig)


def LoadData(input_path):
    total_queries = dict()
    templates = []
    min_date = datetime.max
    max_date = datetime.min
    data = dict()
    data_accu = dict()

    for csv_file in sorted(os.listdir(os.path.expanduser(input_path))):
        print(csv_file)
        with open(os.path.expanduser(input_path) + "/" + csv_file, 'r') as f:
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
            data_accu[template] = SortedDict()

            total = 0

            for line in reader:
                ts = datetime.strptime(line[0], DATETIME_FORMAT)
                time_stamp = datetime(ts.year, ts.month, ts.day, ts.hour, 0, 0)
                count = int(line[1])
                total += count
                if not time_stamp in data[template]:
                    data[template][time_stamp] = 0
                data[template][time_stamp] += count
                data_accu[template][time_stamp] = total

                min_date = min(min_date, time_stamp)
                max_date = max(max_date, time_stamp)

    templates = sorted(templates)

    return min_date, max_date, data, data_accu, total_queries, templates

def GetMSE(input_path):
    dates = []
    actual = []
    predict = []
    predict_dict = SortedDict()
    with open(input_path) as input_file:
        reader = csv.reader(input_file)
        for line in reader:
            dates.append(datetime.strptime(line[0], DATETIME_FORMAT))
            actual.append(max(0, float(line[1])))
            if line[2] == "inf":
                line[2] = 0
            predict.append(max(0, float(line[2])))
            predict_dict[dates[-1]] = predict[-1]

    y = np.array(actual)
    y_hat = np.array(predict)

    return predict_dict, dates[0], dates[-1]

def GetDataDict(input_dir):
    data_dict = {}

    losses = np.array([])
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in sorted(fnmatch.filter(filenames, '*.csv')):
            print(filename)
            file_path = os.path.join(root, filename)

            data = GetMSE(file_path)
            data_dict[file_path] = data

    return data_dict

def EvaluateAccuracy(data_dict, actual_data, project, assignment_dict, total_queries, data_accu,
        rho):
    print("start couting total_queries_per_cluster")
    total_queries_per_cluster = dict()

    for date, assignments in assignment_dict:
        cluster_dict = dict()
        total_queries_per_cluster[date] = cluster_dict
        for template, cluster in assignments.items():
            if cluster == -1:
                continue
            if cluster not in cluster_dict:
                cluster_dict[cluster] = 0

            template_total = data_accu[template][next(data_accu[template].irange(maximum
                = date, reverse = True))]
            cluster_dict[cluster] += template_total

    print("finish couting total_queries_per_cluster")

    accuracy_list = []
    for tp in data_dict:
        if (tp.find(project) >= 0 and tp.find("/" + rho + "/") >=0 and tp.find("horizon-" + HORIZON)
                >= 0 and tp.find("/{}/".format(METHOD)) >= 0):
            #print(project, rho, HORIZON)
            #print(tp)
            data = data_dict[tp]
            predict_cluster = int(os.path.splitext(os.path.basename(tp))[0])


            for current_date, assignments in assignment_dict:
                if current_date < data[1]:
                    continue
                if current_date > data[2] - timedelta(hours = 24):
                    break

                #print(current_date)
                cluster_total = total_queries_per_cluster[current_date]

                for template, cluster in assignments.items():
                    #print(cluster, predict_cluster)
                    if cluster != predict_cluster:
                        continue

                    dates = [current_date + timedelta(hours = i) for i in range(24)]
                    for i, date in enumerate(dates):
                        template_total = data_accu[template][next(data_accu[template].irange(maximum
                            = current_date, reverse = True))]

                        predict_value = data[0][next(data[0].irange(maximum = date, reverse = True))]
                        expected_value = (predict_value * template_total /
                                total_queries_per_cluster[current_date][cluster])

                        actual_value = actual_data[template][next(actual_data[template].irange(maximum = date, reverse = True))]
                        #print(rho, cluster, date, actual_value, expected_value, predict_value)
                        #accuracy_list.append((max(np.log(expected_value), 0) -
                        #    max(np.log(actual_value), 0)) ** 2)
                        accuracy_list.append((expected_value - actual_value) ** 2)

    if len(accuracy_list) > 0:
        return np.sum(np.log(accuracy_list)) / len(assignment_dict) / 3000
        #return np.sum(accuracy_list) / len(assignment_dict) / 3000
    else:
        return 0


def GetAccuracyData(data_dict):
    accuracies = []
    for project in PROJECTS:
        _, _, actual_data, data_accu, total_queries, _ = LoadData(DATA_DICT[project])
        accuracy_list = []
        for rho in RHOS:
            with open(os.path.expanduser(ASSIGNMENT_DIR) + "{}-{}-assignments.pickle".format(project, rho), 'rb') as f:
                num_clusters, assignment_dict, cluster_totals = pickle.load(f)

            accuracy = EvaluateAccuracy(data_dict, actual_data, project, assignment_dict,
                    total_queries, data_accu, rho)
            accuracy_list.append(accuracy)

        accuracies.append(accuracy_list)
        print(accuracy_list)

    return accuracies

def Main():
    color = SetupMplParams()

    data_dict = GetDataDict(INPUT_DIR)

    accuracy_data = GetAccuracyData(data_dict)

    PlotLineChart(RHOS, accuracy_data, "accuracy-sensitivity-{}-horizon{}".format(METHOD, HORIZON),
            "Similarity Threshold($\\rho$)", "MSE (log space)", color)


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    """
    Generate MSE result plots for the sensitivity analysis of rho
    """
    Main()
