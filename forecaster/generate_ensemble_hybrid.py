#!/usr/bin/env python3

import copy
import fnmatch
import csv
from datetime import datetime, timedelta
import sys
import os
import numpy as np

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S" # Strip milliseconds ".%f"

SPIKE = "True"
PERCENT = 150

def GenerateData(data_dicts):

    r = []
    r_hat = []

    for data_dict in data_dicts:
        r.append(data_dict[0][0])
        r_hat.append(data_dict[0][1])

    data_actual = np.mean(np.array(r), axis=0)

    if SPIKE == "True":
        data_ensemble = []
        for i in range(len(r_hat[1])):
            if (r_hat[1][i] > 1000) and abs(r_hat[0][i] - r_hat[1][i]) / r_hat[0][i] > PERCENT / 100:
                data_ensemble.append(r_hat[1][i])
            else:
                data_ensemble.append(r_hat[0][i])

        data_ensemble = np.array(data_ensemble)
    else:
        r_hat = np.array(r_hat)
        data_min = 2 - np.min(r_hat)
        avg = np.log(r_hat + data_min)
        data_ensemble = np.exp(np.mean(avg, axis=0)) - data_min

    return (data_actual, data_ensemble)


def GetMSE(input_path):
    dates = []
    actual = []
    predict = []
    with open(input_path) as input_file:
        reader = csv.reader(input_file)
        for line in reader:
            dates.append(datetime.strptime(line[0], DATETIME_FORMAT))
            actual.append(max(0, float(line[1])))
            predict.append(max(0, float(line[2])))

    y = np.array(actual)
    y_hat = np.array(predict)

    return (y, y_hat), dates

def GetDataDict(input_dir):
    data_dict = {}

    losses = np.array([])
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in sorted(fnmatch.filter(filenames, '*.csv')):
            file_path = os.path.join(root, filename)
            print(filename, file_path)

            data_dict[filename] = GetMSE(file_path)

    return data_dict


def WriteResult(path, dates, actual, predict):
    with open(path, "w") as csvfile:
        writer = csv.writer(csvfile, quoting = csv.QUOTE_ALL)
        for x in range(len(dates)):
            writer.writerow([dates[x], actual[x], predict[x]])



def Main(input_dir1, input_dir2, output_dir, spike):
    global SPIKE
    if spike != None:
        SPIKE = spike
    delimiter = "/"
    output_dir += delimiter
    print("output_dir: " + output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dict1 = GetDataDict(input_dir1)
    data_dict2 = GetDataDict(input_dir2)

    for file_name in data_dict1:
        data = GenerateData([data_dict1[file_name], data_dict2[file_name]])
        if SPIKE:
            tag = ""
        else:
            tag = ""
        WriteResult(output_dir + tag + file_name, data_dict1[file_name][1], data[0], data[1])



# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    """
    Generate the output for ENSEMBLE (or HYBRID) model with the resutls from
    individual models.

    Args:
        arg1 : directory for the prediction result of the first (ensemble) model
        arg2 : directory for the prediction result of the second (kr) model
        arg3 : output directory of ensemble (hybrid)
        arg4 : whether we're generating hybrid method (True) or ensemble (False)
    """
    Main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
