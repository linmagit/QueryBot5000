#!/usr/bin/env python3

import fnmatch
import csv
from datetime import datetime, timedelta
import sys
import os
import numpy as np

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S" # Strip milliseconds ".%f"

def GetMSE(input_path):
    dates = []
    actual = []
    predict = []
    with open(input_path) as input_file:
        reader = csv.reader(input_file)
        for line in reader:
            #dates.append(datetime.strptime(line[0], DATETIME_FORMAT))
            actual.append(float(line[1]))
            predict.append(float(line[2]))

    y = np.array(actual)
    y_hat = np.array(predict)

    data_min = 2 - np.min([np.min(y), np.min(y_hat)])
    se = (np.log(y + data_min) - np.log(y_hat + data_min)) ** 2
    print("MSE of %s: %s" % (input_path, np.mean(se)))

    return se

# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    losses = np.array([])
    for root, dirnames, filenames in os.walk(sys.argv[1]):
        for filename in sorted(fnmatch.filter(filenames, '*.csv')):
            print(filename)
            file_path = os.path.join(root, filename)
            losses = np.append(losses, GetMSE(file_path))

    print(np.mean(losses))
