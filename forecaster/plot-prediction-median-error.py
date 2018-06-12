#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mpdates
import matplotlib.patches as mpatches
import matplotlib as mpl
import copy
import fnmatch
import csv
from datetime import datetime, timedelta
import sys
import os
import numpy as np

import seaborn as sns

PROJECT = "admission-full"
#PROJECT = "admission"
#PROJECT = "oli"
#PROJECT = "tiramisu"

GRAPH_DIR = "../plot/compare-avg-result/"

INPUT_DIM = 1
HORIZON = 0
INTERVAL = 12
TRAIN_NUM = 48

def SetupMplParams():
    #color = sns.color_palette("deep", 7)
    color = sns.color_palette("Set1", n_colors=8, desat=.7)
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

def Plot(data_dict, name, xlabel, ylabel, color):
    fig_length = 12

    fig = plt.figure()
    fig.set_size_inches(fig_length, 2)
    ax = fig.add_subplot(111)
    loc = plticker.MaxNLocator(3) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.grid()

    ax.set_ylabel(ylabel, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=18, weight='bold')

    models = ["ar", "arma", "fnn", 'rnn', 'noencoder-rnn', 'psrnn', 'psrnn-h5']
    labels = ["AR", "arma", "FNN", "RNN", "NE-RNN", "PSRNN", "PSRNN-H5"]
    horizons = ['60', '720', '1440', '2880', '4320', '7200', '10080']
    xlabels = ["1 Hour", "12 Hour", "1 Day", "2 Days", "3 Days", "5 Days", "1 Week"]

    result = []

    for i, m in enumerate(models):
        res = []
        for h in horizons:
            print(m, h)
            r = np.array([])
            r_hat = np.array([])
            for tp in data_dict:
                if tp.find(PROJECT) >= 0 and tp.find("/" + m + "/") >=0 and tp.find("horizon-" + h) >= 0:
                    r = np.append(r, data_dict[tp][0])
                    r_hat = np.append(r_hat, data_dict[tp][1])

            res.append(np.mean((r - r_hat) ** 2))
            print(np.mean(r - r_hat) ** 2)

        #result.append(np.log(res))
        result.append(res)

    x = [x * 8 for x in range(7)]
    ax.bar([i - 1 for i in x], result[0], width = 1, label = labels[0], hatch='\\',color = color[0])
    ax.bar([i for i in x], result[1], width = 1, label = labels[1], hatch='-',color = color[1])
    ax.bar([i + 1 for i in x], result[2], width = 1, label = labels[2], hatch='|',color = color[2])
    ax.bar([i + 2 for i in x], result[3], width = 1, label = labels[3], hatch='-',color = color[3])
    ax.bar([i + 3 for i in x], result[4], width = 1, label = labels[4], hatch='\\',color = color[4])
    ax.bar([i + 4 for i in x], result[5], width = 1, label = labels[5], hatch='|',color = color[5])
    ax.bar([i + 5 for i in x], result[6], width = 1, label = labels[6], hatch='-',color = color[6])

    ax.set_xticks([i + 1 for i in x])
    ax.set_xlim([-2, x[-1] + 7])

    #ax.set_ylim([0, 4.5])

    ax.set_xticklabels(xlabels, fontsize = 15)
    ax.legend(bbox_to_anchor = [0.5, 1], loc = 'lower center', ncol = 6)
    plt.savefig("%s%s-predict-error.pdf" % (GRAPH_DIR, name), bbox_inches='tight')

    plt.close(fig)

def RegressionOnModels(res, res_hat, h):
    data = np.array(res_hat).transpose()
    reg_y = np.array([])
    reg_y_hat = np.array([])
    for i in range(TRAIN_NUM + INPUT_DIM + h - 1, len(data) - INTERVAL, INTERVAL):
        train_data = data[i - TRAIN_NUM - INPUT_DIM:i]
        x, _ = Model.GeneratePair(train_data, HORIZON, INPUT_DIM)
        y = res[i - TRAIN_NUM - h + HORIZON:i - h + 1].reshape((-1, 1))

        xx = Model.GetMatrix(x)

        params = Model.Training(xx, y)
        #print(params)

        test_data = data[i - INPUT_DIM + 1 - HORIZON:i + INTERVAL]
        x, _ = Model.GeneratePair(test_data, HORIZON, INPUT_DIM)
        y = res[i:i + INTERVAL].reshape((-1, 1))
        y_hat = Model.Testing(params, x)
        reg_y = np.append(reg_y, y.flatten())
        reg_y_hat = np.append(reg_y_hat, y_hat.flatten())

    return reg_y, reg_y_hat

def PlotCompareAvg(data_dict, name, xlabel, ylabel, color):
    fig_length = 12

    fig = plt.figure()
    fig.set_size_inches(fig_length, 2)
    ax = fig.add_subplot(111)
    loc = plticker.MaxNLocator(3) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.grid()

    ax.set_ylabel(ylabel, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=18, weight='bold')

    models = ["ar", 'noencoder-rnn', 'psrnn-h5']
    labels = ["AR", "RNN-AVG", "PSRNN-AVG", "AVG", "RNN-REG", "PSRNN-REG", "REG", "LAST", "CONSTANT"]
    horizons = ['60', '720', '1440', '2880', '4320', '7200', '10080']
    xlabels = ["1 Hour", "12 Hour", "1 Day", "2 Days", "3 Days", "5 Days", "1 Week"]

    result = []

    for i, m in enumerate(models):
        res = []
        for h in horizons:
            print(m, h)
            r = np.array([])
            r_hat = np.array([])
            for tp in data_dict:
                if tp.find(PROJECT) >= 0 and tp.find("/" + m + "/") >=0 and tp.find("horizon-" + h) >= 0:
                    print(tp)
                    r = np.append(r, data_dict[tp][0])
                    r_hat = np.append(r_hat, data_dict[tp][1])

            res.append(np.mean((r - r_hat) ** 2))

        result.append(res)

    avg_res = []
    rnn_avg_res = []
    psrnn_avg_res = []
    rnn_reg_res = []
    psrnn_reg_res = []
    reg_res = []
    last_res = []
    constant_res = []
    for h in horizons:
        res = []
        res_hat = []
        for m in models:
            print(m, h)
            r = np.array([])
            r_hat = np.array([])
            for tp in sorted(data_dict.keys()):
                if tp.find(PROJECT) >= 0 and tp.find("/" + m + "/") >=0 and tp.find("horizon-" + h) >= 0:
                    print(tp)
                    r = np.append(r, data_dict[tp][0])
                    r_hat = np.append(r_hat, data_dict[tp][1])

            res = r
            res_hat.append(r_hat)

        res_rnn = np.mean(np.array([res_hat[0], res_hat[1]]), axis = 0)
        res_psrnn = np.mean(np.array([res_hat[0], res_hat[2]]), axis = 0)
        res_avg = np.mean(np.array(res_hat), axis = 0)

        hour_h = int(h) // 60

        rnn_reg_y, rnn_reg_y_hat = RegressionOnModels(res, [res_hat[0], res_hat[1]], hour_h)
        psrnn_reg_y, psrnn_reg_y_hat = RegressionOnModels(res, [res_hat[0], res_hat[2]], hour_h)
        reg_y, reg_y_hat = RegressionOnModels(res, res_hat, hour_h)

        last_res.append(np.mean((res[hour_h:] - res[:-hour_h]) ** 2))
        constant_res.append(np.mean((res - np.mean(res)) ** 2))
        avg_res.append(np.mean((res - res_avg) ** 2))
        rnn_avg_res.append(np.mean((res - res_rnn) ** 2))
        psrnn_avg_res.append(np.mean((res - res_psrnn) ** 2))
        rnn_reg_res.append(np.mean((rnn_reg_y - rnn_reg_y_hat) ** 2))
        psrnn_reg_res.append(np.mean((psrnn_reg_y - psrnn_reg_y_hat) ** 2))
        reg_res.append(np.mean((reg_y - reg_y_hat) ** 2))
        print(h, len(res), len(reg_y))

    result = result[0:1]
    result.append(rnn_avg_res)
    result.append(psrnn_avg_res)
    result.append(avg_res)
    result.append(rnn_reg_res)
    result.append(psrnn_reg_res)
    result.append(reg_res)
    result.append(last_res)
    result.append(constant_res)

    x = [x * 10 for x in range(7)]
    ax.bar([i - 1 for i in x], result[0], width = 1, label = labels[0], hatch='\\',color = color[0])
    ax.bar([i for i in x], result[1], width = 1, label = labels[1], hatch='-',color = color[1])
    ax.bar([i + 1 for i in x], result[2], width = 1, label = labels[2], hatch='|',color = color[2])
    ax.bar([i + 2 for i in x], result[3], width = 1, label = labels[3], hatch='\\',color = color[3])
    ax.bar([i + 3 for i in x], result[4], width = 1, label = labels[4], hatch='-',color = color[4])
    ax.bar([i + 4 for i in x], result[5], width = 1, label = labels[5], hatch='|',color = color[5])
    ax.bar([i + 5 for i in x], result[6], width = 1, label = labels[6], hatch='\\',color = color[6])
    ax.bar([i + 6 for i in x], result[7], width = 1, label = labels[7], hatch='-',color = color[7])
    ax.bar([i + 7 for i in x], result[8], width = 1, label = labels[8], hatch='|',color = color[8])

    ax.set_xticks([i + 1 for i in x])
    ax.set_xlim([-2, x[-1] + 9])

    #ax.set_ylim([0, 4.5])

    ax.set_xticklabels(xlabels, fontsize = 15)
    ax.legend(bbox_to_anchor = [0.5, 1], loc = 'lower center', ncol = 5)
    plt.savefig("%s%s-regression-predict-error-avg.pdf" % (GRAPH_DIR, name), bbox_inches='tight')

    plt.close(fig)

def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                '%.1f' % height, size=10,
                ha='center', va='bottom')


def PlotPaperGraph(data_dict, name, xlabel, ylabel, color):
    fig_length = 20

    fig = plt.figure()
    fig.set_size_inches(fig_length, 2)
    ax = fig.add_subplot(111)
    loc = plticker.MaxNLocator(3) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.grid()

    ax.set_ylabel(ylabel, fontsize=18, weight='bold')
    ax.set_xlabel(xlabel, fontsize=18, weight='bold')

    models = ["ar", 'kr', 'arma', 'fnn', 'noencoder-rnn', 'psrnn-h5']
    labels = ["LR", "KR", "ARMA", "FNN", "RNN", "PSRNN", "ENSEMBLE (LR+RNN)", "HYBRID"]
    horizons = ['60', '720', '1440', '2880', '4320', '7200', '10080']
    xlabels = ["1 Hour", "12 Hour", "1 Day", "2 Days", "3 Days", "5 Days", "1 Week"]

    result = []
    hist_result = []

    for i, m in enumerate(models):
        res = []
        hist_res = []
        for h in horizons:
            print(m, h)
            r = np.array([])
            r_hat = np.array([])
            for tp in data_dict:
                if tp.find(PROJECT) >= 0 and tp.find("/" + m + "/") >=0 and tp.find("horizon-" + h +
                        "/") >= 0:
                    print(tp)
                    r = np.append(r, data_dict[tp][0])
                    r_hat = np.append(r_hat, data_dict[tp][1])

            res.append(np.mean((r - r_hat) ** 2))
            hist_res += (r - r_hat).tolist()

        result.append(res)
        hist_result.append(hist_res)

    avg_res = []
    rnn_avg_res = []
    psrnn_avg_res = []
    mixture_res = []
    hist_res = []
    for h in horizons:
        res = []
        res_hat = []
        for m in models:
            print(m, h)
            r = np.array([])
            r_hat = np.array([])
            for tp in sorted(data_dict.keys()):
                if tp.find(PROJECT) >= 0 and tp.find("/" + m + "/") >=0 and tp.find("horizon-" + h +
                        "/") >= 0:
                    #print(tp)
                    r = np.append(r, data_dict[tp][0])
                    r_hat = np.append(r_hat, data_dict[tp][1])

            res = r
            print(h, m, r_hat.shape)
            res_hat.append(r_hat)

        #res_rnn = np.mean(np.array([res_hat[0], res_hat[1]]), axis = 0)
        res_psrnn = np.mean(np.array([res_hat[0], res_hat[4]]), axis = 0)
        #res_avg = np.mean(np.array(res_hat), axis = 0)


        data_mixture = []
        for i in range(res_hat[1].shape[0]):
            if res_hat[1][i] > 7 and abs(res_hat[1][i] - res_psrnn[i]) > 1:
                data_mixture.append(res_hat[1][i])
            else:
                data_mixture.append(res_psrnn[i])
        data_mixture = np.array(data_mixture)

        hour_h = int(h) // 60

        #avg_res.append(np.mean((res - res_avg) ** 2))
        #rnn_avg_res.append(np.mean((res - res_rnn) ** 2))
        psrnn_avg_res.append(np.mean((res - res_psrnn) ** 2))
        mixture_res.append(np.mean((res - data_mixture) ** 2))
        hist_res += (res - res_psrnn).tolist()

    #result.append(rnn_avg_res)
    result.append(psrnn_avg_res)
    result.append(mixture_res)
    hist_result.append(hist_res)
    #result.append(avg_res)

    print(result)
    #result[1] = [0,0,0,0,0,0,0]

    x = [x * 9 for x in range(7)]
    ax.bar([i - 1 for i in x], result[0], width = 1, label = labels[0], hatch='\\',color = color[0])
    ax.bar([i for i in x], result[1], width = 1, label = labels[1], hatch='-',color = color[1])
    ax.bar([i + 1 for i in x], result[2], width = 1, label = labels[2], hatch='|',color = color[2])
    ax.bar([i + 2 for i in x], result[3], width = 1, label = labels[3], hatch='/',color = color[3])
    ax.bar([i + 3 for i in x], result[4], width = 1, label = labels[4], hatch='.',color = color[4])
    ax.bar([i + 4 for i in x], result[5], width = 1, label = labels[5], hatch='+',color = color[7])
    ax.bar([i + 5 for i in x], result[6], width = 1, label = labels[6], hatch='o',color = color[6])
    ax.bar([i + 6 for i in x], result[7], width = 1, label = labels[7], hatch='*',color = color[5])

    rects = ax.patches
    autolabel(ax, rects)

    ax.set_xticks([i + 1 for i in x])
    ax.set_xlim([-2, x[-1] + 8])

    if PROJECT == "tiramisu":
        ax.set_ylim([0, 5.4])
    if PROJECT == "oli":
        ax.set_ylim([0, 17])
    #if PROJECT == "admission-full":
    #    ax.set_ylim([0, 10])

    ax.set_xticklabels(xlabels, fontsize = 15)
    #ax.legend(bbox_to_anchor = [0.5, 1], loc = 'lower center', ncol = 8)
    plt.savefig("%s%s-predict-paper.pdf" % (GRAPH_DIR, name), bbox_inches='tight')

    plt.close(fig)

    WriteResult("%s%s-predict-paper.csv" % (GRAPH_DIR, name), xlabels, labels, result)

    PlotHistgram(name, hist_result)


def PlotHistgram(name, hist_result):
    indexes = [0, 4, 6]
    models = ['lr', 'rnn', 'ensemble']
    for i, index in enumerate(indexes):
        fig, ax = plt.subplots(figsize=(12,4))
        #if PROJECT == "admission":
        #    ax.set_ylim([0, 30000])
        if PROJECT == "oli":
            ax.set_ylim([0, 0.25])
        #ax.set_ylim([0, 0.7])
        hist_range = (-15, 15)
        ax.hist(hist_result[index], 20, hist_range, normed=1, facecolor='green', alpha=0.75)
        plt.savefig("%s%s-%s-histogram.pdf" % (GRAPH_DIR, name, models[i]), bbox_inches='tight')


def WriteResult(path, xlabels, labels, result):
    with open(path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + xlabels)
        for x in range(len(labels)):
            writer.writerow([labels[x]] + result[x])


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

    return (np.log(y + data_min), np.log(y_hat + data_min))

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

def Main(input_dir):
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)

    color = SetupMplParams()

    data_dict = GetDataDict(input_dir)

    PlotPaperGraph(data_dict, PROJECT, "Prediction Horizon", "MSE (log space)", color)

# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    """
    Generate MSE result plots

    Args:
        arg1 : the result dir
    """
    Main(sys.argv[1])
