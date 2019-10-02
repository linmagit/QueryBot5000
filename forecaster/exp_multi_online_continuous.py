#!/usr/bin/env python3.5

import importlib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init
import numpy as np
import math
import time
import os
import csv
import argparse
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io
import shutil
import pickle

import statsmodels.api as sm

from datetime import datetime, timedelta

from sortedcontainers import SortedDict

import Utilities

from spectral import Two_Stage_Regression

from models import FNN_Model
from models import RNN_Model
from models import PSRNN_Model

# ==============================================
# PROJECT CONFIGURATIONS
# ==============================================

PROJECTS = {
    "tiramisu": {
        "name": "tiramisu",
        "input_dir": "../data/online-tiramisu-clusters",
        "cluster_path": "../data/cluster-coverage/tiramisu-coverage.pickle",
        "output_dir":
        #"../result-interval-new/online-prediction/online-tiramisu-clusters-prediction",
        "../result/online-prediction/online-tiramisu-clusters-prediction",
    },
    "admission": {
        "name": "admission",
        #"input_dir": "../data/online-admission-clusters",
        "input_dir":
        "../../peloton-tf/time-series-clustering/online-admission-full-clusters/",
        "cluster_path":
        "../../peloton-tf/time-series-clustering/admission-full-coverage/coverage.pickle",
        #"cluster_path": "../data/cluster-coverage/admission-coverage.pickle",
        "output_dir":
        #"../tmp/",
        #"../result-admission-full/",
        #"../result-interval/online-prediction/online-admission-clusters-prediction",
        #"../result/online-prediction/online-admission-clusters-prediction",
        "../result/online-prediction/online-admission-full-clusters-prediction",
    },
    "oli": {
        "name": "oli",
        "input_dir": "../data/online-oli-clusters",
        "cluster_path": "../data/cluster-coverage/oli-coverage.pickle",
        "output_dir":
        "../result/online-prediction/online-oli-clusters-prediction",
    },
    "noise": {
        "name": "noise",
        "input_dir":
        "../../peloton-tf/time-series-clustering/online-admission-noise-clusters/",
        "cluster_path":
        "../../peloton-tf/time-series-clustering/admission-noise-coverage/coverage.pickle",
        "output_dir":
        "../result-admission-noise/",
    },
}



# %%
class Args:
    def __init__(self):
        # RNN cell type
        self.model = 'LSTM'
        # Size of each RNN layer
        self.nhid = 20
        # Numbe of RNN layers
        self.nlayers = 2
        # Learning rate
        self.lr = 1
        self.clip = 0.25
        # RNN learning epochs for each workload
        self.epochs = {'tiramisu': 300, "oli": 30, "admission": 30, "noise": 50}
        # Adjust BPTT size accordingly with interval size
        self.bptt = {1: 240, 5:200, 10:120, 20:90, 30:60, 60:48, 120:30}
        self.dropout = 0.2
        self.tied = False
        self.cuda = False
        self.log_interval = 50
        self.save = 'model.pt'
        
        # PSRNN params
        self.kernel_width = 0.02
        self.nRFF = 25
        self.matrix_batch_size = 5000
        self.reg_rate = 0.001
        self.max_lines = 10**3

        # added by Lin
        # Adjust batch size accordingly with interval size to save training time
        self.batch_size = {1: 30, 5:20, 10:15, 20:12, 30:8, 60:4, 120:2}
        # Prediction horizon size
        self.horizon = 60
        # The training start position
        self.start_pos = 14400 
        # Update/Retrain the models every day
        self.interval = 1440

        # Always use the past day arrival rates for the regression. Adjusted
        # according to the interval size.
        self.regress_dim = {1: 1440, 5:288, 10: 144, 20:72, 30:48, 60:24, 120:12}
        # Auto regression order for the ARMA model
        self.ar_order = 6
        # Moving agerage order for the ARMA model
        self.ma_order = 3
        # Moving agerage dimension for the ARMA model
        self.ma_dim = 24
        # Aggregated prediction interval size
        self.aggregate = 1
        # warm up the prediction of RNN using the latest observed data (days)
        self.paddling_intervals = 7
        # training data size (days)
        self.training_intervals = 25
        # Number of clusters to train together
        self.top_cluster_num = 3

# %%
# AR and ARMA model
class LinearModel:
    def __init__(self):
        self.params = None
        self.ma_params = None

# %%
# Kernel Regression (KR) model
class KernelRegressionModel:
    def __init__(self):
        self.data = None

# split the data into batches
def batchify(data, bsz):
    # Dimension of the observation
    nobserve = data.shape[1]
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.shape[0] // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    data = data.reshape(bsz, -1, nobserve)

    # Transpose the data to fit the model input
    data = data.transpose(1,0,2)
    
    # data.shape = (sequense length, batch size, dim of the observation)
    return data


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

            for line in reader:
                count = float(line[1])
                ts = datetime.strptime(line[0], datetime_format)
                hour = ts.hour
                if aggregate > 60:
                    hour //= aggregate // 60
                time_stamp = datetime(ts.year, ts.month, ts.day, hour, ts.minute - (ts.minute %
                    aggregate), 0)

                if not time_stamp in trajs[cluster]:
                    trajs[cluster][time_stamp] = 0

                trajs[cluster][time_stamp] += count

    return trajs


NTOKENS = 3
PSRHORIZON = 5
def obsFun(x): 
    return x[NTOKENS * PSRHORIZON:NTOKENS * (PSRHORIZON + 1),:]

def pastFun(x): 
    return x[:NTOKENS * PSRHORIZON,:]
    
def futureFun(x): 
    return x[NTOKENS * PSRHORIZON :NTOKENS * PSRHORIZON * 2,:]

def shiftedFutureFun(x): 
    return x[NTOKENS * (PSRHORIZON + 1):NTOKENS * (PSRHORIZON * 2 + 1),:]

def outputFun(x): 
    return x[NTOKENS * PSRHORIZON:NTOKENS * (PSRHORIZON + 1),:]

def GeneratePair(data, horizon, input_dim):
    n = data.shape[0]
    m = data.shape[1]

    x = []
    y = []

    for i in range(n - horizon - input_dim + 1):
        x.append(data[i:i + input_dim].flatten())
        y.append(data[i + input_dim + horizon - 1])

    return (np.array(x), np.array(y))

def GetMatrix(x):
    xx = x.T.dot(x)
    xx += np.identity(xx.shape[0])
    return np.linalg.inv(xx).dot(x.T)

def Training(x, y):
    params = []
    for j in range(y.shape[1]):
        params.append(x.dot(y[:, j]))

    return params

def Testing(params, x):
    y_hat = None

    for j in range(len(params)):
        y = x.dot(params[j])

        y = y.reshape((-1, 1))

        if y_hat is None:
            y_hat = y
        else:
            y_hat = np.concatenate((y_hat, y), axis = 1)

    return y_hat

# %% TRAIN THE MODEL

def train_pass(args, method, model, data, criterion, lr, bptt, clip, log_interval):

    if method == 'arma':
        # Train the ARMA model
        order = (args.ar_order, args.ma_order)
        arma_model = sm.tsa.VARMAX(data, order=order, enforce_stationarity=False)
        arma_model = arma_model.fit(maxiter = 10)
        arma_data = arma_model.predict()

        # Combine ARMA output with the original data
        x, y = GeneratePair(data, args.horizon, args.regress_dim)
        arma_x, arma_y = GeneratePair(arma_data, args.horizon, args.regress_dim)
        x = np.concatenate((x, arma_x), axis=1)
        xx = GetMatrix(x)

        # Train the model again using the ARMA output
        model.params = Training(xx, y)

        return


    if method == 'ar' or method == 'arma':
        x, y = GeneratePair(data, args.horizon, args.regress_dim)

        xx = GetMatrix(x)

        model.params = Training(xx, y)

        # This is the old ARMA two-stage regression implementation by myself
        if method == 'arma':
            y_hat = Testing(model.params, x)

            residual = y - y_hat
            x, y = GeneratePair(residual, args.horizon, args.ma_dim)

            if x.shape[0] > 0:
                xx = GetMatrix(x)
                model.ma_params = Training(xx, y)

        return

    if method == 'kr':
        x, y = GeneratePair(data, args.horizon, args.regress_dim)
        model.data = (x, y)
        return

    # Turn on training mode which enables dropout.
    model.train()
        
    total_loss = 0
    losses = []

    if method == 'fnn':
        x, y = GeneratePair(data, args.horizon, args.regress_dim)

        ndata = x.shape[0]
        bptt = ndata // args.batch_size + 1
        horizon = 0
        nbatch = 0
    else:
        batch_size = max(1, min(args.batch_size, len(data) // (args.horizon + args.bptt)))
        data = batchify(data, batch_size)
        ndata = data.shape[0]
        nbatch = data.shape[1]
        horizon = args.horizon
    
    hidden = model.init_hidden(nbatch)
    for batch, i in enumerate(range(0, ndata - horizon, bptt)):
        if method == "fnn":
            data_batch = x[i:i + bptt]
            targets = y[i: i + bptt]
            targets.reshape((1, targets.shape[0], -1))
        else:
            data_batch, targets = Utilities.get_batch(data, i, bptt, False, args.horizon)

        input = Variable(torch.Tensor(data_batch.astype(float)))
        targets = Variable(torch.Tensor(targets.astype(float)))

        if hidden is not None:
            hidden = Utilities.repackage_hidden(hidden)
        model.zero_grad()

        if args.cuda:
            input = input.cuda()
            targets = targets.cuda()
        output, hidden = model(input, hidden)
        
        # Calculations for loss
        loss = criterion(output, targets)
        if args.cuda:
            loss = loss.cpu()
        total_loss += loss.data.numpy()
        losses.append(loss.data.numpy())

        # Perform Gradient Descent
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        # Print Training Accuracy
        if batch % log_interval == 0:
            
            total_loss /= log_interval
            
            Utilities.prettyPrint(' lr: ' + str(lr) + ' batches: '
                        + str(batch) + '/'+str(ndata // bptt),
                        total_loss)       
            
            total_loss = 0

    Utilities.prettyPrint('Average Train Loss: ', np.mean(losses))


def evaluate_pass(args, method, model, data, criterion, bptt):

    if method == 'arma':
        # Train the ARMA model
        order = (args.ar_order, args.ma_order)
        arma_model = sm.tsa.VARMAX(data, order=order, enforce_stationarity=False)
        arma_model = arma_model.fit(maxiter = 10)
        arma_data = arma_model.predict()

        # Combine ARMA output with the original data
        x, y = GeneratePair(data, args.horizon, args.regress_dim)
        arma_x, arma_y = GeneratePair(arma_data, args.horizon, args.regress_dim)
        x = np.concatenate((x, arma_x), axis=1)

        # Test the model using the combined data
        y_hat = Testing(model.params, x)

        return np.mean((y - y_hat) ** 2), y, y_hat


    if method == 'ar' or method == 'arma':
        x, y = GeneratePair(data, args.horizon, args.regress_dim)
        y_hat = Testing(model.params, x)

        if method == 'arma':
            residual = y - y_hat

            r_x, r_y = GeneratePair(residual, args.horizon, args.ma_dim)

            if r_x.shape[0] > 0 and model.ma_params is not None:
                r_y_hat = Testing(model.ma_params, r_x)

                r_n = r_y_hat.shape[0]

                y = y[-r_n:]
                y_hat = y_hat[-r_n:]

                y_hat += r_y_hat

        return np.mean((y - y_hat) ** 2), y, y_hat

    if method == 'kr':
        x, y = GeneratePair(data, args.horizon, args.regress_dim)
        k_x, k_y = model.data

        pairwise_sq_dists = sp.spatial.distance.cdist(x, k_x, 'seuclidean')
        kernel = sp.exp(-pairwise_sq_dists)
        y_hat = kernel.dot(k_y) / np.sum(kernel, axis = 1, keepdims = True)

        return np.mean((y - y_hat) ** 2), y, y_hat


    model.eval()
        
    total_loss = 0

    if method == 'fnn':
        x_data, y_data = GeneratePair(data, args.horizon, args.regress_dim)

        ndata = x_data.shape[0]
        bptt = ndata
        horizon = 0
        nbatch = 0
    else:
        # Because the prediction must be continuous, we cannot use batch here
        data = batchify(data, 1)
        #data = batchify(data, args.batch_size)
        ndata = data.shape[0]
        nbatch = data.shape[1]
        horizon = args.horizon

    hidden = model.init_hidden(nbatch)
    y = None
    y_hat = None
    batch = 0

    for batch, i in enumerate(range(0, ndata - args.horizon, bptt)):
        if method == "fnn":
            data_batch = x_data[i:i + bptt]
            targets = y_data[i: i + bptt]
            targets = targets.reshape((1, targets.shape[0], -1))
        else:
            data_batch, targets = Utilities.get_batch(data, i, bptt, False, args.horizon)

        input = Variable(torch.Tensor(data_batch.astype(float)))
        targets = Variable(torch.Tensor(targets.astype(float)))

        if args.cuda:
            input = input.cuda()
            targets = targets.cuda()

        if hidden is not None:
            hidden = Utilities.repackage_hidden(hidden)
        output, hidden = model(input, hidden)
        
        # Calculations for loss
        loss = criterion(output, targets)
    
        if args.cuda:
            loss = loss.cpu()
            targets = targets.cpu()
            output = output.cpu()

        total_loss += loss.data.numpy()
        
        if y is None:
            y = targets.data.numpy()
        else:
            y = np.concatenate((y, targets.data.numpy()))

        if y_hat is None:
            y_hat = output.data.numpy()
        else:
            y_hat = np.concatenate((y_hat, output.data.numpy()))

    # transpose the output back to normal order
    y = y.transpose(1, 0, 2).reshape((-1, y.shape[2]))
    y_hat = y_hat.transpose(1, 0, 2).reshape((-1, y_hat.shape[2]))

    return (total_loss / (batch+1)), y, y_hat


def GetModel(args, data, method):
    train = batchify(data, args.batch_size)

    ntokens = train.shape[2]
    bsz = train.shape[1]

    #%% BUILD THE MODEL
    if method == "fnn":
        model = FNN_Model.FNN_Model(args.model, ntokens, args.regress_dim, args.nRFF, args.nhid,
                args.nlayers, args.dropout, args.tied)

    if method == "rnn":
        model = RNN_Model.RNN_Model(args.model, ntokens, args.nRFF, args.nhid, args.nlayers, args.dropout, args.tied)

    if method == "psrnn":
        model = PSRNN_Model.PSRNN_Model(args.model, ntokens, args.nRFF, args.nhid, args.nlayers,
                args.dropout, args.tied, args.cuda)

        # %% TWO STAGE REGRESSION
        data_len = 2 * PSRHORIZON + 1
        stacked_data = np.empty((ntokens * data_len,0))
        for i in range(bsz):
            data_list = []
            train_len = len(train)
            for j in range(data_len):
                data_list.append(train[j:train_len - data_len + j + 1, i, :])
            batch = np.hstack(tuple(data_list))
            stacked_data = np.concatenate([stacked_data, batch.T],1)

        rbf_sampler, U_obs, W_FE_F_weight, W_FE_F_bias, W_pred_weight, W_pred_bias, x_1 = \
        Two_Stage_Regression.two_stage_regression(stacked_data, obsFun, pastFun, futureFun, shiftedFutureFun, outputFun, args)

        # Initialize with PSR
        model.init_weights_psr(rbf_sampler.random_weights_.T,
                               rbf_sampler.random_offset_, 
                               U_obs,
                               W_FE_F_weight,
                               W_FE_F_bias, 
                               W_pred_weight,
                               W_pred_bias,
                               x_1)

    if method == "ar" or method == 'arma':
        return LinearModel()

    if method == "kr":
        return KernelRegressionModel()


    #model = RNN_RBF_Model(args.model, ntokens, args.nRFF, args.nhid, args.nlayers, args.dropout, args.tied)
    #model = PSRNN_Nonlinear_Model_backup.PSRNN_Nonlinear_Model_backup(args.model, ntokens, args.nRFF, args.nhid, args.nlayers, args.dropout, args.tied)
    #model = PSRNN_Factorized_Model.PSRNN_Factorized_Model(args.model, ntokens, args.nRFF, \
    #                                                      args.nhid, args.nlayers, args.dropout, args.nhid*30)

    if args.cuda:
        model.cuda()

    return model

def GetMultiData(trajs, clusters, date, num_days, interval, num_mins, aggregate):
    date_list = [date - timedelta(minutes = x) for x in range(num_days * interval * aggregate,
        -num_mins, -aggregate)]

    traj = []

    for date in date_list:
        obs = []
        for c in clusters:
            if c in trajs:
                data_date = next(trajs[c].irange(maximum = date, inclusive = (True, False), reverse =
                    True), None)
            else:
                data_date = None
                print("cluster %d is not in trajs!!!", c)

            if data_date is None:
                data_point = 0
            else:
                data_point = trajs[c][data_date]
            obs.append(data_point)

        traj.append(obs)

    traj = np.array(traj)

    return traj

def Normalize(data):
    # normalizing data
    data_min = 1 - np.min(data)
    data = np.log(data + data_min)
    data_mean = np.mean(data)
    data -= data_mean
    data_std = np.std(data)
    data /= data_std

    return data, data_min, data_mean, data_std

def Predict(args, config, top_cluster, trajs, method):

    for date, cluster_list in top_cluster[args.start_pos // args.interval:- max(args.horizon //
            args.interval, 1)]:
        # Training delta
        if config['name'] == "admission":
            first_date = datetime(2016, 9, 18)
        elif config['name'] == "noise":
            first_date = datetime(2018, 1, 28, 0, 58)
        else:
            first_date = top_cluster[0][0]
        train_delta_intervals = min(((date - first_date).days * 1440 + (date - first_date).seconds // 60
            ) // (args.aggregate * args.interval), args.training_intervals)
        #print(train_delta_intervals)
        #print(date, first_date)
        # Predict delta
        predict_delta_mins = args.horizon * args.aggregate

        print(date, first_date, date + timedelta(minutes = predict_delta_mins))

        clusters = next(zip(*cluster_list))[:args.top_cluster_num]

        data = GetMultiData(trajs, clusters, date, train_delta_intervals, args.interval, predict_delta_mins, args.aggregate)


        data, data_min, data_mean, data_std = Normalize(data)

        #print(data)
        print(data.shape)
        #print(args.interval, args.horizon)
        train_data = data[:-args.interval - args.horizon]
        print(train_data.shape)
        test_data = data[-(args.paddling_intervals * args.interval + args.horizon + args.interval):]
        print(test_data.shape)

        model = GetModel(args, train_data, method)

        criterion = nn.MSELoss()

        # Loop over epochs.
        for epoch in range(1, args.epochs + 1):
            print('epoch: ', epoch)
            epoch_start_time = time.time()
            lr = args.lr
            if epoch > 100:
                lr = 0.2
            train_pass(args, method, model, train_data, criterion, lr, args.bptt, args.clip, args.log_interval)
            print('about to evaluate: ')
            val_loss, y, y_hat, = evaluate_pass(args, method, model, test_data, criterion, args.bptt)
            Utilities.prettyPrint('Validation Loss: Epoch'+str(epoch), np.mean((y[-args.interval:] -
                y_hat[-args.interval:]) ** 2))

        # Run on test data.
        print('about to test')
        test_loss, y, y_hat= evaluate_pass(args, method, model, test_data, criterion, args.bptt)

        y = y[-args.interval:]
        y_hat = y_hat[-args.interval:]
        Utilities.prettyPrint('Test Loss', np.mean((y - y_hat) ** 2))
        Utilities.prettyPrint('Test Data Variance', np.mean(y ** 2))

        y = np.exp(y * data_std + data_mean) - data_min
        y_hat = np.exp(y_hat * data_std + data_mean) - data_min

        predict_dates = [date + timedelta(minutes = args.horizon * args.aggregate - x) for x in
                range(args.interval * args.aggregate, 0, -args.aggregate)]
        for i, c in enumerate(clusters):
            WriteResult(config['output_dir'] + str(c) + ".csv", predict_dates, y[:, i], y_hat[:, i])

        #WriteResult(config['output_dir'] + "total.csv", predict_dates, np.sum(y, axis = 1),
        #        np.sum(y_hat, axis = 1))


def WriteResult(path, dates, actual, predict):
    with open(path, "a") as csvfile:
        writer = csv.writer(csvfile, quoting = csv.QUOTE_ALL)
        for x in range(len(dates)):
            writer.writerow([dates[x], actual[x], predict[x]])

def Main(config, method, aggregate, horizon, input_dir, output_dir, cluster_path):
    args = Args()
    args.epochs = args.epochs[config["name"]]

    if method == 'ar' or method == "arma" or method == "kr":
        args.epochs = 1

    if config['name'] == "admission":
        args.start_pos = 1440 * 50
        if method == 'kr':
            args.training_intervals = 10000
            args.regress_dim[aggregate] = 480
        args.paddling_intervals = 30
        if method == 'rnn':
            args.epochs = 200
        
    if config['name'] == "noise":
        args.interval = 60
        args.start_pos = 120
        args.training_intervals = 3
        args.paddling_intervals = 1
        args.regress_dim[aggregate] = 5


    global NTOKENS
    NTOKENS = args.top_cluster_num

    args.horizon = horizon

    args.aggregate = aggregate
    args.horizon //= aggregate
    if args.horizon == 0:
        args.horizon = 1
    args.start_pos //= aggregate
    args.interval //= aggregate

    args.bptt = args.bptt[aggregate]
    args.batch_size = args.batch_size[aggregate]
    args.regress_dim = args.regress_dim[aggregate]

    input_dir = input_dir or config['input_dir']
    output_dir = output_dir or config['output_dir']
    cluster_path = cluster_path or config['cluster_path']

    trajs = LoadData(input_dir, aggregate)

    with open(cluster_path, 'rb') as f:
        top_cluster, _ = pickle.load(f)

    method_name = method
    if method == "rnn":
        method_name = "noencoder-rnn"
    if method == "psrnn":
        method_name = "psrnn-h5"

    output_dir += "/agg-%s/horizon-%s/%s/" % (aggregate, horizon, method_name)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    config['output_dir'] = output_dir
    Predict(args, config, top_cluster, trajs, method)


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='Online timeseries prediction')
    aparser.add_argument('project', choices=PROJECTS.keys(), help='Data source type')
    aparser.add_argument('--method', help='Input Data Directory')
    aparser.add_argument('--aggregate', default=1, type=int, help='Aggregate the results by how many minutes')
    aparser.add_argument('--horizon', default=60, type=int, help='How far do we predict ahead'
            '(minutes)')
    aparser.add_argument('--input_dir', help='Input directory')
    aparser.add_argument('--output_dir', help='Output directory')
    aparser.add_argument('--cluster_path', help='Path of the clustering assignment')
    args = vars(aparser.parse_args())

    Main(PROJECTS[args['project']], args['method'], args['aggregate'], args['horizon'],
            args['input_dir'], args['output_dir'], args['cluster_path'])

