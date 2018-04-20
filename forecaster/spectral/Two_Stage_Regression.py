import numpy as np
import math
from sklearn.kernel_approximation import RBFSampler
import scipy as sp
import Utilities

###############################################################################
# Project into RKHS
###############################################################################

def ridgeRegression(X, Y, lr):
    
    A = X.dot(Y.T)
    B = Y.dot(Y.T) + lr*np.eye(Y.dot(Y.T).shape[0])
    X, _, _, _ = np.linalg.lstsq(B.T, A.T)
    return X.T

def batchRidgeRegression(XY_T, Y, lr):
    A = XY_T
    B = Y.dot(Y.T) + lr*np.eye(Y.dot(Y.T).shape[0])
    Z, _, _, _ = np.linalg.lstsq(B.T, A.T)
    return Z.T

def doubleBatchRidgeRegression(XY_T, YY_T, lr):
    A = XY_T
    B = YY_T + lr*np.eye(YY_T.shape[0])
    Z, _, _, _ = np.linalg.lstsq(B.T, A.T)
    return Z.T

def batch_transform(data, dataFun, rbf_sampler, U, matrix_batch_size):
    nSvd = U.shape[0]
    nData = data.shape[1]
    
    X_rbf = np.zeros((nSvd, nData))
    for i in range(0,data.shape[1], matrix_batch_size):
        if i % (math.ceil(data.shape[1]/(matrix_batch_size*10))*matrix_batch_size) == 0:
            print(i,'/',nData)
        end = min(i+matrix_batch_size, nData)
        y = dataFun(data[:,i:end])
        X_rbf[:,i:end] = U.dot(rbf_sampler.transform(y.T).T)
    return X_rbf

def batch_svd(data, dataFun, rbf_sampler, args):
    C = np.zeros((args.nRFF,args.nRFF))
    for i in range(0, data.shape[1], args.matrix_batch_size):
        if i % (math.ceil(data.shape[1]/(args.matrix_batch_size*10))*args.matrix_batch_size) == 0:
            print(i,'/', data.shape[1])
        end = min(i+args.matrix_batch_size, data.shape[1])
        x = dataFun(data[:,i:end])
        x = rbf_sampler.transform(x.T).T
        C += x.dot(x.T)
    U, S, V = sp.sparse.linalg.svds(C, args.nhid)
    return U.T

#def two_stage_regression2(corpus, batch_size, kernel_width, seed, nRFF, nSvd, reg_rate):
    #onehot(x[i:end],[input_dim,end-i])
    
def two_stage_regression(data, obsFun, pastFun, futureFun, shiftedFutureFun, outputFun, args):
    
    # create RBF projection
    obs_rbf_sampler = RBFSampler(gamma=args.kernel_width, random_state=args.seed*5, n_components=args.nRFF)
    obs_rbf_sampler.fit(np.zeros((1, obsFun(data[:,0].reshape(-1,1)).shape[0])))
    past_rbf_sampler = RBFSampler(gamma=args.kernel_width, random_state=args.seed*7, n_components=args.nRFF)
    past_rbf_sampler.fit(np.zeros((1, pastFun(data[:,0].reshape(-1,1)).shape[0])))
    future_rbf_sampler = RBFSampler(gamma=args.kernel_width, random_state=args.seed*9, n_components=args.nRFF)
    future_rbf_sampler.fit(np.zeros((1, futureFun(data[:,0].reshape(-1,1)).shape[0])))
    
    # Calculate linear projection of RFF
    U_obs = batch_svd(data, obsFun, obs_rbf_sampler, args)
    U_past = batch_svd(data, pastFun, past_rbf_sampler, args)
    U_future = batch_svd(data, futureFun, future_rbf_sampler, args)

    # Project data using RBF then U
    Obs_U = batch_transform(data, obsFun, obs_rbf_sampler, U_obs, args.matrix_batch_size)
    P_U = batch_transform(data, pastFun, past_rbf_sampler, U_past, args.matrix_batch_size)
    F_U = batch_transform(data, futureFun, future_rbf_sampler, U_future, args.matrix_batch_size)
    FS_U = batch_transform(data, shiftedFutureFun, future_rbf_sampler, U_future, args.matrix_batch_size)

    data = data

    
    # stage 1 regression
    W_F_P = ridgeRegression(F_U, P_U, args.reg_rate)
    FE_P = np.zeros((F_U.shape[0]*Obs_U.shape[0], F_U.shape[0]))
    for i in range(0,F_U.shape[1], args.matrix_batch_size):
            if i % (math.ceil(F_U.shape[1]/(args.matrix_batch_size*10))*args.matrix_batch_size) == 0:
                print(i,'/',F_U.shape[1])
                
            end = min(i+args.matrix_batch_size, F_U.shape[1])
            FE_U_batch = Utilities.flat_prod(FS_U[:,i:end],Obs_U[:,i:end])
            P_U_batch = P_U[:,i:end]
            FE_P += FE_U_batch.dot(P_U_batch.T)
    W_FE_P = batchRidgeRegression(FE_P, P_U, args.reg_rate)      
    
    # apply stage 1 regression to data to generate input for stage2 regression
    E_F = W_F_P.dot(P_U)
    E_FE_F = np.zeros((W_FE_P.shape[0], F_U.shape[0]))
    for i in range(0,F_U.shape[1], args.matrix_batch_size):
        if i % (math.ceil(F_U.shape[1]/(args.matrix_batch_size*10))*args.matrix_batch_size) == 0:
            print(i,'/',F_U.shape[1])
            
        end = min(i+args.matrix_batch_size, F_U.shape[1])
        E_FE_batch = W_FE_P.dot(P_U[:,i:end])
        E_F_batch = W_F_P.dot(P_U[:,i:end])
        E_FE_F += E_FE_batch.dot(E_F_batch.T)
    
    # stage 2 regression
    W_FE_F = batchRidgeRegression(E_FE_F, E_F, args.reg_rate)
    
    # calculate initial state
    x_1 = np.mean(F_U,1).reshape(1,-1)
    
    # regress from state to predictions
    F_FU = np.zeros((outputFun(data[:,0].reshape(-1,1)).shape[0], args.nhid))
    for i in range(0, data.shape[1], args.matrix_batch_size):
        if i % (math.ceil(data.shape[1]/(args.matrix_batch_size*10))*args.matrix_batch_size) == 0:
            print(i,'/',data.shape[1])
            
        end = min(i+args.matrix_batch_size, data.shape[1])
        output_batch = outputFun(data[:,i:end])
        FU_batch = F_U[:,i:end]
        F_FU += output_batch.dot(FU_batch.T)
    
    W_pred = batchRidgeRegression(F_FU, F_U, args.reg_rate)
    
    return obs_rbf_sampler, U_obs, W_FE_F, np.zeros(W_FE_F.shape[0]), W_pred, np.zeros(W_pred.shape[0]), x_1
