import numpy as np
from torch.autograd import Variable
import math
import torch
import matplotlib.pyplot as plt 
import os

def onehot(X, dim):
    Xind = np.zeros(dim)
    Xind[X, np.arange(dim[1])] = 1
    return Xind

def flat_prod(X,Y):
    XY = np.zeros((X.shape[0]*Y.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        XY[:,i] = np.kron(X[:,i], Y[:,i].T).reshape(X.shape[0]*Y.shape[0])
    return XY

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()

def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, source.shape[0] - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target

def get_batch(source, i, bptt, evaluation=False, horizon=1):
    seq_len = min(bptt, source.shape[0] - horizon - i)
    data = source[i:i+seq_len]
    target = source[i+horizon:i+horizon+seq_len]
    return data, target

def prettyPrint(description, loss):
    print('=' * 89)
    print('|| ',description, ' || loss {:5.3f}'.format(loss))
    print('=' * 89)

def my_plot(x_tst, y, i_plt,j_plt):
    plt.plot(x_tst[:,i_plt,j_plt])
    plt.plot(y[:,i_plt,j_plt])
    plt.show()

def save_plot(x_tst, y, i_plt):
    x_tst = x_tst.transpose(1, 0, 2)
    y = y.transpose(1, 0, 2)
    plt.figure(figsize = (120, 2.5))
    plt.plot(x_tst[:, :, i_plt].flatten(), linewidth = 0.5)
    plt.plot(y[:, :, i_plt].flatten(), linewidth = 0.5)
    #plt.ylim([0, 8000])
    plot_dir = "../plot/regressed-admission-psrnn-lr1-log"
    #plot_dir = "../plot/regressed-admission-rnn-lr1-log"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig("%s/%d.pdf" % (plot_dir, i_plt))
    plt.close()
    
def plot_weights(W):
    plt.set_cmap('jet')
    plt.imshow(W)
    plt.show()



