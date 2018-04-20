import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as Functional

class PSRNN_Model(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False,
            cuda=False):
        super(PSRNN_Model, self).__init__()
        
        self.RBF = nn.Linear(ntoken, ninp)
        self.embedding = nn.Linear(ninp, nhid)
        self.W_FE_F = nn.Linear(nhid, nhid*nhid)
        self.decoder = nn.Linear(nhid, ntoken)
        
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        
        self.x_1 = np.ones((1,self.nhid))
        
        self.init_weights_random()

        self.use_cuda = cuda
        
    def init_weights_random(self):
        initrange = 0.1
        self.RBF.weight.data.uniform_(-initrange, initrange)
        self.RBF.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.fill_(0)
        self.W_FE_F.weight.data.uniform_(-initrange, initrange)
        self.W_FE_F.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        
    def init_weights_psr(self, RBF_weight, RBF_bias, embedding_weight, W_FE_F_weight, W_FE_F_bias,
                         decoder_weight, decoder_bias, x_1):
        self.RBF.weight.data = torch.Tensor(RBF_weight)
        self.RBF.bias.data = torch.Tensor(RBF_bias)
        self.embedding.weight.data = torch.Tensor(embedding_weight)
        self.embedding.bias.data.fill_(0)
        self.W_FE_F.weight.data = torch.Tensor(W_FE_F_weight)
        self.W_FE_F.bias.data = torch.Tensor(W_FE_F_bias)
        self.decoder.weight.data = torch.Tensor(decoder_weight)
        self.decoder.bias.data = torch.Tensor(decoder_bias)
        self.x_1 = x_1

    def init_hidden(self, bsz):
        hidden = Variable(torch.Tensor(np.ones((bsz, 1)).dot(self.x_1)))
        if self.use_cuda:
            hidden = hidden.cuda()

        return hidden
        
    def forward(self, input, b):
        
        bptt = input.size(0)
        bsz = input.size(1)
        
        # encode observation
        input = input.view(bptt*bsz, input.size(2))
        encoded = self.RBF(input).cos()*np.sqrt(2.)/np.sqrt(self.ninp)
        obs = self.embedding(encoded)
        obs = obs.view(bptt, bsz, self.nhid)
        
        # update state
        output = [0]*bptt
        for i in range(bptt):
            W = self.W_FE_F(b)
            W = W.view(self.nhid*bsz, self.nhid)
            b = [0]*bsz
            for j in range(bsz):
                obs_ij = obs[i,j,:].view(1,self.nhid)
                b[j] = W[j*self.nhid:(j+1)*self.nhid,:].mm(obs_ij.t()).t()
                ones = Variable(torch.ones(1,b[j].size()[1]))
                if self.use_cuda:
                    ones = ones.cuda()
                b[j] = b[j].div((b[j].mm(b[j].t())).sqrt().mm(ones))
            b = torch.cat(b)
            output[i] = self.decoder(b).view(1,b.size(0),-1)
        output = torch.cat(output)
        

        return output, b
