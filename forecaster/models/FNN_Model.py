import torch.nn as nn
import math
from torch.autograd import Variable

     
class FNN_Model(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, regress_dim, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(FNN_Model, self).__init__()

        nlayers = 1
        ninp = 10
        self.encoder = nn.Linear(ntoken * regress_dim, ninp)

        # First layer
        self.layers = [nn.Linear(ninp, nhid)]
        for i in range(nlayers - 1):
            layer = nn.Linear(nhid, nhid)
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 1

        encoder = self.encoder
        encoder.bias.data.fill_(0)
        encoder.weight.data.normal_(0, math.sqrt(2. / (encoder.in_features + encoder.out_features)))
        #encoder.weight.data.uniform_(-initrange, initrange)

        for layer in self.layers:
            layer.bias.data.fill_(0)
            layer.weight.data.normal_(0, math.sqrt(2. / (layer.in_features + layer.out_features)))
            #layer.weight.data.uniform_(-initrange, initrange)

        decoder = self.decoder
        decoder.bias.data.fill_(0)
        decoder.weight.data.normal_(0, math.sqrt(2. / (decoder.in_features + decoder.out_features)))
        #decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        
        bptt = input.size(0)
        
        emb = self.encoder(input)

        output = emb
        for layer in self.layers:
            output = self.relu(layer(output))
            output = self.dropout(output)
        
        decoded = self.decoder(output)
        return decoded.view(1, bptt, -1), hidden

    def init_hidden(self, bsz):
        return None
     
