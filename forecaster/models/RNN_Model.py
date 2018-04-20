import torch.nn as nn
import math
from torch.autograd import Variable

     
class RNN_Model(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNN_Model, self).__init__()
        self.encoder = nn.Linear(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout = dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers

        self.init_weights()


    def init_weights(self):
        initrange = 1
        encoder = self.encoder
        self.encoder.bias.data.fill_(0)
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        encoder.weight.data.normal_(0, math.sqrt(2. / (encoder.in_features + encoder.out_features)))

        decoder = self.decoder
        self.decoder.bias.data.fill_(0)
        #self.decoder.weight.data.uniform_(-initrange, initrange)
        decoder.weight.data.normal_(0, math.sqrt(2. / (decoder.in_features + decoder.out_features)))
        
        self.rnn.weight_ih_l0.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.weight_ih_l0.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)

        self.rnn.weight_ih_l1.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.weight_ih_l1.data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
        self.rnn.bias_ih_l1.data.fill_(0)
        self.rnn.bias_hh_l1.data.fill_(0)



    def forward(self, input, hidden):
        
        bptt = input.size(0)
        bsz = input.size(1)
        
        input = input.view(bptt*bsz, -1)
        emb = self.encoder(input)
        emb = emb.view(bptt, bsz, -1)
        
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
     
