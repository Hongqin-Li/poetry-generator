import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from custom_lstm import LSTM

class RecurrentNetwork(nn.Module):

    def __init__(self, vocab_size, target_size, embedding_dim, hidden_size, num_layers=1, batch_first=False, mode='LSTM', bidirectional=False):
        super(RecurrentNetwork, self).__init__()

        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.batch_first = batch_first
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size // self.num_directions
        self.target_size = target_size # In this case target size is equal to vocab_size



        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # self.embed = nn.Linear(vocab_size, embedding_dim)

        '''
        self.rnn = nn.LSTM(embedding_dim,
                           self.hidden_size,
                           num_layers=num_layers,
                           bias=True,
                           bidirectional=True,
                           # dropout=0.7,
                           batch_first=batch_first)
        '''
        self.rnn = LSTM(embedding_dim,
                           self.hidden_size,
                           num_layers=num_layers,
                           bidirectional=False,
                           )

        self.linear = nn.Linear(hidden_size, target_size)

        self.softmax = nn.LogSoftmax(dim=2)


    def init_hidden(self, batch_size):
        # return: (num_layers * num_directions, batch, hidden_size)
        a = self.num_layers * self.num_directions
        hidden = (Variable(torch.zeros((a, batch_size, self.hidden_size))), Variable(torch.zeros((a, batch_size, self.hidden_size))) ) 

        self.hidden = hidden

        return hidden


    def forward(self, padded_input, input_lengths):
        # input:(seq_len, batch_size)

        seq_len, batch_size = padded_input.size()
        # print ('seq_len: {}, batch_size: {}'.format(seq_len, batch_size))

        padded_embed = self.embed(padded_input)
        # shape: (seq_len, batch_size, embedding_dim)
        # print ('embed: ', padded_embed.shape)
        
        padded_out, _ = self.rnn(padded_embed, self.hidden)

        # shape: (seq_len, batch_size, hidden_size)
        # print ('unpacked: ', padded_out.shape)

        linear_out = self.linear(padded_out)
        # shape: (seq_len, batch_size, vocab_size)

        output = self.softmax(linear_out)
        # shape: (seq_len, batch_size, vocab_size)

        return output 

    
    def loss(self, output, target, padding_value):
        # output shape: (seq_len, batch_size, vocab_size)
        # target shape: (seq_len, batch_size)
        
        output = output.view(-1, self.vocab_size) # (seq_len * batch_size, vocab_size)
        tokens = target.view(-1) # (seq_len * batch_size)

        # != is broadcastable
        mask = (tokens != padding_value).float()

        # Ignoring the padding value, we get the number of total chars of this batch
        num_valid_tokens = int(torch.sum(mask))

        # Pick out softmaxed value for target word and masked out padding
        picked_values = output[range(output.shape[0]), tokens] * mask
        # This is tensor indexing, not slicing, which is output[:, tokens]

        cross_entropy_loss = - torch.sum(picked_values) / num_valid_tokens
        return cross_entropy_loss
        
        

