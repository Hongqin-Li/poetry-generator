import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CharRNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1):
        # num_layers: number of RNN layers

        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # self.embedding = nn.Linear(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):

        '''
        input shape: (num_char, 1)
        hidden shape:  ((num_char, 1, hidden_size), 
                        (num_char, 1, hidden_size))
        output shape: (num_char, vocab_size)
        '''
        length = input.size()[0] # num_char

        embeds = self.embedding(input).view((length, 1, -1)) # (seq_size, batch_size=1, input_size)
        output, hidden = self.lstm(embeds, hidden)
        output = F.relu(self.linear(output.view(length, -1)))
        output = self.softmax(output)
        return output, hidden
       
    def init_hidden(self, length=1):
        return (Variable(torch.zeros(length, 1, self.hidden_size)),
                Variable(torch.zeros(length, 1, self.hidden_size)))
