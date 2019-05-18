import random

import torch
import torch.nn as nn
import torch.optim as optim

from model import CharRNN
from utils import get_vocab_and_dataset
from utils import DataProvider, get_vocab_and_dataset


# Hyperparameters
learning_rate = 1
embedding_size = 256
hidden_size = 12
num_layers = 1
num_epochs = 1000
train_size = 0.8
batch_size = 100

vocab, dataset = get_vocab_and_dataset()
vocab_size = len(vocab)
print ('Vocab size: ', vocab_size)

provider = DataProvider(dataset, batch_size=batch_size, padding_value=vocab.padding_idx)

# checkpoint_path = './checkpoints/poetry-gen.pt'
checkpoint_path = './checkpoints/poetry-gen1.pt'

def get_model():
    try:
        model = torch.load(checkpoint_path)
        model.eval()
        print ('Load previous model from ', checkpoint_path)
        return model
    except:
        model = CharRNN(vocab_size=vocab_size,
                        embedding_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers)
        print ('Create a new model!')
        return model
    
model = get_model()
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.0001)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


for e in range(num_epochs):
    print ('Epoch: ', e)


    for batch in provider.batches():

        model.zero_grad()
        loss = 0

        for input_data, target_data in batch:

            hidden = model.init_hidden()
            output, hidden = model(input_data, hidden) 
            loss += criterion(output, target_data)

        loss.backward()
        optimizer.step()
        print ('Loss: ', loss)

    # torch.save(model, checkpoint_path)     

        








