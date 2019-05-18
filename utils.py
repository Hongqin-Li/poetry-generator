import random
import json
import torch

from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence

from vocabulary import Vocabulary
from collections import namedtuple

BEGIN = '<BOS>'
END = '<EOS>'
UNKNOWN = '<OOV>'

paths = ['raw_data/poet.tang.0.json']

tag_list = [BEGIN, END, UNKNOWN]

def json2poems(path):
    # path: path of json file, string-like
    # return: string
    data = json.loads(open(path).read())
    poems = []
    for p in data:
        paragraphs = p['paragraphs'] # type: list
        poem = ''.join(paragraphs)
        poems.append(poem)
    return poems

def jsons2poems(paths):
    # paths: [path1(string), ...]
    # return: [string, ...]
    poems = []
    for p in paths:
        poems += json2poems(p) 
    return poems

def poems2vocab(poems, vocab=None):
    # poems: [string, ...]

    if vocab is None:
        vocab = Vocabulary(unknown=UNKNOWN)
    vocab.update(tag_list)

    for p in poems:
        words = [c for c in p]
        vocab.update(words)

    return vocab

def poem2data(poem, vocab):
    # poem type: string, shape: (number of chars)
    # return: ([int, ...], [int, ...]) 

    token = torch.LongTensor([vocab[BEGIN]] + [vocab[w] for w in poem] + [vocab[END]])
    input_data = token[:-1]
    target_data = token[1:]
    return input_data, target_data
        
def poems2data(poems, vocab):

    return [poem2data(p, vocab) for p in poems]

def get_vocab_and_dataset():
    poems = jsons2poems(paths)
    vocab = poems2vocab(poems)

    dataset = [poem2data(p, vocab) for p in poems]

    return vocab, dataset


def get_poem(path):
    # :path: str
    # :return: [str, ...]

    data = json.loads(open(path).read())
    poems = []
    for p in data:
        paragraphs = p['paragraphs'] # type: list
        poem = ''.join(paragraphs)
        poems.append(poem)
    return poems

def update_vocab(vocab, poems):
    # :poems: [str, ...]
    
    if vocab is None:
        vocab = Vocabulary(unknown=UNKNOWN)
    vocab.update(tag_list)

    for p in poems:
        words = [c for c in p]
        vocab.update(words)

def get_vocab(paths):

    vocab = None
    for path in paths:
        poems = get_poems(path)
        update_vocab(vocab, poems)

    return vocab
        
class DataProvider2():

    def __init__(self, files, batch_size=50, padding_value=0):
        self.files = files
        self.batch_size = batch_size
        self.padding_value = padding_value
        self.vocab = get_vocab(files)


    def padded_batch(self, shuffle=True):

        for path in self.files:

            dataset = poems2data(get_poem(path), self.vocab)
            dataset = [(torch.LongTensor(inp), torch.LongTensor(t)) for inp, t in dataset]

            batch_size = self.batch_size
            num_data = len(dataset)
            num_batch = num_data // batch_size
            if shuffle: random.shuffle(shuffle)
            
            for start in range(0, num_data, batch_size):

                end = min(num_data, start + batch_size)

                sorted_batch = sorted(dataset[start:end], key=lambda t: len(t[0]), reverse=True)
                sorted_lengths = [len(i) for i, t in sorted_batch]

                input_batch, target_batch = [inp for inp, t in sorted_batch], [t for inp, t in sorted_batch]

                pib, ptb = pad_sequence(input_batch, padding_value=self.padding_value), pad_sequence(target_batch, padding_value=self.padding_value)

                print ('padded input batch: ', pib.shape)
                print ('padded target batch: ', ptb.shape)
                yield pib, ptb, sorted_lengths


class DataProvider():
    def __init__(self, dataset, batch_size, padding_value):
        # input_data(inp): (seq_len)
        # target_data(t): (seq_len)
        # data(d): (input_data, target_data)

        # dataset(ds): [data, ...]
        # padding_value: Int

        print ('Initializing data provider...')
        

        # Transfrom each input data and target data to tensor-like data 
        self.dataset = [(torch.LongTensor(inp), torch.LongTensor(t)) for inp, t in dataset]

        self.batch_size = batch_size
        self.num_data = len(dataset)
        self.num_batch = self.num_data // batch_size
        self.padding_value = padding_value

        print ('Dataset len: ', self.num_data)
        print ('Batch size: ', self.batch_size)
        print ('Number of batch: ', self.num_batch)

    def shuffle(self):
        random.shuffle(self.dataset)

    def batches(self, shuffle=True):

        if shuffle:
            random.shuffle(self.dataset)
            print ('Shuffle dataset')

        for end_idx in range(self.batch_size, self.num_data, self.batch_size):
            yield self.dataset[end_idx - self.batch_size: end_idx]


    def padded_batches(self, shuffle=True):
        # return: ([max_seq_len, batch_size], [max_seq_len, batch_size])

        if shuffle:
            random.shuffle(self.dataset)
            print ('Shuffle dataset')

        for batch_idx in range(self.num_batch):
            start = batch_idx * self.batch_size
            end = min(self.num_data, (batch_idx + 1) * self.batch_size)

            sorted_batch = sorted(self.dataset[start:end], key=lambda t: len(t[0]), reverse=True)
            sorted_lengths = [len(i) for i, t in sorted_batch]

            input_batch, target_batch = [inp for inp, t in sorted_batch], [t for inp, t in sorted_batch]

            pib, ptb = pad_sequence(input_batch, padding_value=self.padding_value), pad_sequence(target_batch, padding_value=self.padding_value)

            # print ('padded input batch: ', pib.shape)
            # print ('padded target batch: ', ptb.shape)
            yield pib, ptb, sorted_lengths

if __name__ == '__main__':
    # Usage
    vocab, dataset = get_vocab_and_dataset()

    provider = DataProvider(dataset, batch_size=100, padding_value=vocab.padding_idx)

    for batch in provider.batches():
        print (len(batch))
        for inp, target in batch:
            print (inp.shape)
            print (target.shape)
        input()


    for input_batch, target_batch, sorted_lengths in provider.padded_batches():
        print (input_batch.shape)
        print (target_batch.shape)
        print (input_batch)
        print (target_batch)
        input()


