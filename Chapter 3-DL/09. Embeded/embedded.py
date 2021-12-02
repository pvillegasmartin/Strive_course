import torch
from numpy import matmul
from torch import nn
from torch.autograd  import Variable

import numpy as np

def preprocess(pth):
    with open(pth,'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace(".","").replace(",","").replace(":","").replace(";","").replace("?","").replace("!","").replace("-","").replace("_","").replace("\"","").replace("\'","")
        with open("./text/processed.txt", "w") as p:
            p.writelines(lines)

def get_sentences(pth):
    f = open(pth, 'r')
    lines = f.readlines()
    sentences = [line.lower().split() for line in lines]
    f.close()
    return sentences

def clean_sentences(sentences):
    i=0
    while i<len(sentences):
        if sentences[i] == []:
            sentences.pop(i)
        else:
            i += 1
    return sentences

def get_dicts(sentences):
    vocab = []
    for sentence in sentences:
        for token in sentence:
            if token not in vocab:
                vocab.append(token)
    w2i = {w:i for (i,w) in enumerate(vocab)}
    i2w = {i:w for (i,w) in enumerate(vocab)}

    return w2i, i2w, len(vocab)

def get_pairs(sentences, w2i, r):
    pairs = []
    for sentence in sentences:
        tokens = [w2i[word] for word in sentence]
        for center in range(len(tokens)):
            for context in range(-r, r+1):
                context_word = center + context
                if context_word<0 or context_word>=len(tokens) or context_word==center:
                    continue
                else:
                    pairs.append((tokens[center], tokens[context_word]))
    return np.array(pairs)

def get_dataset(pth):
    #Hyperparameters
    num_context = 4

    sentences = get_sentences(pth)
    clean_sents = clean_sentences(sentences)
    w2i, i2w, vocal_len = get_dicts(clean_sents)
    pairs = get_pairs(clean_sents, w2i, num_context)
    return pairs, vocal_len

def input_layer(word_idx,vocab_size):

    x = torch.zeros(vocab_size)
    x[word_idx] = 1.0
    return x



def train(n_epochs = int, lr = float, embedding_size = int):
    dataset, vocab_size = get_dataset(r'texts/blakepoems.txt')

    W1 = Variable(torch.randn(vocab_size, embedding_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(embedding_size, vocab_size).float(), requires_grad=True)

    for epoch in range(n_epochs):

        loss_val = 0

        for data, target in dataset:
            x = Variable(input_layer(data,vocab_size))#.float
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            z1 = matmul(x.detach().numpy(), W1.detach().numpy())
            z2 = matmul(z1, W2.detach().numpy())

            #log_softmax = nn.LogSoftmax(torch.from_numpy(z2), dim=1)
            log_softmax = nn.LogSoftmax(dim=0)
            loss = nn.NLLloss(log_softmax(1, -1), y_true)

            loss_val += loss

            W1.data -= lr * W1.gradient_data

            W2.data -= lr * W2.gradient_data

            W1.gradient_data = 0
            W2.gradient_data = 0

            if epoch % 10 == 0:
                print(f'Loss at epoch {epoch}: {loss_val / len(dataset)}')

    return W1, W2

if __name__ == '__main__':
    W1, W2 = train(10, 0.1, 2)
    print('jndkf')