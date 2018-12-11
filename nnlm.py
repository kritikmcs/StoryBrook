# sexy file
import os
import torch
print('Torch loaded.')
import torch.nn as nn
import model
import data

import numpy as np
import pickle

# model_path = 'model.pt'
model_path = 'model-50.defhidden.pt'
corpus_path = 'corpus.pk'
cache_path = 'prob_cache.pk'

corpus = pickle.load(open(corpus_path, 'rb'))
model = torch.load(open(model_path, 'rb'))
print('Language model loaded.')
if os.path.exists(cache_path):
    cache = pickle.load(open(cache_path, 'rb'))
else:
    cache = {}

log_prob_criterion = nn.LogSoftmax()


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
    
def get_ids(sentence, corpus):
    vocabulary_size = len(corpus.dictionary.idx2word)
    words = sentence.split() + ['<eos>']
    sentence_sequence = torch.LongTensor(len(words)).cuda()
    for i, word in enumerate(words):
        word = word.strip(',.')
        if word in corpus.dictionary.word2idx:
            sentence_sequence[i] = corpus.dictionary.word2idx[word]
        else:
            sentence_sequence[i] = 0
    return sentence_sequence.view(-1, 1)


def get_batch(source, i, seq_len=3):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def id2word(word_id):
    if word_id < len(corpus.dictionary.idx2word):
        return corpus.dictionary.idx2word[word_id]
    return '<unk>'

def evaluate(data_source, seq_len=2, eval_batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    log_sentence_probability = 0
    sentence_probability = 1
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            for j, tensor in enumerate(output):
                
                targets_id = targets[j].data[0].item()                
                if targets_id > len(corpus.dictionary.idx2word):
                    continue
                
                log_probabilities = log_prob_criterion(tensor[0])
                log_sentence_probability += log_probabilities[targets_id]
                
            hidden = repackage_hidden(hidden)
    
    return log_sentence_probability.item()


def probability_of(sent, given_sent=''):
    
    # P(sent, given_sent) / P(given_sent)
    # = evaluate(sent + given_sent) / P(given_sent)
    if given_sent == '':
        if sent in cache:
            return cache[sent]
        sentence_ids = get_ids(sent, corpus)
        sentence_probability = evaluate(sentence_ids)
        cache[sent] = sentence_probability
        pickle.dump(cache, open(cache_path, 'wb'))
        return sentence_probability
    
    
    combined_sentence = ' '.join([given_sent, sent])
    
    if sent in cache:
        sent_probability = cache[sent]
    else:
        sent_probability = probability_of(sent)
        cache[sent] = sent_probability
        pickle.dump(cache, open(cache_path, 'wb'))
        
    if combined_sentence in cache:
        combined_sent_probability = cache[combined_sentence]
    else:
        combined_sent_probability = probability_of(combined_sentence)
        cache[combined_sentence] = combined_sent_probability
        pickle.dump(cache, open(cache_path, 'wb'))
    
    if sent_probability != 0:
        return combined_sent_probability - sent_probability
    return 0


if __name__ == '__main__':
        
    s1 = 'The good'
    sentence = 'I like gaming.'
    given_sentence = 'Battlefield is coming out.'
    
#     print('Prob of sentence', probability_of(s1))
    print('Prob of sentence', probability_of(sentence))
#     print('Conditional probability', probability_of(sentence, given_sent=given_sentence))