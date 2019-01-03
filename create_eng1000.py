#!python

import sys
import numpy as np
import logging

import textcore
from SemanticModel import SemanticModel
from textcore.word_sets import english1000
from config import standard_corpora

logging.basicConfig(level=logging.INFO)

def get_vocab(vocab_data_path, top_words=10000):
    # load vocab data and extract selected set
    vocabdata = np.load(vocab_data_path)
    words = set(vocabdata['sorted_vocab'][:top_words]) | set(vocabdata['stimvocab'])
    vocab = [str(w.decode()) for w in words]
    vocab.append('unk') # append 'unk' token that will take the place of unknown words
    return vocab

def make_english1000(corpus, vocab, targets=english1000, window=15):
    num_vocab = len(vocab)
    num_targets = len(targets)
    default = vocab.index('unk')
    
    target_inds = np.array([vocab.index(w[0]) for w in targets if w[0] in vocab])
    target2idx = {v_i:i for i,v_i in enumerate(target_inds)}

    cooccur = np.zeros([num_targets, num_vocab])

    for doc in corpus.gen_document_arrays(vocab, default=default):
        # which words are targets?
        target_words = np.nonzero(np.in1d(doc, target_inds))[0]

        for w in target_words:
            w1 = target2idx[doc[w]] # find index of w among targets

            for w2 in doc[max(w-window, 0):w]: # before w
                cooccur[w1][w2] += 1
            for w2 in doc[w+1:min(w+window, len(doc))]: # after w
                cooccur[w1][w2] += 1
    
    sm = SemanticModel(np.log(cooccur + 1), vocab)
    sm.zscore(axis=0)
    sm.zscore(axis=1)

    return sm

if __name__ == "__main__":
    vocab_file = sys.argv[1]
    outfile = sys.argv[2]

    print('getting vocab from %s..' % vocab_file)
    print('saving result to %s..' % outfile)

    vocab = get_vocab(vocab_file)

    stim_transcripts = textcore.get_transcript_uris()
    corpus = textcore.Corpus(standard_corpora + stim_transcripts,
                             split_documents=100000000,
                             min_length=200)

    sm = make_english1000(corpus, vocab)
    sm.save(outfile)
