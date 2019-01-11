#!/usr/bin/env python

import sys
from collections import Counter
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

import textcore
from config import standard_corpora

def create_vocab(outfile):
    ## Count all the words in the standard corpora + stimuli
    stim_transcripts = textcore.get_transcript_uris()

    corpus = textcore.Corpus(standard_corpora + stim_transcripts)

    vocab, wf_list = zip(*corpus.get_vocabulary().items())
    wordfreq = np.array(wf_list)

    ## Discard words with 50 or more characters
    wordlens = np.array(list(map(len, vocab)))
    sel_words = wordlens < 50
    sel_vocab = np.array(vocab, dtype='|S50')[sel_words]
    sel_wordfreq = wordfreq[sel_words]

    ## Sort vocab according to frequency
    wforder = np.argsort(sel_wordfreq)[::-1]
    sorted_vocab = list(sel_vocab[wforder])
    sorted_freq = sel_wordfreq[wforder]

    ## Find all words in sub-corpus that we want to have full coverage on
    stimcorpus = textcore.Corpus(stim_transcripts)
    stimvocab = list(np.array(stimcorpus.get_vocabulary().keys(), dtype='|S50'))

    np.savez(outfile,
             sorted_vocab=sorted_vocab,
             sorted_freq=sorted_freq,
             stimvocab=stimvocab)

if __name__ == "__main__":
    filename = sys.argv[1]
    if not filename.endswith('.npz'):
        filename += '.npz'

    print('outputting vocab to %s' % filename)
    create_vocab(filename)
