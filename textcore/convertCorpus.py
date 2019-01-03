#!/usr/bin/python
"""This script's purpose is to convert a Corpus from a textual description to a set of arrays and an associated vocabulary.
"""
import tables
import logging
from textcore.Corpus import Corpus

try:
    import cPickle
except ImportError:
    import pickle as cPickle

logging.basicConfig(level=logging.INFO)

vocab_file = "/auto/k1/huth/text/data/dict.pickle"
vocab = cPickle.load(open(vocab_file))

def convert_corpus(corpusfile, outfile):
    """Loads the given [corpusfile] (which should be of a type loadable by Corpus) as a list of lists of sentence
    arrays (the output of Corpus.build_sentence_document_arrays) and saves it as an HDF5 file at [outfile].
    """
    c = Corpus(corpusfile)#, ndocs=10000)
    print "Building document arrays.."
    docarrays,mvocab = c.build_sentence_document_arrays(vocab)

    print "Opening output file.."
    tf = tables.openFile(outfile, mode="w", title="converted_corpus")

    print "Saving document arrays.."
    darr = tf.createVLArray(tf.root, "docarrays", tables.Int32Atom(shape=()))
    for da in docarrays:
        darr.append(da)

    print "Saving vocabulary.."
    varr = tf.createVLArray(tf.root, "vocab", tables.StringAtom(max(map(len, mvocab))))
    for w in mvocab:
        varr.append(w)

    print "Closing output file.."
    tf.close()

def read_corpus(fname):
    """Loads a converted corpus from the given [outfile].
    """
    tf = tables.openFile(fname, mode="w")
    docarrays = tf.getNode("/docarrays").read()
    vocab = [vn[0] for vn in tf.getNode("/vocab").read()]
    return docarrays, vocab

## Main
if __name__=="__main__":
    import sys
    convert_corpus(sys.argv[1], sys.argv[2])
