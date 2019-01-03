import tables
import numpy as np
import logging

logger = logging.getLogger("textcore.HDFCorpus")

class HDFCorpus(object):
    """This class mimics the Corpus interface to expose a pre-indexed HDF5-stored text corpus.
    """
    def __init__(self, filename, vocab, startfrom=0, skip=0, ndocs=0):
        """Initializes an HDFCorpus that will read documents from the HDF5 file at [filename],
        where the documents are stored in a VLArray (variable-length array) called '/docs' as
        lists of integer word indexes in the [vocab].
        """
        self.filename = filename
        self.vocab = vocab
        self.tf = tables.openFile(self.filename)
        self.docnode = self.tf.root.docs
        
        self.startfrom = startfrom
        if not ndocs:
            self.ndocs = self.docnode.nrows
        else:
            self.ndocs = ndocs
        self.skip = skip
        
    def gen_document_arrays(self, another_vocab):
        """Generates documents from this corpus.  Ignores [another_vocab], it's only here for
        interface compatibility with the original Corpus... this is kind of ugly.
        """
        for doc in self.docnode.read(self.startfrom, self.ndocs, self.skip+1):
            yield doc

    def break_into_chunks(self, nchunks):
        """Breaks this corpus into [nchunks] chunks.  Returns a list of HDFCorpus objects.
        """
        newcorpora = []
        for n in range(nchunks):
            newcorp = HDFCorpus(self.filename, self.vocab, self.startfrom+n, nchunks-1, self.ndocs)
            newcorpora.append(newcorp)
        return newcorpora
