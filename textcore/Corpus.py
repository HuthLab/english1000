"""Defines the Corpus class, which is used to wrap a lot of useful functions for dealing with text corpora.
"""
import os
import re
import operator
import logging
import tarfile
import numpy as np
import itertools
import gzip
import zlib
import time
from collections import Counter

import cottoncandy as cc

# for python2/3 compatibility
if hasattr(itertools, "imap"):
    imap = itertools.imap
else:
    imap = map

#from ngram import ngram_string
from .corpustools import gen_sentences_from_string, strip_string, word_index, replace_word_ind

logger = logging.getLogger("textcore.Corpus")

class Corpus(object):
    """The Corpus class can be used to load textual corpora from many sources and process them in useful ways."""

    def __init__(self, filename, min_length=0, split_documents=0, replace_uncommon=0, ndocs=0, startfrom=0, skip=0):
        """Initializes a Corpus object with the [filename] of the file it is meant to draw from.
        The file can be of the following types:
          - A gzipped TAR file, e.g. "file.tar.gz".  In this case we will assume that every file contained in
            the archive is a separate document, and will be treated thus (though documents can be broken down
            further).
          - A regular TAR file, similarly to above.
          - A text file, e.g. "file.txt", in which case each line will be treated as a separate document.

        The [min_length] argument is used to filter out very short documents (e.g. stubs from wikipedia) that
        would only waste space and time.  Documents with fewer than [min_length] words are ignored.
        
        The [split_documents] argument is used to define roughly how long each document can be allowed to be
        before being split, or documents are not split if it is zero.  When constructing topic models, very
        long documents (e.g. books or other documents containing 1e5+ words) are not terribly useful because
        they are often not semantically coherent throughout, as topics models often assume.  To deal with this
        problem we can split long documents into smaller sub-documents which should be much more semantically
        stable and thus should yield better training results.  With [split_documents] you can set the maximum
        number of words in each document before it is subdivided.

        The [replace_uncommon] argument can be used to remove very uncommon words from the corpus, which will
        reduce the chance of getting extremely noisy estimates.  Only words that appear in at least
        [replace_uncommon] documents will be included in the document arrays.

        If [ndocs] is given as a nonzero value, only the first [ndocs] documents will be yielded.

        [startfrom] documents will be skipped at the beginning.

        If [skip] is given, [skip] documents will be skipped before each document is read.
        """
        if isinstance(filename, (list, tuple, set)):
            self.filenames = list(filename)
        else:
            self.filenames = [filename]
        self.min_length = min_length
        self.split_documents = split_documents
        self.replace_uncommon = replace_uncommon
        self.ndocs = ndocs
        self.startfrom = startfrom
        self.skip = skip

        self.logevery = 10000 ## Print log messages every this many documents read
        self.tarext = "txt" ## Only files with this extension will be loaded from TAR files

        self.ccinterfaces = dict() ## Holds cottoncandy interface objects

    def append_corpus(self, filename):
        """Appends the corpus in [filename] to this corpus, reading it after this corpus is exhausted.
        """
        self.filenames.append(filename)

    def _get_cc_interface(self, bucket):
        """Get a cottoncandy interface for the given bucket. Will first check the
        local cache.
        """
        if bucket not in self.ccinterfaces:
            self.ccinterfaces[bucket] = cc.get_interface(bucket)
        return self.ccinterfaces[bucket]

    def _get_corpus_iterator(self, filename):
        """Returns an iterator for the given [filename], dispatching to get_docs_from_tar or
        get_docs_from_txt.
        """
        logger.info("Starting to read corpus %s.." % filename)
        if filename.startswith("s3://"):
            # the corpus is in the cloud, get a stream object using cottoncandy
            bucket, object_name = cc.utils.split_uri(filename)
            cci = self._get_cc_interface(bucket)
            #stream = cci.download_stream(object_name).content
            stream = cci.get_object(object_name).get()['Body'] # avoids download
        else:
            stream = None

        if filename.endswith("tar.gz") or filename.endswith("tar"):
            dociter = self._gen_docs_from_tar(filename, self.tarext, fileobj=stream)
        elif filename.endswith("txt") or filename.endswith("txt.gz"):
            dociter = self._gen_docs_from_txt(filename, fileobj=stream)
        else:
            raise ValueError("The provided filename, %s, is not one of the allowed types (.tar.gz, .tar, .txt, .txt.gz)."%filename)

        return dociter

    def _gen_docs_from_tar(self, filename, tarext, fileobj=None):
        """Generates documents from the stored [filename] TARfile.
        Each document is returned as a string.
        """
        tar = tarfile.open(filename, fileobj=fileobj, mode='r|*')
        ndone = 0
        for finfo in itertools.islice(tar, 0, None, self.skip+1):
            ndone += 1
            logger.debug("Loading file %d..." % (ndone+1))
            ## Check to make sure the thing we're extracting has correct extension ##
            efile = tar.extractfile(finfo)
            if finfo.name.endswith(tarext) and efile:
                fstr = strip_string(str(efile.read()))
                logger.debug("Yielding file %s" % finfo.name)
                yield fstr
            tar.members = [] ## Hack to reduce memory usage.  Fucking sucks. ##

    def _gen_docs_from_txt(self, filename, fileobj=None):
        """Generates documents from the stored [filename] text file.
        Each document is returned as a single string.
        """
        # with open(filename) as tfile:
        #     for line in itertools.islice(tfile, 0, None, self.skip+1):
        #         yield line
        if filename.endswith("txt"):
            if fileobj is None:
                tfile = open(filename)
            else:
                tfile = fileobj
        elif filename.endswith("gz"):
            #tfile = gzip.GzipFile(filename, fileobj=fileobj)
            tfile = stream_gzip_decompress(fileobj)

        for line in itertools.islice(tfile, 0, None, self.skip+1):
            yield line
    
    def get_documents(self):
        """Generates documents from the stored [filename] constrained by the stored [split_documents] and
        [min_length] parameters.  If present, any [appended_corpora] will also be fetched.

        Each document is yielded as a list of sentences, with each sentence as a list of words.
        """
        ## Build an iterator chain out of all the files to read
        dociter = itertools.chain.from_iterable(imap(self._get_corpus_iterator, self.filenames))
        
        ## Slice down to only the documents we want if we only want a limited number
        if self.ndocs:
            dociter = itertools.islice(dociter, self.ndocs)

        ## Slice away first documents using startfrom
        dociter = itertools.islice(dociter, self.startfrom, None)
        
        ## Finally iterate through the documents
        ndone = 0
        for doc in dociter:
            ## Do logging stuff
            ndone += 1+self.skip
            if not ndone%self.logevery:
                logger.info("Read %d documents"%ndone)

            ## Convert document to a list of lists of words
            sentences = strip_string(str(doc)).split(".")
            sentwords = filter(bool, [s.strip().split() for s in sentences])
            sentwords = [[w.strip('\'') for w in s] for s in sentwords]
            
            ## Now decide if we need to throw this document out for being too short
            sentlens = [len(s) for s in sentwords]
            doclen = sum(sentlens)
            if doclen >= self.min_length:
                ## Document is long enough, now let's decide if we need to split it up
                if doclen > self.split_documents and self.split_documents:
                    ## We've got to split the document
                    for dchunk in itersplit(sentwords, self.split_documents, len):
                        if dchunk:
                            yield dchunk
                else:
                    yield sentwords
                    #if sentwords:
                    #    yield sentwords

    def break_into_chunks(self, nchunks):
        """Breaks this corpus into [nchunks] chunks.  Returns a list of Corpus objects.
        """
        newcorpora = []
        for n in range(nchunks):
            newcorp = Corpus(self.filenames, self.min_length, self.split_documents, self.replace_uncommon,
                             self.ndocs, self.startfrom+n, nchunks-1)
            newcorp.appended_corpora = self.appended_corpora
            newcorpora.append(newcorp)
        return newcorpora
    
    def get_document_arrays(self, vocabulary):
        """Builds a list of vectors with each vector corresponding to a document.  The j'th element of the
        i'th vector is the index in the [vocabulary] of the j'th word in the i'th document.

        Uncommon words may be replaced, depending on the [replace_common] argument to the Corpus constructor.
        The word arrays and a reduce vocabulary will be returned.
        """
        nwords = len(vocabulary)
        vocab_idx = dict(zip(vocabulary, range(len(vocabulary))))
        docarrays = [word_index(vocab_idx, reduce(operator.add, doc, [])) for doc in self.get_documents()]
        return docarrays
        #return self.replace_uncommon_words(docarrays, vocabulary)

    def gen_sentence_arrays(self, vocabulary):
        """Generates sentence arrays.
        """
        nwords = len(vocabulary)
        vocab_idx = dict(zip(vocabulary, range(len(vocabulary))))
        for doc in self.get_documents():
            for sent in doc:
                yield word_index(vocab_idx, sent)

    def gen_document_arrays(self, vocabulary, default=-1):
        """Generates document arrays but doesn't replace uncommon words.
        """
        vocab_idx = dict(zip(vocabulary, range(len(vocabulary))))
        for doc in counter(self.get_documents(), 1000, logger=logger):
            #yield word_index(vocab_idx, reduce(operator.add, doc)) ## 122 seconds for 50 documents
            #yield map(lambda s:word_index(vocab_idx, s), doc) ## 10 seconds for 50 documents (but not flat)
            #yield word_index(vocab_idx, sum(doc, [])) ## 130 seconds for 50 documents
            yield word_index(vocab_idx, list(itertools.chain(*doc)), default=default) ## 8.7 seconds for 50 documents

    def dump_document_arrays(self, vocabulary, outfile):
        """Dumps document arrays in the given [outfile] using HDF5.
        """
        pass
    
    def build_sentence_document_arrays(self, vocabulary, replace_uncommon=True, groupdocs=1000):
        """Builds a list of vectors with each vector corresponding to a document.  Special markers are
        inserted for sentence starts and ends, and the modified vocabulary is returned along with the
        vectors.
        
        'SENTENCE_START' and 'SENTENCE_END' are used for the start- and end-of-sentence markers, respectively.
        """
        sstart = "SENTENCE_START"
        send = "SENTENCE_END"
        default = "*"

        flagsent = lambda s: [sstart]+s+[send]
        #flagsents = lambda ss: reduce(operator.add, [flagsent(s) for s in ss], [])
        #flagsents = lambda ss: list(itertools.chain(*(flagsent(s) for s in ss)))
        flagsents = lambda ss: list(itertools.chain(*(itertools.chain([sstart], s, [send]) for s in ss)))
        
        newvocab = list(vocabulary)
        newvocab += [sstart, send, default]

        logger.info("Reading documents..")
        vocab_idx = dict(zip(newvocab, range(len(newvocab))))
        defindex = newvocab.index(default)
        docarrays = [word_index(vocab_idx, flagsents(itertools.chain(*docs)), defindex)
                     for docs in grouper(groupdocs, counter(self.get_documents(), 1000, logger=logger), [])]
        #docarrays = [word_index(vocab_idx, flagsents(docs), defindex)
        #             for docs in counter(self.get_documents(), 1000, logger=logger)]

        if replace_uncommon:
            return self.replace_uncommon_words(docarrays, newvocab)
        else:
            return docarrays, newvocab
    
    def gen_simple_sentence_document_arrays(self, vocabulary, doflag=True):
        """Like build_sentence_document_arrays, but don't screw with the [vocabulary], just use it directly.
        """
        sstart = "SENTENCE_START"
        send = "SENTENCE_END"

        flagsent = lambda s: [sstart]+s+[send]
        flagsents = lambda ss: reduce(operator.add, [flagsent(s) for s in ss], [])
        
        logger.info("Reading documents..")
        vocab_idx = dict(zip(vocabulary, range(len(vocabulary))))
        if doflag:
            return (word_index(vocab_idx, flagsents(doc)) for doc in self.get_documents())
        else:
            return (word_index(vocab_idx, list(itertools.chain(*doc))) for doc in self.get_documents())
    
    def build_simple_sentence_document_arrays(self, vocabulary):
        return list(self.gen_simple_sentence_document_arrays(vocabulary))

    def replace_uncommon_words(self, docarrays, vocabulary):
        """Replaces uncommon words (words that appear in fewer than [replace_uncommon] documents) with a "*" and
        returns the modified vocabulary along with the modified document arrays.
        """
        ## First check to see if we have to do anything
        #if self.replace_uncommon <= 1:
        #    ## We don't have to replace anything, because every word that appears appears in at least one document
        #    return docarrays, vocabulary

        logger.info("Replacing missing words with *..")
        starind = len(vocabulary) ## Index of "*" semaphore word
        ## replace missing words with *
        for d in docarrays:
            for di in range(len(d)):
                if d[di]==-1:
                    d[di] = starind

        svocab = vocabulary + ["*"]
        logger.info("Replacing uncommon words with *..")
        uwords = [np.unique(words) for words in docarrays] ## Unique words in each document
        alluwords = np.sort(np.hstack(uwords)) ## All words that are unique in each document
        appwords = np.unique(alluwords) ## Unique words across corpus
        nappear = alluwords.searchsorted(appwords, side="right")-alluwords.searchsorted(appwords, side="left")
        goodwords = np.hstack([starind, appwords[nappear>=self.replace_uncommon]])
        rvocab = [svocab[i] for i in goodwords]
        word_inds = dict(zip(goodwords, range(len(goodwords))))
        wordindarrays = [np.array([replace_word_ind(i, word_inds, 0) for i in wa]) for wa in docarrays]
        
        return wordindarrays, rvocab
    
    def get_all_words(self):
        """Returns a list of all the words in the corpus (the concatenation of all the documents).
        """
        logger.info("Reading and concatenating all documents..")
        words = list(itertools.chain(*[itertools.chain(*doc) for doc in self.get_documents()]))
        # words = list()
        # for doc in self.get_documents():
        #     words += reduce(operator.add, doc, [])
        return words

    def gen_all_words(self):
        """Generates words from the corpus (the concatenation of all documents).
        """
        for doc in counter(self.get_documents(), 1000, logger=logger):
            for sent in doc:
                for word in sent:
                    yield word
    
    def get_all_word_inds(self, vocab):
        """Returns a list of all the words in this corpus as a numerical array where each word is replaced
        by its index in [vocab].
        """
        logger.info("Replacing words with indices..")
        return word_index(dict([(word,wi) for wi,word in enumerate(vocab)]), self.get_all_words())

    def get_vocabulary(self):
        """Returns a Counter object containing the count of every unique token
        in the corpus/corpora.
        """
        return Counter(self.gen_all_words())
    
    # def get_vocabulary(self, chunklen=1000):
    #     """Returns a vocabulary for this corpus composed of all the unique words in the corpus.
    #     The vocabulary is a list of words.
        
    #     [chunklen] is the number of documents per processing chunk. It needs to be tuned for
    #     performance.
    #     """
    #     logger.info("Building vocabulary..")
    #     im = itertools.imap
    #     ch = itertools.chain
    #     docset = lambda doc: set(ch(*doc))
    #     vocab = set()
    #     #chunklen = 1
    #     for docset in grouper(chunklen, ch(im(docset, self.get_documents())), ""):
    #         vocab = vocab.union(set(ch(*docset)))
        
    #     return list(vocab)
        #return list(set(self.get_all_words()))
    
    def get_word_frequency(self, vocab):
        """Returns the frequency of each word in [vocab] in the corpus as an array the same length as [vocab].
        """
        #logger.info("Counting word frequency..")
        widx = self.get_all_word_inds(vocab)
        logger.info("Actually counting words..")
        nV = float(len(widx))
        counter = np.zeros((len(vocab),)).astype(np.uint32)
        for w in widx:
            counter[w]+=1
        return counter


def itersplit(iterator, maxval, fun):
    """Splits [iterator] into smaller chunks by mapping [fun] over its values.
    reduce(fun, chunk) will be just greater than [maxval] for every chunk yielded.
    """
    newlist = []
    accum = 0
    for val in iterator:
        accum += fun(val)
        newlist += [val]
        if accum>maxval:
            yield list(newlist)
            newlist = []
            accum = 0

    if newlist:
        yield newlist

def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"""
    return itertools.izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def sliding_grouper(n, iterable):
    """sliding_grouper(3, 'abcde') --> ('a','b','c'), ('b','c','d'), ('c','d','e')"""
    myiters = itertools.tee(iterable, n)
    return itertools.izip(*[itertools.islice(myiters[ni], ni, None) for ni in range(n)])
    #return itertools.izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def stream_gzip_decompress(stream):
    dec = zlib.decompressobj(32 + zlib.MAX_WBITS)  # offset 32 to skip the header
    next_line = ""
    for chunk in stream:
        dchunk = dec.decompress(chunk)
        if dchunk:
            next_line += dchunk
            if "\n" in next_line:
                splits = next_line.split("\n")
                next_line = splits[-1]
                for out in splits[:-1]:
                    yield out

import time
import logging
def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()
    
    for count, thing in enumerate(iterable):
        yield thing
        
        if not count%countevery:
            current_time = time.time()
            rate = float(count+1)/(current_time-start_time)
            
            if total is not None:
                remitems = total-(count+1)
                remtime = remitems/rate
                timestr = ", %s remaining" % time.strftime('%H:%M:%S', time.gmtime(remtime))
                itemstr = "%d/%d"%(count+1, total)
            else:
                timestr = ""
                itemstr = "%d"%(count+1)
            
            logger.info("%s items complete (%0.2f items/second%s)"%(itemstr,rate,timestr))
