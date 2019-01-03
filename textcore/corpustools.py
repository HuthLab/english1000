import os
import re
import operator
import logging
import tarfile
import numpy as np
import itertools as itools

#from ngram import ngram_string

logger = logging.getLogger("textcore.corpustools")

def load_vocabulary(vocab_file):
    """Loads a set of words from the given [vocab_file]."""
    return list(get_word_set(open(vocab_file).read()))

def get_word_set(string):
    """Returns a set of all words in [string]."""
    return set([ng[0] for ng in ngram_string(string, 1)]) ## Need this to load the vocab consistently with old code ##
    #return set(reduce(operator.add, gen_sentences_from_string(string)))

def filter_sentences(sentences, minlength, maxlength, dictionary_file):
    """Reads sentences from [sentences], each of which should be a single string.
    Each sentence is stripped of punctuation and its length is determined.  If the sentence
    contains any words not found in [dictionary_file], the sentence is excluded.  If the 
    sentence is longer than [maxlength] or shorter than [minlength] it is excluded. A list of
    non-stripped sentences is returned.
    """
    ## Define test dictionary ##
    print("Loading dictionary...")
    dictionary = load_vocabulary(dictionary_file)
    
    ## Read sentences ##
    good_sentences = []
    for sentence in sentences:
        punctuation_strip = sentence.replace(".", "")
        punctuation_strip = punctuation_strip.replace("?", "")
        punctuation_strip = punctuation_strip.replace("!", "")
        sentence_words = set([w[0] for w in ngram_string(punctuation_strip, 1)])
        
        ## Filter sentence ##
        if minlength <= len(sentence_words) <= maxlength and sentence_words.issubset(dictionary):
            good_sentences.append(sentence)
    
    ## Return good sentences ##
    return good_sentences

def gen_sentences_from_string(string):
    """Yields each sentence found in [string] as a separate list of word strings."""
    chunks = strip_string(string).split(".")
    for chunk in chunks:
        sent = chunk.strip().split()
        if sent: ## False if the sentence is empty ##
            yield sent

def gen_sentences_from_file(filename):
    """Yields each sentence found in [filename] as a separate list of word strings."""
    return gen_sentences_from_string(open(filename).read())

def gen_documents_from_tar(path, min_length=0, ext="txt", startfrom=0):
    """Yields the cleaned-up full text from each document with the extension [ext] found in the
    tar file at [path].  If a [min_length] is given, only documents with more than [min_length] 
    characters will be returned.  If [startfrom] is specified, the first [startfrom]-1 documents
    will be skipped.
    """
    tar = tarfile.open(path)
    ndone = 0
    for finfo in tar:
        ndone += 1
        logger.debug("Loading file %d..." % (ndone+1))
        ## Check to make sure the thing we're extracting has correct extension ##
        efile = tar.extractfile(finfo)
        if finfo.name.endswith(ext) and efile:
            if ndone>startfrom:
                fstr = strip_string(efile.read())
                if not min_length or len(fstr) > min_length:
                    logger.debug("Yielding file %s" % finfo.name)
                    yield fstr
        tar.members = [] ## Hack to reduce memory usage.  Fucking sucks. ##

def count_documents_in_tar(path, ext="txt"):
    """Counts the number of files with the extension [ext] in the tarball found at [path]."""
    tar = tarfile.TarFile(path)
    ndone = 0
    for finfo in tar:
        if finfo.name.endswith(ext):
            ndone += 1
        tar.members = []
    return ndone

def gen_documents_from_dir(path, min_length=0, ext="txt"):
    """Yields the cleaned-up full text from each document with the extension [ext] found in [path].
    If a [min_length] is given, only documents with more than [min_length] characters will be returned.
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]
    for fi,f in enumerate(files):
        logger.debug("Loading file %d / %d..." % (fi+1, len(files)))
        fstr = open(f).read()
        if not min_length or len(fstr) > min_length:
            logger.debug("Yielding file %s" % f)
            yield strip_string(open(f).read())

def gen_sentences_from_dir(path, ext="txt"):
    """Yields each sentence found in the files in [path] with the extension [ext]."""
    for doc in gen_documents_from_dir(path, ext):
        for s in gen_sentences_from_string(doc):
            yield s

nonalpha = re.compile(r"[^a-z\.\?!;\-']")
multispace = re.compile(r"\s+")
def strip_string(string, include_period=True):
    """Lower-cases the given [string] and strips it of extraneous punctuation."""
    lstring = string.lower()
    ## Remove all non-alphabetic characters ##
    ## (excluding periods, question marks, semicolons, apostrophes and exclamation marks) ##
    astring = nonalpha.sub(" ", lstring)
    
    ## Replace all multiple spaces (created everywhere) with single spaces ##
    fstring = multispace.sub(" ", astring)
    
    ## Replace alternate punctuation with periods for easy sentence parsing ##
    fstring = fstring.replace('?','.') ## Period from question mark
    fstring = fstring.replace(';','.') ## Period from semicolon
    fstring = fstring.replace('!','.') ## Period from exclamation mark
    
    ## Remove punctuation if we don't need it ##
    if not include_period:
        fstring.replace('.', '')
    
    return fstring

def word_index(vocab_idx, words, default=-1):
    """Takes [vocab_idx], a dictionary of word:number pairs, and [words], a list of words, and returns
    an array of numbers the same size as the list of words.
    If a word in [words] doesn't appear in the vocab, the [default] value is inserted instead.
    """
    narray = np.zeros((len(words),), dtype=np.int32)
    for wi, w in enumerate(words):
        try:
            narray[wi] = vocab_idx[w]
        except KeyError:
            narray[wi] = default
    return narray

def replace_word_ind(word, inds, default):
    try:
        return inds[word]
    except KeyError:
        return default

def combine_textfiles(filelist, outfile, sep=" "):
    """Horizontally concatenates the textfiles in [filelist] into [outfile].
    Thus line 1 of outfile will be line1 of each file in [filelist] joined by [sep].
    """
    fobs = [open(f) for f in filelist]
    outob = open(outfile, "w")
    for lines in itools.izip(*fobs):
        outob.write(sep.join([l.strip() for l in lines])+"\n")
    outob.close()
