{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import pickle\n",
    "\n",
    "data_path = 'model-data'\n",
    "\n",
    "min_doc_length = 500\n",
    "top_words = 10000\n",
    "\n",
    "vocabdata_path = '%s/vocab3_books+reddit+wiki+stories+poems+iarpa.npz' % data_path\n",
    "vocabdata = np.load(vocabdata_path)\n",
    "words = set(vocabdata['svocab'][:top_words]) | set(vocabdata['storyvocab'])\n",
    "vocab = map(str, list(words))\n",
    "vocab.append('unk')\n",
    "num_vocab = len(vocab)\n",
    "\n",
    "int_to_word = {i: vocab[i] for i in xrange(num_vocab)}\n",
    "word_to_int = {int_to_word[i]: i for i in xrange(num_vocab)}\n",
    "\n",
    "# Un-comment if pickle doesn't already exist.\n",
    "\"\"\"\n",
    "from text.textcore.Corpus import Corpus\n",
    "corpus_files = ['raw-transcripts/stories1.tar.gz', 'raw-transcripts/stories2.tar.gz', \n",
    "                'reddit/allsubs-20120423.txt.gz', 'wiki/wikipedia_txt.tar.gz']\n",
    "\n",
    "corpus = Corpus('%s/'%data_path+corpus_files[2], min_length = min_doc_length, \n",
    "\t\t   split_documents = 100000000, replace_uncommon=5)\n",
    "# for i in xrange(3, 4):\n",
    "#     corpus.append_corpus('%s/'%data_path+corpus_files[i])\n",
    "documents = list(corpus.gen_document_arrays(vocab))\n",
    "num_documents = len(documents)\n",
    "\n",
    "with open('%s/reddit_docs_%d.pkl'%(data_path, min_doc_length), 'wb') as f:\n",
    "    pickle.dump(documents, f, pickle.HIGHEST_PROTOCOL)\n",
    "\"\"\"\n",
    "\n",
    "with open('%s/reddit_docs_%d.pkl'%(data_path, min_doc_length), 'rb') as f:\n",
    "    documents = pickle.load(f)\n",
    "\n",
    "num_documents = len(documents)\n",
    "print num_documents\n",
    "\n",
    "for i in xrange(num_documents):\n",
    "    documents[i] = np.array(documents[i], np.int32)\n",
    "    documents[i][documents[i]==-1] = num_vocab - 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cooccurrence(vocab, documents, window=15):\n",
    "    num_vocab = len(vocab)\n",
    "    word2int = {vocab[i] for i in xrange(num_vocab)}\n",
    "    cooccur = np.zeros([num_vocab, num_vocab])\n",
    "    for i in xrange(len(documents)):\n",
    "        doc_int = documents[i]\n",
    "#         doc_int = [word2int.get(word, 'unk') for word in doc]\n",
    "        for w in xrange(window, len(doc_int)):\n",
    "            w1 = doc_int[w]\n",
    "            for w2 in doc_int[w-window:window]:\n",
    "                cooccur[w1][w2] += 1\n",
    "                cooccur[w2][w1] += 1\n",
    "        if i % 10000 == 0: print i,\n",
    "    return map(np.log, cooccur+1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "C = cooccurrence(vocab, documents)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "safe_div = lambda a,b: a/b if b !=0 else np.zeros(a.shape[0])\n",
    "\n",
    "#Z-score\n",
    "E = map(safe_div, C - np.mean(C, axis=0), np.std(C, axis=0))\n",
    "E = map(safe_div, E - np.mean(E, axis=1), np.std(E, axis=1))\n",
    "\n",
    "E = np.array(E)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print E"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
