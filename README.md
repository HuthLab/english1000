# English1000

This code can be used to create new English1000 word embedding spaces. It automatically reads from corpora stored in [corral](http://c3-dtn01.corral.tacc.utexas.edu:9002), and uses them to generate vocabulary files as well as co-occurrence matrices. These scripts require cottoncandy and numpy to run (and those are the only dependencies..?). To install them, run this line in this directory:

```bash
pip install -r requirements.txt
```

## Step 1: Create a vocab
To create a new English1000 embedding space you first must create a vocabulary. This can be done using the `create_vocab.py` script, which (1) counts all the words in the standard corpora (this takes a while), and (2) finds all the unique words in the stimulus transcripts. The results are stored in an `npz` file of your choosing, which will have: `sorted_vocab`, all the words that appear in any of the standard corpora sorted by (decreasing) frequency; `sorted_freq`, the number of occurrences for each word; and `stimvocab`, the list of all words appearing in the stimulus transcripts.

```bash
python create_vocab.py my_vocab.npz
```

## Step 2: Create embeddings
Next, you can use that vocabulary to count co-occurrences and create the embedding space. This is done using the `create_eng1000.py` script. By default, this creates an embedding space with a vocabulary consisting of the 10,000 most common words across the corpora (according to the counts above) combined with all the words appearing in the stimulus transcripts. It counts co-occurrences between each of these words and each of the 985 target words (see bottom of [the word set file](textcore/word_sets.py)). These counts are log-transformed, then the embeddings are normalized by z-scoring first across the vocab, then across the target words. The result is stored in an HDF-5 file of your choosing. This will take a while to run.

```bash
python create_eng1000.py my_vocab.npz my_english1000.hdf5
```

## Step 3: Use embeddings
Finally, you can load and use the embeddings with the [SemanticModel](SemanticModel.py) class.

```python
In [1]: from SemanticModel import SemanticModel
In [2]: sm = SemanticModel.load("my_english1000.hdf5")
In [3]: sm.find_words_like_word('finger')
Out[3]:
[(1.0, u'finger'),
 (0.4754624694722626, u'fingers'),
 (0.45213539324814495, u'neck'),
 (0.43652699294556874, u'cheek'),
 (0.4347026412612912, u'knees'),
 (0.4340886407280357, u'nose'),
 (0.42065469865933214, u'holding'),
 (0.4187342246033555, u'chin'),
 (0.417408122896769, u'chest'),
 (0.4143456430190933, u'thumb')]

In [4]: sm['finger']
Out[4]:
array([-4.09120218e-02, -5.90669127e-02,  4.84000528e-01,  7.87462823e-01,
        ...
```
