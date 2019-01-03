from textcore import Corpus

cfile1 = 's3://corpora/corpora/books/books_txt.tar.gz'
cfile2 = 's3://corpora/corpora/reddit/reddit.txt'
cfile3 = 's3://corpora/corpora/reddit/askreddit-20140721.txt.gz'


c1 = Corpus(cfile1, min_length=200)
print(c1.get_documents().next())

c2 = Corpus(cfile2, min_length=200)
print(c2.get_documents().next())

c3 = Corpus(cfile3, min_length=200)
print(c3.get_documents().next())
