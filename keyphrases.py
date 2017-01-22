#################################################
import os

os.getcwd()

os.chdir(r'F:\Projects\keyphrase')

#################################################
'''
Testing the method described in: ""

Code referenced from: https://gist.github.com/ojedatony1616/e605222e4442da2db9e5d670d0f797b3
'''
#################################################


import gensim
import nltk
from nltk import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import itertools, string

#Read single file
f = open('/Users/tonyojeda/Desktop/keyphrase/ddl_corpus/corpus_text/2014-08-08-how-to-transition-from-excel-to-r.txt', 'rU')
text = f.read()

#Read entire corpus
#CORPUS_TEXT = '/Users/tonyojeda/Desktop/keyphrase/ddl_corpus/corpus_text'

CORPUS_TEXT = r'F:\Projects\keyphrase\Files2015A'
texts = PlaintextCorpusReader(CORPUS_TEXT, '.*\.txt')

def corpus_info(corpus):
    """
    Prints out information about the status of a corpus.
    """
    fids   = len(corpus.fileids())
    paras  = len(corpus.paras())
    sents  = len(corpus.sents())
    sperp  = sum(len(para) for para in corpus.paras()) / float(paras)
    tokens = FreqDist(corpus.words())
    count  = sum(tokens.values())
    vocab  = len(tokens)
    lexdiv = float(count) / float(vocab)
    
    print((
        "Text corpus contains {} files\n"
        "Composed of {} paragraphs and {} sentences.\n"
        "{:0.3f} sentences per paragraph\n"
        "Word count of {} with a vocabulary of {}\n"
        "lexical diversity is {:0.3f}"
    ).format(
        fids, paras, sents, sperp, count, vocab, lexdiv
    ))

corpus_info(texts)    

def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
#    candidates = [' '.join(word for word, pos, chunk in group).lower()
#                  for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]

    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda x: x[2] != 'O') if key]

    return [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
            
def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
	# exclude candidates that are stop words or entirely punctuation
	punct = set(string.punctuation)
	stop_words = set(nltk.corpus.stopwords.words('english'))
	# tokenize and POS-tag words
	tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
		for sent in nltk.sent_tokenize(text)))
		# filter on certain POS tags and lowercase all words
	candidates = [word.lower() for word, tag in tagged_words if tag in good_tags and word.lower() not in stop_words and not all(char in punct for char in word)]
	return candidates

def score_keyphrases_by_tfidf(texts, candidates='chunks'):    
    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts.fileids()]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts.fileids()]
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf, dictionary

