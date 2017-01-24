#text = 'I often apply natural language processing for purposes of automatically extracting structured information from unstructured (text) datasets. One such task is the extraction of important topical words and phrases from documents, commonly known as terminology extraction or automatic keyphrase extraction. Keyphrases provide a concise description of a document’s content; they are useful for document categorization, clustering, indexing, search, and summarization; quantifying semantic similarity with other documents; as well as conceptualizing particular knowledge domains.Despite wide applicability and much research, keyphrase extraction suffers from poor performance relative to many other core NLP tasks, partly because there’s no objectively “correct” set of keyphrases for a given document. While human-labeled keyphrases are generally considered to be the gold standard, humans disagree about what that standard is! As a general rule of thumb, keyphrases should be relevant to one or more of a document’s major topics, and the set of keyphrases describing a document should provide good coverage of all major topics. (They should also be understandable and grammatical, of course.) The fundamental difficulty lies in determining which keyphrases are the most relevant and provide the best coverage. As described in Automatic Keyphrase Extraction: A Survey of the State of the Art, several factors contribute to this difficulty, including document length, structural inconsistency, changes in topic, and (a lack of) correlations between topics.Automatic keyphrase extraction is typically a two-step process: first, a set of words and phrases that could convey the topical content of a document are identified, then these candidates are scored/ranked and the “best” are selected as a document’s keyphrases.A brute-force method might consider all words and/or phrases in a document as candidate keyphrases. However, given computational costs and the fact that not all words and phrases in a document are equally likely to convey its content, heuristics are typically used to identify a smaller subset of better candidates. Common heuristics include removing stop words and punctuation; filtering for words with certain parts of speech or, for multi-word phrases, certain POS patterns; and using external knowledge bases like WordNet or Wikipedia as a reference source of good/bad keyphrases.For example, rather than taking all of the n-grams (where 1 ≤ n ≤ 5) in this post’s first two paragraphs as candidates, we might limit ourselves to only noun phrases matching the POS pattern {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+} (a regular expression written in a simplified format used by NLTK’s RegexpParser()). This matches any number of adjectives followed by at least one noun that may be joined by a preposition to one other adjective(s)+noun(s) sequence, and results in the following candidates:Compared to the brute force result, which gives 1100+ candidate n-grams, most of which are almost certainly not keyphrases (e.g. “task”, “relative to”, “and the set”, “survey of the state”, …), this seems like a much smaller and more likely set of candidates, right? As document length increases, though, even the number of likely candidates can get quite large. Selecting the best keyphrase candidates is the objective of step 2.Researchers have devised a plethora of methods for distinguishing between good and bad (or better and worse) keyphrase candidates. The simplest rely solely on frequency statistics, such as TF*IDF or BM25, to score candidates, assuming that a document’s keyphrases tend to be relatively frequent within the document as compared to an external reference corpus. Unfortunately, their performance is mediocre; researchers have demonstrated that the best keyphrases aren’t necessarily the most frequent within a document. (For a statistical analysis of human-generated keyphrases, check out Descriptive Keyphrases for Text Visualization.) A next attempt might score candidates using multiple statistical features combined in an ad hoc or heuristic manner, but this approach only goes so far. More sophisticated methods apply machine learning to the problem. They fall into two broad categories.Unsupervised machine learning methods attempt to discover the underlying structure of a dataset without the assistance of already-labeled examples (“training data”). The canonical unsupervised approach to automatic keyphrase extraction uses a graph-based ranking method, in which the importance of a candidate is determined by its relatedness to other candidates, where “relatedness” may be measured by two terms’ frequency of co-occurrence or semantic relatedness. This method assumes that more important candidates are related to a greater number of other candidates, and that more of those related candidates are also considered important; it does not, however, ensure that selected keyphrases cover all major topics, although multiple variations try to compensate for this weakness.Essentially, a document is represented as a network whose nodes are candidate keyphrases (typically only key words) and whose edges (optionally weighted by the degree of relatedness) connect related candidates. Then, a graph-based ranking algorithm, such as Google’s famous PageRank, is run over the network, and the highest-scoring terms are taken to be the document’s keyphrases.The most famous instantiation of this approach is TextRank; a variation that attempts to ensure good topic coverage is DivRank. For a more extensive breakdown, see Conundrums in Unsupervised Keyphrase Extraction, which includes an example of a topic-based clustering method, the other main class of unsupervised keyphrase extraction algorithms (which I’m not going to delve into).Unsupervised approaches have at least one notable strength: No training data required! In an age of massive but unlabled datasets, this can be a huge advantage over other approaches. As for disadvantages, unsupervised methods make assumptions that don’t necessarily hold across different domains, and up until recently, their performance has been inferior to supervised methods. Which brings me to the next section.Supervised machine learning methods use training data to infer a function that maps a set of input variables called features to some desired (and known) output value; ideally, this function can correctly predict the (unknown) output values of new examples based on their features alone. The two primary developments in supervised approaches to automatic keyphrase extraction deal with task reformulation and feature design.Early implementations recast the problem of extracting keyphrases from a document as a binary classification problem, in which some fraction of candidates are classified as keyphrases and the rest as non-keyphrases. This is a well-understood problem, and there are many methods to solve it: Naive Bayes, decision trees, and support vector machines, among others. However, this reformulation of the task is conceptually problematic; humans don’t judge keyphrases independently of one another, instead they judge certain phrases as more key than others in a intrinsically relative sense. As such, more recently the problem has been reformulated as a ranking problem, in which a function is trained to rank candidates pairwise according to degree of “keyness”. The best candidates rise to the top, and the top N are taken to be the document’s keyphrases.The second line of research into supervised approaches has explored a wide variety of features used to discriminate between keyphrases and non-keyphrases. The most common are the aforementioned frequency statistics, along with a grab-bag of other statistical features: phrase length (number of constituent words), phrase position (normalized position within a document of first and/or last occurrence therein), and “supervised keyphraseness” (number of times a keyphrase appears as such in the training data). Some models take advantage of a document’s structural features — titles, abstracts, intros and conclusions, metadata, and so on — because a candidate is more likely to be a keyphrase if it appears in notable sections. Others are external resource-based features: “Wikipedia-based keyphraseness” assumes that keyphrases are more likely to appear as Wiki article links and/or titles, while phrase commonness compares a candidate’s frequency in a document with respect to its frequency in an external corpus. The list of possible features goes on and on.A well-known implementation of the binary classification method, KEA (as published in Practical Automatic Keyphrase Extraction), used TF*IDF and position of first occurrence (while filtering on phrase length) to identify keyphrases. In A Ranking Approach to Keyphrase Extraction, researchers used a Linear Ranking SVM to rank candidate keyphrases with much success (but failed to give their algorithm a catchy name).Supervised approaches have generally achieved better performance than unsupervised approaches; however, good training data is hard to find (although here’s a decent place to start), and the danger of training a model that doesn’t generalize to unseen examples is something to always guard against (e.g. through cross-validation).'
text = 'I often apply natural language processing for purposes of automatically extracting structured information from unstructured (text) datasets. One such task is the extraction of important topical words and phrases from documents, commonly known as terminology extraction or automatic keyphrase extraction. Keyphrases provide a concise description of a document’s content; they are useful for document categorization, clustering, indexing, search, and summarization; quantifying semantic similarity with other documents; as well as conceptualizing particular knowledge domains.'

texts = 'BEIJING--China abolished a quota system limiting its exports of rare-earth minerals due to a decision by the World Trade Organization that ruled against the policy, the commerce ministry said Tuesday.  "The government has had to take a full range of economic means to effectively manage and bring order to rare earth resources," the ministry said in a statement. "The abolition of the quota is to coordinate domestic and international markets."  The Wall Street Journal reported last week that China had abolished the quota.  The U.S., European Union and Japan in 2012 complained to the WTO that China was using the quota in violation of WTO rules. Beijing officially lost the dispute in 2013 over the 17 metals, which are used in high-technology industries such as smartphones and missile systems.  "As a WTO member, China has always respected WTO rules and ways to strengthen the protection of resources and the environment to achieve sustainable development," the ministry said. '

https://gist.github.com/ojedatony1616/e605222e4442da2db9e5d670d0f797b3



def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string
    
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
    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]

# Test
aaa = extract_candidate_chunks(text)

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates
    
# Test   
extract_candidate_words(text)    

import gensim


def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    import gensim, nltk
    
    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    return corpus_tfidf, dictionary
    
    
b, c =score_keyphrases_by_tfidf(doc_complete)

import pandas as pd


for k, v in c.items():
    print (k, v)

for v in b:
    print( v)
    
    print(b)

texts = doc_complete



import numpy as np
import pandas as pd

probMatrix = np.zeros(shape=(3,9))  # size of (num docs, k topics)

for doc_num, probs in enumerate(b):
    for k_index, prob in probs:
        probMatrix[doc_num, k_index] = prob

df_tfidf_matrix = pd.DataFrame(probMatrix)












###############################################################
def score_keyphrases_by_textrank(text, n_keywords=0.05):
    from itertools import takewhile, tee, izip
    import networkx, nltk
    
    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    
    return sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)

score_keyphrases_by_textrank(text)