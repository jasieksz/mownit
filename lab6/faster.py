import heapq

import nltk
import numpy as np
import time
import math
from connector import dbConnector
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, lil_matrix, diags
from sklearn.preprocessing import normalize

resultpath = 'd:/db/faster'
dbpath = 'd:/db/articles.db'


class SearchEngine:
    def __init__(self, records=1000, load=True):
        self.wnl = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        self.rawData = self.addIdToDbData(self.getArticlesFromDb(dbpath, records))
        self.tokens = set(np.load(resultpath + '/1k/tokens.npy')) if load else sorted(self.getTokens(self.rawData))
        self.termDocMat = np.load(resultpath + '/1k/termDoc.npy') if load else self.getTermDocumentMatrix([self.rawData[r][2] for r in self.rawData])
        self.idfTermDocMat = load_sparse_csr(resultpath + '/1k/idfTermDoc.npz') if load else self.applyIFTD(self.termDocMat)
        self.lraIdfTermDocMat = np.load(resultpath + '/1k/lra500.npy') if load else self.applyLRA(self.idfTermDocMat, 1000)

    #
    # Getting aritcles from database and adding IDs
    #

    def getArticlesFromDb(self, path, records):
        db = dbConnector(path)
        return db.get_articles('''SELECT id, title, content FROM articles''', records)

    def addIdToDbData(self, data):
        result = {}
        myId = 0
        for r in data:
            result[myId] = r
            myId += 1
        return result

    #
    # text tokenize, lemmatize, stem, stopwords,
    #

    def getArticleTokens(self, data):
        return word_tokenize(data)

    def getTokens(self, data):
        word_bag = set()
        for record in data:
            title = self.getArticleTokens(data[record][1])
            title = self.cleanText(title)
            title = self.proccessText(title)
            article = self.getArticleTokens(data[record][2])
            article = self.cleanText(article)
            article = self.proccessText(article)

            [word_bag.add(t) for t in title]
            [word_bag.add(a) for a in article]
        return word_bag

    def cleanText(self, text):
        text = [re.sub("[^a-zA-Z]", "", t) for t in text]
        text = [t.lower() for t in text]
        text = [t for t in text if not t in STOP_WORDS]
        return text

    def proccessText(self, text):
        text = [self.wnl.lemmatize(t) for t in text]
        text = [self.stemmer.stem(t) for t in text]
        text = [t for t in text if len(t) > 1]
        return text

    #
    # text vector, term-document matrix, IDF
    #

    def getArticleVector(self, article):
        tokenDict = {t: 0 for t in self.tokens}
        article = self.getArticleTokens(article)
        article = self.cleanText(article)
        article = self.proccessText(article)
        for word in article:
            tokenDict[word] = tokenDict.get(word, 0) + 1
        return [v for v in tokenDict.values()]

    def getTermDocumentMatrix(self, data):
        a = []
        for article in data:
            vec = self.getArticleVector(article)
            a.append(vec)
        return np.array(a)

    def applyIFTD(self, data):
        transformer = TfidfTransformer(use_idf=True)
        return transformer.fit_transform(data)


    #
    # SVD, low rank approximation
    #

    def applyLRA(self, data, r):
        def reduce(r1, U1, s1, V1):
            s1 = s1[:r1]  # r first singular values
            U1 = U1[:, :r1]
            V1 = V1[:r1, :]
            return U1, s1, V1

        def compose(U2, s2, V2):
            D2 = np.diag(s2)
            return U2 @ D2 @ V2
        U, s, V = svds(data, k=r)
        result = compose(U, s, V)
        return result

    #
    # correlate
    #

    def correlation(self, q, d):
        l = q @ d
        m = len(q) * d.shape[0]
        return l/m


    def getBest(self, query, k=3):
        queryVec = np.array(self.getArticleVector(query)).reshape(-1, 1)
        queryVec = normalize(queryVec)
        correlations = []
        for i in range(self.idfTermDocMat.shape[0]):
            cor = self.correlation(queryVec, normalize(self.idfTermDocMat[i]))
            correlations.append(cor)
        return np.argmax(correlations), np.max(correlations)


    def normalizeV(self, vector):
        n = math.sqrt(sum([e**2 for e in vector]))
        return [e / n for e in vector]

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

if __name__ == '__main__':
    start = time.clock()
    q = 'Donald Trump Russia Election'
    google = SearchEngine(load=True, records=1000)
    ind, m = google.getBest(q)
    print(google.rawData[ind][0])
    end = time.clock()
    print(end - start)


