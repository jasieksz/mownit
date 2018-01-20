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
        self.tokens = set(np.load(resultpath + '/5k/tokens.npy')) if load else sorted(self.getTokens(self.rawData))
        # self.termDocMat = np.load(resultpath + '/5k/termDoc.npy') if load else self.getTermDocumentMatrix([self.rawData[r][2] for r in self.rawData])
        # self.idfTermDocMat = load_sparse_csr(resultpath + '/5k/idfTermDoc.npz') if load else self.applyIFTD(self.termDocMat)
        self.lraIdfTermDocMat = np.load(resultpath + '/5k/lra4000.npy')# if load else self.applyLRA(self.idfTermDocMat,1000)

    #
    # Getting aritcles from database and adding IDs
    #

    def getArticlesFromDb(self, path, records):
        db = dbConnector(dbpath)
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
            if word in tokenDict.keys():
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
        l = np.dot(q, d)
        m = q.shape[0] * d.shape[0]
        return l / m
        # return np.correlate(q, d)

    def getBest(self, query, data, k=3):
        start = time.clock()
        queryVec = np.array(self.getArticleVector(query))
        queryVec = queryVec / np.linalg.norm(queryVec)
        correlations = []
        for i in range(data.shape[0]):
            art = data[i]
            art = art / np.linalg.norm(art)
            cor = self.correlation(queryVec, art)
            correlations.append(cor)
        print("TIME : ",time.clock() - start)
        return heapq.nlargest(k, range(len(correlations)), correlations.__getitem__)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def printResult(ind, data):
    for i in ind:
        tid = data[i][0]
        db = dbConnector(dbpath)
        print(db.get_articles("SELECT id, title FROM articles WHERE id = '%s'" % str(tid), 1))

if __name__ == '__main__':
    start = time.clock()

    q1 = 'president donald trump election hilary clinton russia hack'
    q2 = 'nfl superbowl sport team'

    google = SearchEngine(load=True, records=5000)
    ind = google.getBest(q1, google.lraIdfTermDocMat, 8)
    ind2 = google.getBest(q2, google.lraIdfTermDocMat, 8)
    printResult(ind, google.rawData)
    print("\nQ2\n")
    printResult(ind2, google.rawData)
    end = time.clock()
    print(end - start)
