import nltk
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from lrma import *
import math

class searcher():
    def __init__(self, ts, ivm, ids):
        self.termSet = ts
        self.idfVectorMatrix = ivm
        self.ids = ids

    def makeVector(self, words):
        vd = {k: 0 for k in self.termSet}
        for word in words:
            if word in vd:
                vd[word] = vd.get(word, 0) + 1
        res = list(vd.values())
        return res

    def query(self, terms, k):
        cor = []
        words = self.stem(terms)
        words = [w for w in words if not w in STOP_WORDS]
        q = self.makeVector(words)
        q = normalizeV(q)
        lrmaVecMat = self.idfVectorMatrix
        for j in range(lrmaVecMat.shape[1]): # col = 0 .. 1000 (n articles)
            d = self.idfVectorMatrix[:, j]
            d = normalizeV(d)
            corel = self.correlation(np.array(q), np.array(d))
            cor.append(corel)
        return self.ids[np.argmax(cor)]

    def correlation(self, q, d):
        l = q @ d
        m = len(q) * len(d)
        return l/m


    def stem(self, words):
        doc = words.split()
        stemmer = nltk.stem.PorterStemmer()
        return [stemmer.stem(token) for token in doc]


def normalizeV(vector):
    n = math.sqrt(sum([e**2 for e in vector]))
    return [e / n for e in vector]