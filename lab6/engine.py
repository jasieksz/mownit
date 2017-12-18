import gc

from connector import dbConnector
from persistenceManager import persistenceManager
from searcher import searcher
import numpy as np
import time
from spacy.lang.en.stop_words import STOP_WORDS
import nltk


class Engine():
    def __init__(self, data, termCountpath, idPath, vecPath, make=False):
        self.data = data  # (id, content)
        self.pm = persistenceManager()
        self.termCountPath = termCountpath
        self.idpath = idPath
        self.vecpath = vecPath

        if make:
            self.termSet = sorted(self.makeTermSet())  # <term>
            self.termCountMap = self.makeTermCount()  # <term, count>
            self.ids, self.idVectorMatrix = self.makeIdVector()  # <id> , <vec>
        else:
            self.termCountMap = self.readTermCount()
            self.termSet = sorted(set(self.termCountMap.keys()))  # <term>
            self.ids, self.idVectorMatrix = self.readIdVector()

    def makeTermSet(self):
        ts = set()
        docs = [d[1] for d in self.data]
        for doc in docs:
            words = doc.split()
            for word in words:
                if (len(word) > 1):
                    ts.add(word)
        return ts

    def makeTermCount(self):
        termCount = {k: 0 for k in self.termSet}
        docs = [d[1] for d in self.data]
        for doc in docs:
            words = doc.split()
            for word in words:
                if (len(word) > 1):
                    termCount[word] = termCount.get(word, 0) + 1
        return termCount

    def makeIdVector(self):
        ids = []
        vecs = []
        for element in self.data:
            i = element[0]
            doc = element[1]
            vec = self.makeVector(doc)
            ids.append(i)
            vecs.append(vec)
        vecs = np.array(vecs)
        vecs = np.transpose(vecs)
        return np.array(ids), vecs

    def makeVector(self, doc):
        termCount = {k: 0 for k in self.termSet}
        words = doc.split()
        for word in words:
            if (len(word) > 1):
                termCount[word] = termCount.get(word, 0) + 1
        res = list(termCount.values())
        return res

    def readTermCount(self):
        return self.pm.csv2map(self.termCountPath, ['term', 'count'])

    def readIdVector(self):
        i = np.load(self.idpath)
        v = np.load(self.vecpath)
        return i, v

    def print(self):
        print("PRINTER")

        s = len(self.termSet)
        b = np.transpose(self.idVectorMatrix)

        print("TERMCOUNT & SET : ", len(self.termCountMap), s)
        print("IDS & VEC.T : ", self.ids.shape, b.shape)

        i = 0
        for k, v in zip(self.ids, b):
            if len(v) != s:
                i += 1
        print("WRONG VEC LEN COUNT : ", i)

    def termIdf(self, term):
        n = len(self.ids)
        count = 0  # w ilu doc jest term
        docs = [d[1] for d in self.data]
        for doc in docs:
            words = doc.split()
            if term in words:
                count += 1
        return np.log(n / count)

    def applyIdf(self):
        res = []
        for t in zip(range(len(self.termSet)), self.termSet):
            # start = time.clock()
            tidf = self.termIdf(t[1])
            col = self.idVectorMatrix[t[0]] * tidf
            res.append(col)
            # end = time.clock()
            # print("TIDF MULT : ", end - start)
        res = np.array(res)
        # res = np.transpose(res)
        return res


term_count_csv_path = 'd:/db/persistence5k/term_count.csv'
vec_mat_path = 'd:/db/persistence5k/vec_matrix.npy'
idf_vec_mat_path = 'd:/db/persistence5k/idf_vec_matrix.npy'
id_mat_path = 'd:/db/persistence5k/id_matrix.npy'

term_count_csv_path2 = 'd:/db/persistence500/term_count.csv'
vec_mat_path2 = 'd:/db/persistence500/vec_matrix.npy'
idf_vec_mat_path2 = 'd:/db/persistence500/idf_vec_matrix.npy'
id_mat_path2 = 'd:/db/persistence500/id_matrix.npy'


def create():
    db = dbConnector('d:/db/clean.db')
    pm = persistenceManager()
    data = db.get_articles('''SELECT id, content FROM articles''', 1000)
    en = Engine(data, term_count_csv_path, id_mat_path, vec_mat_path, True)
    en.print()
    pm.map2csv(term_count_csv_path, ['term', 'count'], en.termCountMap)
    np.save(vec_mat_path, en.idVectorMatrix)
    np.save(id_mat_path, en.ids)
    idf = en.applyIdf()
    np.save(idf_vec_mat_path, idf)


def create2():
    db = dbConnector('d:/db/clean.db')
    pm = persistenceManager()
    data = db.get_articles('''SELECT id, content FROM articles''', 50)
    en = Engine(data, term_count_csv_path2, id_mat_path2, vec_mat_path2, True)
    en.print()
    pm.map2csv(term_count_csv_path2, ['term', 'count'], en.termCountMap)
    np.save(vec_mat_path2, en.idVectorMatrix)
    np.save(id_mat_path2, en.ids)
    idf = en.applyIdf()
    np.save(idf_vec_mat_path2, idf)


def read():
    db = dbConnector('d:/db/clean.db')
    pm = persistenceManager()
    data = db.get_articles('''SELECT id, content FROM articles''', 1000)
    en = Engine(data, term_count_csv_path, id_mat_path, vec_mat_path, False)
    en.print()


def searcherLoad():
    pm = persistenceManager()
    tokenSet = set(pm.csv2map(term_count_csv_path, ['term', 'count']).keys())
    ids = np.load(id_mat_path)
    q = 'bitcoin cryptocurrency speculation bubble wallstreet bank dollar'

    idfMat = np.load(idf_vec_mat_path)
    sr = searcher(tokenSet, idfMat, ids)
    r = sr.query(q, 1)
    print(r)
    gc.collect()

    lrma50 = np.load('d:/db/persistence5k/lrma50.npy')
    sr = searcher(tokenSet, lrma50, ids)
    r = sr.query(q, 1)
    print(r)
    gc.collect()

    lrma100 = np.load('d:/db/persistence5k/lrma100.npy')
    sr = searcher(tokenSet, lrma100, ids)
    r = sr.query(q, 1)
    print(r)
    gc.collect()

    lrma200 = np.load('d:/db/persistence5k/lrma200.npy')
    sr = searcher(tokenSet, lrma200, ids)
    r = sr.query(q, 1)
    print(r)
    gc.collect()

    lrma500 = np.load('d:/db/lrma500.npy')
    sr = searcher(tokenSet, lrma500, ids)
    r = sr.query(q, 1)
    print(r)
    gc.collect()

    lrma750 = np.load('d:/db/persistence5k/lrma750.npy')
    sr = searcher(tokenSet, lrma750, ids)
    r = sr.query(q, 1)
    print(r)
    gc.collect()

    # print("5k")
    # np.save('d:/db/persistence5k/lrma200.npy', compress(idfMatrix, 200))
    # print("100")
    # np.save('d:/db/persistence5k/lrma100.npy', compress(idfMatrix, 100))
    # print("2k")
    # np.save('d:/db/persistence5k/lrma750.npy', compress(idfMatrix, 750))
    # print("10k")
    # np.save('d:/db/persistence5k/lrma50.npy', compress(idfMatrix, 50))



def reduce(r, U, s, V):
    s = s[:r]  # r first singular values
    U = U[:, :r]
    V = V[:r, :]
    return U, s, V


def compose(U, s, V):
    D = np.diag(s)
    return U @ D @ V


def compress(data, k):
    U, s, V = np.linalg.svd(data)
    Ur, sr, Vr = reduce(k, U, s, V)
    result = compose(Ur, sr, Vr)
    return result


if __name__ == '__main__':
    start = time.clock()
    searcherLoad()
    end = time.clock()
    print(end - start)
