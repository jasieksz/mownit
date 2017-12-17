from connector import dbConnector
from persistenceManager import persistenceManager
import numpy as np
import time


class Engine():
    def __init__(self, data, termCountpath, idPath, vecPath, make=False):
        self.data = data  # (id, content)
        self.pm = persistenceManager()
        self.termCountPath = termCountpath
        self.idpath = idPath
        self.vecpath = vecPath

        if make:
            self.termSet = sorted(self.makeTermSet()) # <term>
            self.termCountMap = self.makeTermCount()  # <term, count>
            self.ids, self.idVectorMatrix = self.makeIdVector()  # <id> , <vec>
        else:
            self.termSet = sorted(set(self.termCountMap.keys()))  # <term>
            self.termCountMap = self.readTermCount()
            self.ids, self.idVectorMatrix = self.readIdVector()

    def makeTermSet(self):
        ts = set()
        docs = [d[1] for d in self.data]
        for doc in docs:
            words = doc.split()
            for word in words:
                ts.add(word)
        return ts

    def makeTermCount(self):
        termCount = {k: 0 for k in self.termSet}
        docs = [d[1] for d in self.data]
        for doc in docs:
            words = doc.split()
            for word in words:
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
        print("IDS & VEC.T : ", self.ids.shape, b.shape )

        i = 0
        for k, v in zip(self.ids, b):
            if len(v) != s:
                i += 1
        print("WRONG VEC LEN COUNT : ", i)


    def termIdf(self, term):
        n = len(self.ids)
        count = 0 # w ilu doc jest term
        docs = [d[1] for d in self.data]
        for doc in docs:
            words = doc.split()
            if term in words:
                count += 1
        return np.log(n / count)

    def applyIdf(self):
        row = 0
        for term in self.termSet:
            tidf = self.termIdf(term)
            for col in range(self.idVectorMatrix.shape[1]):
                self.idVectorMatrix[row][col] *= tidf
            row += 1



term_count_csv_path = 'd:/db/persistence5k/term_count.csv'
vec_mat_path = 'd:/db/persistence5k/vec_matrix.npy'
idf_vec_mat_path = 'd:/db/persistence5k/idf_vec_matrix.npy'
id_mat_path = 'd:/db/persistence5k/id_matrix.npy'



def create():
    db = dbConnector('d:/db/clean.db')
    pm = persistenceManager()
    data = db.get_articles('''SELECT id, content FROM articles''', 5000)
    en = Engine(data, term_count_csv_path, id_mat_path, vec_mat_path, True)
    en.print()
    pm.map2csv(term_count_csv_path, ['term', 'count'], en.termCountMap)
    np.save(vec_mat_path, en.idVectorMatrix)
    np.save(id_mat_path, en.ids)
    en.applyIdf()
    np.save(idf_vec_mat_path, en.idVectorMatrix)


def read():
    db = dbConnector('d:/db/clean.db')
    pm = persistenceManager()
    data = db.get_articles('''SELECT id, content FROM articles''', 5000)
    en = Engine(data, term_count_csv_path, id_mat_path, vec_mat_path, False)
    en.print()


if __name__ == '__main__':
    start = time.clock()
    create()
    end = time.clock()
    print(end - start)
