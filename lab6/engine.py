from connector import dbConnector
import numpy as np

class Engine():
    def __init__(self, data):
        self.data = data  # (id, content)
        self.termCountMap = self.makeTermCount() # <term, count>
        self.termSet = sorted(set(self.termCountMap.keys())) # <term>
        self.idVectorMap = self.makeIdVector() # <id, vector>

    def makeTermCount(self):
        termCount = {}
        docs = [d[1] for d in self.data]
        for doc in docs:
            words = doc.split()
            for word in words:
                termCount[word] = termCount.get(word, 0) + 1
        return termCount

    def makeIdVector(self):
        idVector = {}
        for element in self.data:
            i = element[0]
            doc = element[1]
            vec = self.makeVector(doc)
            vec = normalizeVector(vec)
            idVector[i] = vec
        return idVector

    def makeVector(self, doc):
        termCount = {k: 0 for k in self.termSet}
        words = doc.split()
        for word in words:
            termCount[word] = termCount.get(word, 0) + 1
        return list(termCount.values())

    def makeIdVectorMatrix(self):
        a = []
        for k, v in self.idVectorMap.items():
            a.append(v)
        return np.array(a).transpose()


def normalizeVector(vector):
    norm = max(vector)
    return [v / norm for v in vector]


if __name__ == '__main__':
    db = dbConnector('persistence/clean.db')
    data = db.get_articles('''SELECT id, content FROM articles''', 1000)
    en = Engine(data)
    print(en.makeIdVectorMatrix().shape)
