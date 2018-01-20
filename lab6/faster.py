import nltk
import numpy as np
import time

from connector import dbConnector
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

resultpath = 'd:/db/faster'
dbpath = 'd:/db/articles.db'


class SearchEngine:
    def __init__(self, records=1000, load=True):
        self.wnl = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        self.rawData = self.addIdToDbData(self.getArticlesFromDb(dbpath, records))
        self.tokens = set(np.load(resultpath + '/5k/tokens.npy')) if load else sorted(self.getTokens(self.rawData))
        self.termDocMat = np.load(resultpath +'/5k/termDoc.npy') if load else self.getTermDocumentMatrix([self.rawData[r][2] for r in self.rawData])

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
    #
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
        return np.array(a).transpose()



if __name__ == '__main__':
    start = time.clock()
    google = SearchEngine(load=True, records=5000)
    end = time.clock()
    print(google.termDocMat.shape)
    print(end - start)
