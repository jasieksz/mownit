import csv
from bs4 import BeautifulSoup
import re
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import nltk


class dataCleaner():
    def __init__(self, data, lem=True, stem=False, stopwords=STOP_WORDS):
        self.lematization = lem
        self.stemming = stem
        self.stopwords = stopwords
        self.data = data
        self.token_set = self.word_set()
        self.token_dict = self.word_dict()

    def get_words(self, raw_text):
        text = raw_text  # BeautifulSoup(raw_text, "lxml").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", text)
        words = letters_only.lower().split()
        return [w for w in words if not w in self.stopwords]

    def clean_words(self, words):
        if self.lematization:
            words = self.lematize(words)
        if self.stemming:
            words = self.stem(words)
        return words

    def word_set(self):
        result = set()
        for doc in self.data:
            raw_words = self.get_words(doc)
            words = self.clean_words(raw_words)
            for w in words:
                result.add(w)
        return sorted(result)

    def bag_of_words(self, doc):
        bag = {k: 0 for k in self.token_set}
        raw_words = self.get_words(doc)
        words = self.clean_words(raw_words)
        for w in words:
            bag[w] = bag.get(w, 0) + 1
        return bag

    def word_dict(self):
        dict = {}
        for doc in self.data:
            dict = dict_union(dict, self.bag_of_words(doc))
        return dict


    def stem(self, words):
        nlp = spacy.load('en')
        doc = nlp(words)
        stemmer = nltk.stem.PorterStemmer()
        return [stemmer.stem(token.norm_.lower()) for token in doc]

    def lematize(self, words):
        nlp = spacy.load('en')
        doc = nlp(words)
        return [token.lemma_ for token in doc]

    def persist_dict(self, path):
        with open(path, 'w', newline='') as f:
            fn = ['word', 'count']
            writer = csv.DictWriter(f, fieldnames=fn)
            writer.writeheader()
            for k, v in self.word_dict().items():
                writer.writerow({'word': k, 'count': v})


def dict_union(d1, d2):
    d3 = {}
    for k in d1.keys():
        d3[k] = d3.get(k, 0) + d1[k]
    for k in d2.keys():
        d3[k] = d3.get(k, 0) + d2[k]
    return d3
