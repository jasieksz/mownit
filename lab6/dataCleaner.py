import csv
import re
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import nltk
import time
import numpy as np


class dataCleaner():
    def __init__(self, data, cons=True, lem=True, stem=False, stopwords=STOP_WORDS):
        self.nlp = spacy.load('my_model')
        self.lematization = lem
        self.stemming = stem
        self.stopwords = stopwords
        self.data = data  ## (id, content)
        self.id_words_dict = {}
        self.token_set = set()
        self.token_dict = {}
        if cons:
            self.construct()
        else:
            self.construct_csv('persistence/clean_art.csv')

    def construct(self):
        self.id_words_dict = self.clean_all()
        self.token_set = self.word_set()
        self.token_dict = self.word_dict()

    def construct_csv(self, path):
        print("CONSTRUCTOR CSV")
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:  # row = 1 article
                k = row['id']
                v = row['content']
                self.id_words_dict[k] = v.split()
        self.token_set = self.word_set()
        self.token_dict = self.word_dict()
        print("TOKEN SET : ", len(self.token_set), " TOKEN DICT : ", len(self.token_dict))


    def clean_all(self):
        res = {}
        for doc in self.data:
            dirty = self.get_words(doc[1])
            clean = self.clean_words(dirty)
            res[doc[0]] = clean
        return res

    def get_words(self, raw_text):
        text = raw_text
        letters_only = re.sub("[^a-zA-Z]", " ", text)
        letters_only = re.sub(' {2}', ' ', letters_only)
        return letters_only.lower()

    def clean_words(self, words):
        self.stopwords.add(' ')
        self.stopwords.add('  ')
        if self.lematization:
            words = self.lematize(words)
        if self.stemming:
            words = self.stem(words)
        if not self.stemming and not self.lematization:
            words = words.split()
        return [w for w in words if not w in self.stopwords]

    def word_set(self):
        result = set()
        for doc in self.data:
            for w in self.id_words_dict[doc[0]]:
                result.add(w)
        return sorted(result)

    def bag_of_words(self, doc):
        bag = {k: 0 for k in self.token_set}
        for w in self.id_words_dict[doc[0]]:
            bag[w] = bag.get(w, 0) + 1
        return bag

    def word_dict(self):
        dict = {k: 0 for k in self.token_set}
        for doc in self.data:
            for w in self.id_words_dict[doc[0]]:
                dict[w] = dict.get(w, 0) + 1
        return dict

    def stem(self, words):
        doc = words.split()
        stemmer = nltk.stem.PorterStemmer()
        return [stemmer.stem(token) for token in doc]

    def lematize(self, words):
        doc = self.nlp(words)
        return [token.lemma_ for token in doc]

    def persist_dict(self, path):
        with open(path, 'w', newline='') as f:
            fn = ['word', 'count']
            writer = csv.DictWriter(f, fieldnames=fn)
            writer.writeheader()
            for t in self.token_dict.items():
                writer.writerow({'word': t[0], 'count': t[1]})

    def persist_clean_articles(self, path):
        with open(path, 'w', newline='') as f:
            fn = ['id', 'content']
            writer = csv.DictWriter(f, fieldnames=fn)
            writer.writeheader()
            for k, v in self.id_words_dict.items():
                writer.writerow({'id': k, 'content': join_words(v)})

    def vec_by_doc(self):
        a = []
        for doc in self.data:
            b = list(self.bag_of_words(doc).values())
            a.append(b)
        return np.array(a).transpose()

    def persist_vec_doc(self, path):
        with open(path, 'w', newline='') as f:
            fn = ['id', 'content']
            writer = csv.DictWriter(f, fieldnames=fn)
            writer.writeheader()
            for k, v in self.id_words_dict.items():
                bag = {k: 0 for k in self.token_set}
                for w in v:
                    bag[w] = bag.get(w, 0) + 1
                writer.writerow({'id': k, 'content': "".join([str(e) for e in list(bag.values())])})

def join_words(words):
    return " ".join(words)
