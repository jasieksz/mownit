import time
from db_test import dbConnector
from dataCleaner import dataCleaner


def run():
    start = time.clock()
    db = dbConnector('persistence/news.db')
    end = time.clock()
    print("DB connection and fetch articles | ", end - start)

    start = time.clock()
    dc = dataCleaner(db.articles)
    dc.persist_dict('persistence/word_dict.csv')
    end = time.clock()
    print("Parce data and persist | ", end - start)


if __name__ == '__main__':
    run()