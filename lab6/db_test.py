import sqlite3
import csv
from bs4 import BeautifulSoup
import re
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


def csv_to_db(file, db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute('''DROP TABLE IF EXISTS articles''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS articles
                    (id, title, publication, author, date, year, month, url, content)''')
    f = open(file, 'r', encoding="utf8")
    next(f, None)
    reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)

    for row in reader:
        cursor.execute("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row[1:])

    f.close()
    connection.commit()
    connection.close()


def connect_db(path):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    return connection, cursor


def get_articles(path):
    con, cur = connect_db(path)
    cur.execute('''SELECT title, content FROM articles''')
    result = cur.fetchmany(5)
    titles = [t[0] for t in result]
    articles = [t[1] for t in result]
    con.close()
    return titles, articles


def article_to_words(raw_article):
    article = BeautifulSoup(raw_article, "lxml").get_text()
    art_letters = re.sub("[^a-zA-Z]", " ", article)
    words = art_letters.lower().split()
    return [w for w in words if not w in STOP_WORDS and len(w) > 2]

def gen_bag(articles):
    bag = {}
    for a in articles:
        words = article_to_words(a)
        for w in words:
            bag[w] = bag.get(w, 0) + 1
    return bag

def persist_bag(bag, path):
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        #writer.writeheader()
        writer.writerow(bag.items())


if __name__ == '__main__':
    titles, articles = get_articles('persistence/news.db')
    persist_bag(gen_bag(articles), 'persistence/bag.csv')
