import sqlite3


class dbConnector():
    def __init__(self, path):
        self.path = path
        self.db = self.connect_db()
        self.connection = self.db[0]
        self.cursor = self.db[1]
        self.articles = self.get_articles()

    def connect_db(self):
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()
        return connection, cursor

    def get_articles(self):
        self.cursor.execute('''SELECT title, content FROM articles''')
        result = self.cursor.fetchall()
        articles = [t[1] for t in result]
        self.connection.close()
        return articles

#
# def csv_to_db(file, db):
#     connection = sqlite3.connect(db)
#     cursor = connection.cursor()
#     cursor.execute('''DROP TABLE IF EXISTS articles''')
#     cursor.execute('''CREATE TABLE IF NOT EXISTS articles
#                     (id, title, publication, author, date, year, month, url, content)''')
#     f = open(file, 'r', encoding="utf8")
#     next(f, None)
#     reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
#
#     for row in reader:
#         cursor.execute("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row[1:])
#
#     f.close()
#     connection.commit()
#     connection.close()
