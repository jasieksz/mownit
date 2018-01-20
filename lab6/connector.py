import sqlite3


class dbConnector():
    def __init__(self, path):
        self.path = path
        self.db = self.connect_db()
        self.connection = self.db[0]
        self.cursor = self.db[1]

    def connect_db(self):
        connection = sqlite3.connect(self.path)
        cursor = connection.cursor()
        return connection, cursor

    def get_articles(self, query, records):
        self.cursor.execute(query)
        result = self.cursor.fetchmany(records)
        self.connection.close()
        return result


if __name__ == '__main__':
    db = dbConnector('d:/db/articles.db')
    tid = 20050
    data = db.get_articles("SELECT id, title, content FROM articles WHERE id = '%s'" % str(tid), 1)
    data = data[0]
    print(data[0], data[1], '\n')
    print(data[2])