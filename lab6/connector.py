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