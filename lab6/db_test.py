import sqlite3
import csv

def csv_to_db(file, db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS articles
                    (id, title, publication, author, date, year, month, url, content)''')
    f = open(file, 'r')
    next(f, None)
    reader = csv.reader(f)

    for row in reader:
        cursor.execute("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)

    f.close()
    connection.commit()
    connection.close()



if __name__ == '__main__':
    file = 'persistence/articles1.csv'
    db = 'persistence/news'
    csv_to_db(file, db)