import sqlite3
from sqlite3 import Connection
from queue import Queue
import os
from path import *

class Database:
    def __init__(self, db_name='image_cropratio.db', pool_size=5):
        self.db_name = db_name
        self.pool = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self.pool.put(self._create_connection())

    def _create_connection(self) -> Connection:
        return sqlite3.connect(self.db_name)

    def get_connection(self) -> Connection:
        return self.pool.get()

    def release_connection(self, conn: Connection):
        self.pool.put(conn)

    def execute(self, query, params=()):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        self.release_connection(conn)

    def fetchone(self, query, params=()):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()
        self.release_connection(conn)
        return result

    def fetchall(self, query, params=()):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()
        self.release_connection(conn)
        return result

db = Database()

def create_table():
    db.execute('''CREATE TABLE IF NOT EXISTS image_data (
                    id INTEGER PRIMARY KEY,
                    image_name TEXT NOT NULL UNIQUE,
                    value1 INTEGER,
                    value2 INTEGER,
                    value3 INTEGER,
                    value4 INTEGER
                  )''')

def add_image_data(image_name, value1, value2, value3, value4):
    db.execute('''INSERT INTO image_data (image_name, value1, value2, value3, value4)
                  VALUES (?, ?, ?, ?, ?)''', (image_name, value1, value2, value3, value4))


def verify_database_count():
    count = db.fetchone('SELECT COUNT(*) FROM image_data')[0]
    if count == 10000:
        print("The database contains exactly 10,000 elements.")
        return True
    else:
        print(f"The database contains {count} elements, which is not 10,000.")
        return False

def delete_database():
    db_name = 'image_cropratio.db'
    if os.path.exists(db_name):
        os.remove(db_name)
        print(f"Database {db_name} deleted.")
    else:
        print(f"Database {db_name} does not exist.")

def get_image_data(image_name):
    result = db.fetchone('''SELECT value1, value2, value3, value4 FROM image_data WHERE image_name = ?''', (image_name,))
    if result is not None:
        return result
    else:
        print(f"No data found for image name: {image_name}")
        print("not cropping the image")
        return [0, 0, 0, 0]

def create_table_values():
    with open(os.path.join(path, "images_box.txt"), "rb") as f:
        num_lines = sum(1 for _ in f)
    i = 0
    with open(os.path.join(path, "images_box.txt"), 'r') as box_file:
        for box_line in box_file:
            if ((i / num_lines) * 100) % 2 == 0:
                print('\r' + ' ' * 30, end='', flush=True)  # Clear the line
                print('\r', end='', flush=True)  # Move the cursor back to the
                print("exracting data... " + str((i / num_lines) * 100) + "%", end='', flush=True)
            # Process each line as needed
            parts = box_line.split()
            # Assuming the first part is the image name and the rest are bounding box coordinates
            image_name = parts[0]
            value1 = parts[1]
            value2 = parts[2]
            value3 = parts[3]
            value4 = parts[4]
            add_image_data(image_name, value1, value2, value3, value4)
            i += 1