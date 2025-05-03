import os
from pymongo import MongoClient, errors
from .config import MONGO_URL

class DatabaseConnection:
    def __init__(self, mongo_url=None):
        self.mongo_url = mongo_url or MONGO_URL
        self.client = None
        self.databases = {}
        self.connect()

    def connect(self):
        try:
            self.client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping') 
        except errors.ConnectionFailure as e:
            print(f"Ошибка подключения к MongoDB серверу: {e}")
            self.client = None

    def add_database(self, alias, db_name):
        if self.client:
            self.databases[alias] = self.client[db_name]
        else:
            raise ConnectionError(f"no connection: cannot add {db_name}")

    def get_database(self, alias):
        return self.databases.get(alias)
