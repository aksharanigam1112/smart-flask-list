from src.DB_helper.DB_configuration import configuration
from pymongo import MongoClient
import logging

class db_fecth(configuration):
    def __init__(self):
        try:
            self.client = MongoClient(self.url)
            self.db = self.client.get_database(self.db_name)
            logging.debug("connected to the database....")
        except Exception as err:
            logging.exception(err)
            self.client=None
            self.db=None

    def read_json(self, table_name):
        if table_name == "customers":
            return self.db.customers
        elif table_name == "transactions":
            return self.db.transactions
        elif table_name == "itemlist":
            return self.db.itemlist
        elif table_name == "category":
            return self.db.category
        elif table_name == "rta":
            return self.db.rta
        elif table_name == "Recent_purchases":
            return self.db.Recent_purchases

