import pandas as pd
import numpy as np
import sqlite3 as sql
import talib.abstract as tlba
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from .log import logger
from .common import CPUS

class StockDataProcessor:
    def __init__(self, db_path, window_size: int = 10,
                 start_date: str = None, end_date: str = None,
                 seed: int = None, stock_ids: list = None):
        self.db_path = db_path
        self.ws = window_size
        self.start_date = start_date
        self.end_date = end_date

        self.conn = None

        self.stock_ids = stock_ids if stock_ids is not None else self.get_stock_ids()
        if seed:
            random.seed(seed)
        
        self.memeroy = set()

    def multiprocess(self, data_list, max_workers: int = -1):
        max_workers = CPUS if max_workers == -1 else max_workers
        stock_data = []
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            future_to_stock = {executor.submit(self.process, df) for df in data_list}
            for future in as_completed(future_to_stock):
                try:
                    data, label = future.result()
                    if not data.empty:
                        stock_data.append([data, label])
                except Exception as exc:
                    logger.error(f"{exc}")
        return stock_data

    def process(self, df: pd.DataFrame, means: list, stds: list):
        if df.empty:
            return df, []
        try:
            if "stock_id" in df.columns:
                df = df.drop("stock_id", axis = 1)
            if "date" in df.columns:
                df = df.drop("date", axis = 1)
            if "data_id" in df.columns:
                df = df.drop("data_id", axis = 1)
            ## z-score normalization
            df = self.normalize(df, means, stds)
            return df.iloc[:, :-4], df.iloc[:, -4:]
        except Exception as e:
            logger.error(f"Generate technical index occur error: {e}.")
            return pd.DataFrame(), []
    
    def normalize(self, df, means, stds):
        df = df.astype("float64")
        return (df - means)/stds

    def connect(self):
        try:
            self.conn = sql.connect(self.db_path)
            return True
        except sql.Error as e:
            logger.error(e)
            return False

    def get_stock_ids(self):
        query_sql = "SELECT stock_id FROM stock_price_data_rm GROUP BY stock_id HAVING count(*) >= ?;"
        if not self.conn:
            self.connect()
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(query_sql, (self.ws, ))
            return [row[0] for row in result.fetchall()]
        except sql.Error as e:
            logger.error(f"get id occur error : {e}")
            return []
    
    def get_stock_df(self, stock_id: int | str | list):
        if isinstance(stock_id, (int, str)):
            pass
        else:
            stock_id = ",".join(stock_id)
        query_sql = f"SELECT * FROM stock_price_data_rm WHERE stock_id IN (?)"
        if self.start_date is not None and self.end_date is not None:
            query_sql = f"{query_sql} AND date BETWEEN date(?) AND date(?)"
        if not self.conn:
            self.connect()
        try:
            df = pd.read_sql(query_sql, self.conn, params=(stock_id, self.start_date, self.end_date, ))
            # df = df[~df.apply(lambda x: (x == -99.0).any(), axis = 1)].reset_index(drop = True)
            return df
        except sql.Error as e:
            logger.error(f"Get {stock_id} df failed: {e}")
            return pd.DataFrame()

    def check_df_sufficiency(self, df: pd.DataFrame):
        return len(df) >= self.ws
    
    def generate_window(self, stock_id, batch_size):
        df = self.get_stock_df(stock_id)
        nrow = len(df)
        data = []
        if not self.check_df_sufficiency(df) or nrow - self.ws <=0:
            return data
        if batch_size >= nrow-self.ws:
            batch_size = max(1, (nrow-self.ws) // 4)
        start_idxs = np.random.randint(0, nrow -1, batch_size)
        start_idxs.sort()
        start_idxs = list(set(start_idxs))  #non-duplicate
        for start_idx in start_idxs:
            window_df = df.iloc[start_idx: start_idx+ self.ws, :]
            window_df.reset_index(drop = True, inplace=True)
            data.append(window_df)
        return data

    def close(self):
        if self.conn:
            self.conn.close()

