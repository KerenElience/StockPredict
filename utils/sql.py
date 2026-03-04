import sqlite3 as sql
import akshare as ak
import pandas as pd
import numpy as np
import random, time
import talib as tlb
from .log import logger
from .common import CPUS
from concurrent.futures import ThreadPoolExecutor, as_completed

class SQLData():
    def __init__(self, db_path: str = "./stock_data.db", start_time: str = "20010101", end_time: str = "20251030"):
        self.db_path = db_path
        self.stime = start_time
        self.etime = end_time
        self.timeout = 10.0
        self.conn = None

    def initial(self):
        shanghai_data = self.gather_codename("sh")
        shenzhen_data = self.gather_codename("sz")
        data = pd.concat([shanghai_data, shenzhen_data])
        if len(data) > 0:
            self.batch_insert_code_name(data)
            self.multifetch(data["code"], max_workers=16)
            self.batch_insert_stock_index()
        else:
            logger.error("Can't request any stock, please check your internet or `akshare` api setting.")
        return None

    def connect(self):
        try:
            self.conn = sql.connect(self.db_path)
            ## launch primary key restiction
            self.conn.execute("PRAGMA foreign_keys = ON;")
            return True
        except sql.Error as e:
            logger.error(f"Connected occur error: {e}")
            return False
        
    def close(self):
        if self.conn:
            self.conn.close()

    def creat_tb(self):
        if not self.conn:
            self.connect()

        create_stock_codes_sql = """
        CREATE TABLE IF NOT EXISTS stock_codes (
            stock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT UNIQUE NOT NULL,
            stock_name TEXT,
            created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_stock_data_sql = """
        CREATE TABLE IF NOT EXISTS stock_price_data (
            data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            date TEXT,
            open REAL,
            close REAL,
            low REAL,
            high REAL,
            amount REAL,
            created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (stock_id) REFERENCES stock_codes (stock_id) ON DELETE CASCADE,
            UNIQUE(stock_id, date)
        );
        """

        create_stock_price_rm_sql = """
        CREATE TABLE IF NOT EXISTS stock_price_data_rm (
            data_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            date TEXT,
            open REAL,
            close REAL,
            low REAL,
            high REAL,
            amount REAL,
            up_limit REAL, down_limit REAL,
            SMA_10 REAL, SMA_20 REAL, SMA_50 REAL, EMA_12 REAL, EMA_26 REAL, MACD REAL, MACD_Signal REAL, MACD_Hist REAL, RSI_14 REAL, ROC_10 REAL, MOM_10 REAL,
            ATR_14 REAL, STD_20 REAL, BB_b REAL, OBV REAL, AD REAL, ADOSC REAL,
            hl2 REAL, hlc3 REAL, ohlc4 REAL, Log_profit REAL, future_avg_return REAL, vol_ma5, vol_ratio REAL, 
            FOREIGN KEY (stock_id) REFERENCES stock_codes (stock_id) ON DELETE CASCADE,
            UNIQUE(stock_id, date)
        );
        """

        create_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_stock_code ON stock_codes(stock_code);",
            "CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_price_data(date);",
            "CREATE INDEX IF NOT EXISTS idx_stock_data_code_date ON stock_price_data(stock_id, date);",
        ]

        try:
            cursor = self.conn.cursor()
            
            # 创建主表
            cursor.execute(create_stock_codes_sql)
            logger.info("Primary table created successfully.")
            
            # 创建数据表
            cursor.execute(create_stock_data_sql)
            logger.info("Stock trade data table created successfully")
            
            # 创建技术指标数据表
            cursor.execute(create_stock_price_rm_sql)
            logger.info("Stock price data rm table created successfully")

            # 创建索引
            for index_sql in create_indexes_sql:
                cursor.execute(index_sql)
            logger.info("Index created successfully")
            
            self.conn.commit()
            return True
            
        except sql.Error as e:
            logger.error(f"Create table failed: {e}")
            return False
        
    def insert_code_name(self, code, name = None):
        if not self.conn:
            self.connect()

        insert_sql = """
        INSERT OR REPLACE INTO stock_codes
        (stock_code, stock_name)
        VALUES (?, ?);
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(insert_sql, (code, name))
            self.conn.commit()
            return cursor.lastrowid
        except sql.Error as e:
            logger.error(f"Insert stock code failed: {e}")
            return None

    def insert_stock_price_data(self, stock_code, trad_date, open, close, low, high, amount):
        if not self.conn:
            self.connect()
        
        insert_sql = """
        INSERT OR REPLACE INTO stock_price_data
        (stock_id, date, open, close, low, high, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        stock_id = self.get_stock_id(stock_code)
        if stock_id is None:
            return False
        try:
            cursor = self.conn.cursor()
            cursor.execute(insert_sql, (stock_id, stock_code, trad_date, open, close, low, high, amount))
            self.conn.commit()
            return True
        except sql.Error as e:
            logger.error(f"Insert stock price data failed: {e}")
            return None

    def get_stock_id(self, stock_code):
        query_sql = "SELECT stock_id FROM stock_codes WHERE stock_code = ?;"
        if not self.conn:
            self.connect()

        try:
            cursor = self.conn.cursor()
            cursor.execute(query_sql, (stock_code,))
            result = cursor.fetchone()
            self.conn.commit()
            if result is None:
                logger.error(f"No found {stock_code}'s id")
                return None
            else:
                return result[0]
        except sql.Error as e:
            logger.error(f"get id occur error {stock_code}: {e}")
            return None

    def batch_insert_code_name(self, data: pd.DataFrame):
        insert_sql = """
        INSERT OR REPLACE INTO stock_codes
        (stock_code, stock_name)
        VALUES (?, ?);
        """
        batch_data = [i for i in data.apply(lambda x: tuple(x), axis = 1)]
        try:
            cursor = self.conn.cursor()
            cursor.executemany(insert_sql, batch_data)
            self.conn.commit()
            return True
        except sql.Error as e:
            print(f"Batch insert failed: {e}")
            return False

    def batch_insert_stock_data(self,  stock_code, data: pd.DataFrame):
        stock_id = self.get_stock_id(stock_code)
        if stock_id is None:
            return False
        
        insert_sql = """
        INSERT OR REPLACE INTO stock_price_data 
        (stock_id, date, open, close, low, high, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        batch_data = []
        for row in range(len(data)):
            batch_data.append((
                stock_id, 
                data.loc[row, 'date'], 
                data.loc[row, 'open'], 
                data.loc[row, 'close'], 
                data.loc[row, 'low'], 
                data.loc[row, 'high'], 
                data.loc[row, 'amount']
            ))

        try:
            cursor = self.conn.cursor()
            cursor.executemany(insert_sql, batch_data)
            self.conn.commit()
            return True
        except sql.Error as e:
            print(f"Batch insert failed: {e}")
            return False
    
    def batch_insert_stock_index(self):
        query_sql = "SELECT stock_id, date, open, close, low, high, amount FROM stock_price_data WHERE open>0 AND close >0;"

        insert_sql = """
        INSERT OR REPLACE INTO stock_price_data_rm
        (stock_id, date, open, close, low, high, amount, up_limit, down_limit, SMA_10, SMA_20, SMA_50, EMA_12, 
        EMA_26, MACD, MACD_Signal, MACD_Hist, RSI_14, 
        ROC_10, MOM_10, ATR_14, STD_20, BB_b, OBV, AD, ADOSC,
        hl2, hlc3, ohlc4, Log_profit, future_avg_return, vol_ma5, vol_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
        df = pd.read_sql_query(query_sql, self.conn)
        for tmp in df.groupby("stock_id"):
            tmp = self.process(tmp[1])
            batch_data = []
            for row in range(len(tmp)):
                record = (
                    int(tmp.loc[row, "stock_id"]), str(tmp.loc[row, "date"]), float(tmp.loc[row, "open"]), 
                    float(tmp.loc[row, "close"]), float(tmp.loc[row, "low"]), float(tmp.loc[row, "high"]), float(tmp.loc[row, "amount"]),
                    float(tmp.loc[row, "up_limit"]), float(tmp.loc[row, "down_limit"]),
                    float(tmp.loc[row, 'SMA_10']), float(tmp.loc[row, 'SMA_20']), 
                    float(tmp.loc[row, 'SMA_50']), float(tmp.loc[row, 'EMA_12']), float(tmp.loc[row, 'EMA_26']), 
                    float(tmp.loc[row, 'MACD']), float(tmp.loc[row, 'MACD_Signal']), float(tmp.loc[row, 'MACD_Hist']),
                    float(tmp.loc[row, 'RSI_14']), float(tmp.loc[row, "ROC_10"]), float(tmp.loc[row, "MOM_10"]),
                    float(tmp.loc[row, 'ATR_14']), float(tmp.loc[row, 'STD_20']), float(tmp.loc[row, 'BB_b']),
                    float(tmp.loc[row, "OBV"]), float(tmp.loc[row, "AD"]), float(tmp.loc[row, "ADOSC"]),
                    float(tmp.loc[row, 'hl2']), float(tmp.loc[row, 'hlc3']), float(tmp.loc[row, 'ohlc4']), 
                    float(tmp.loc[row, 'Log_profit']), float(tmp.loc[row, "future_avg_return"]), float(tmp.loc[row, "vol_ma5"]), float(tmp.loc[row, "vol_ratio"])
                )
                batch_data.append(record)
            cursor.executemany(insert_sql, batch_data)
            self.conn.commit()
            
    def delet_stock_code(self, stock_code):
        if not self.conn:
            self.connect()

        delete_sql = "DELETE FROM stock_codes WHERE stock_code = ?"
        try:
            cursor = self.conn.cursor()
            cursor.execute(delete_sql, (stock_code, ))
            self.conn.commit()
        except sql.Error as e:
            logger.error(f"Delete {stock_code} failed: {e}")
        return None

    def gather_codename(self, region: str = "sh"):
        """
        From different trade institution gather stock's code and name.
        - region: choise from Union[`sh`, `sz`]
        """
        try:
            if region == "sh":
                data = ak.stock_info_sh_name_code(symbol = "主板A股")
                data = data.iloc[:, :2]
            else:
                data = ak.stock_info_sz_name_code(symbol = "A股列表")
                data = data[data["板块"] == "主板"].iloc[:, 1:3]
            data.columns = ["code", "name"]
            data["code"] = data["code"].apply(lambda x: f"{region}{x}")
        except Exception as e:
            logger.error(f"Failed to scripy the {region} data {e}")
            data = None
        finally:
            return data

    def process(self, df: pd.DataFrame):
        df = df.copy()
        df.reset_index(drop = True, inplace = True)

        ##增加涨停、跌停信号
        df['up_limit'] = ((df['close'] - df["close"].shift(1)) / df["close"].shift(1) >= 0.099) & (df['close'] == df['high'])
        df['down_limit'] = ((df["close"].shift(1) - df['close']) / df["close"].shift(1) >= 0.099) & (df['close'] == df['low'])
        ##常规技术指标
        df["SMA_10"] = tlb.SMA(df["close"], 10)
        df["SMA_20"] = tlb.SMA(df["close"], 20)
        df["SMA_50"] = tlb.SMA(df["close"], 50)
        
        df["EMA_12"] = tlb.EMA(df["close"], 12)
        df["EMA_26"] = tlb.EMA(df["close"], 26)
        macd, macd_signal, macd_hist = tlb.MACD(df["close"])
        df["MACD"] = macd
        df["MACD_Signal"] = macd_signal
        df["MACD_Hist"] = macd_hist

        ## momentum
        df["RSI_14"] = tlb.RSI(df["close"], 14)
        df["ROC_10"] = tlb.ROC(df["close"], 10)
        df["MOM_10"] = tlb.MOM(df["close"], 10)
        
        df["ATR_14"] = tlb.ATR(df['high'], df["low"], df["close"],14)
        df["STD_20"] = tlb.STDDEV(df["close"], 20)

        upper, middle, lower = tlb.BBANDS(df["close"], 20)
        df["BB_b"] = (df["close"] - lower)/(upper - lower)

        df["OBV"] = tlb.OBV(df["close"], df["amount"])
        df["AD"] = tlb.AD(df["high"], df["low"], df["close"], df["amount"])
        df["ADOSC"] = tlb.ADOSC(df["high"], df["low"], df["close"], df["amount"])

        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        ## predict store in last column
        df = self.add_future_avg_return(df)
        ### future 5days average profit
        df["vol_ma5"] = df["amount"].rolling(5).mean()
        df["vol_ratio"] = df["amount"]/(df["vol_ma5"] + 1e-8)
        df = df.dropna(axis=0, ignore_index=True).reset_index(drop = True)
        return df
    
    def add_future_avg_return(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        为每个交易日 t 添加标签：未来 horizon 日的平均日对数收益率
        
        Parameters:
            df: pd.DataFrame with columns ['stock_id', 'date', 'close']
            horizon: int, e.g., 5
        
        Returns:
            df with new column 'future_avg_return'
        """
        df = df.reset_index(drop=True)
        df['Log_profit'] = np.log(df['close']) - np.log(df.groupby('stock_id')['close'].shift(1))

        future_returns = []
        for i in range(1, horizon + 1):
            future_returns.append(df.groupby('stock_id')['Log_profit'].shift(-i))

        future_df = pd.concat(future_returns, axis=1)
        df['future_avg_return'] = future_df.mean(axis=1)
        return df

    def fetch(self, symbol):
        try:
            data = ak.stock_zh_a_hist_tx(symbol=symbol, 
                                         start_date=self.stime, 
                                         end_date=self.etime, 
                                         adjust = "qfq",
                                         timeout=self.timeout)
        except Exception as e:
            data = None
            logger.error(f"Occur unexpcepted error: {e}")
        time.sleep(random.choice(range(1, 5)))
        if data.empty:
            data = None
        return data

    def multifetch(self, symbol_iterable, max_workers: int = -1):
        max_workers = CPUS if max_workers == -1 else max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {executor.submit(self.fetch, symbol): symbol for symbol in symbol_iterable}
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    data = future.result()
                    if data is not None:
                        self.batch_insert_stock_data(stock, data)
                        logger.info(f"Scripted and store {stock} data")
                    else:
                        self.delet_stock_code(stock)
                except Exception as exc:
                    logger.error(f'{stock} generated an exception: {exc}')
        return None
    
    def update(self, end_date: str = "20500101"):
        if not self.conn:
            self.connect()
        ## 从stock_price_data获取所有symbol数据
        query_sql = "SELECT stock_id, date FROM stock_price_data_rm;"
        index_sql = "SELECT stock_id, stock_code FROM stock_codes;"
        df = pd.read_sql(query_sql, self.conn)
        id_codes = pd.read_sql(index_sql, self.conn)
        id_codes = id_codes.drop_duplicates("stock_id")
        ## 获取symbol最后日期
        df = df.groupby("stock_id").agg("max").reset_index()
        df = pd.merge(df, id_codes, how = "left", on = "stock_id",)
        ## 获取更新数据
        for row in range(len(df)):
            code = df.loc[row, "stock_code"]
            data = ak.stock_zh_a_hist_tx(symbol=code, 
                                         start_date=df.loc[row, "date"],
                                         end_date=end_date,
                                         adjust="qfq")
            self.batch_insert_stock_data(code, data)
        ## 在stock_price_data_rm中更新symbol对应的数据
        logger.info("Re-Building rm price data...")
        self.batch_insert_stock_index()
        logger.info("Completed update.")
        self.close()

if __name__ == "__main__":
    sqldata = SQLData()
    sqldata.connect()
    sqldata.creat_tb()
    sqldata.initial()
    sqldata.close()