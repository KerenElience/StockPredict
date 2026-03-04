# precompute_hdf5.py
import sqlite3 as sql
import pandas as pd
import numpy as np
import lmdb, pickle
from tqdm import tqdm
from stockagent.utils.process import StockDataProcessor

def serialize_sample(X: np.ndarray, y: float) -> bytes:
    """将 (X, y) 序列化为 bytes，便于 LMDB 存储"""
    return pickle.dumps((X, y))

def compute_global_stats(db_path, start_date, end_date):
    conn = sql.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(stock_price_data_rm);")
    columns = cursor.fetchall()
    columns_names = [col[1] for col in columns][3:]
    query_sql = f"SELECT {",".join(columns_names)} FROM stock_price_data_rm WHERE date BETWEEN ? AND ?;"
    df = pd.read_sql(query_sql, conn, params=(start_date, end_date, ))
    conn.close()

    means = df.mean().to_list()
    std = df.std(ddof = 0)
    std = std.replace({0: 1e-8}).to_list()
    stats = {"columns": columns_names, "means": means, "stds": std}
    print(stats)
    return stats

def precompute_samples_to_lmdb(
    db_path: str,
    lmdb_path: str,
    window_size: int = 30,
    start_date: str = "2001-01-01",
    end_date: str = "2020-01-01",
    means: list = [],
    stds: list = [],
    samples_per_stock: int = 20,
    max_total_samples: int = 200_000,
    map_size_gb: int = 10  # LMDB 最大容量（GB）
):
    
    # Step 1: 获取有效 stock_id
    conn = sql.connect(db_path)
    query = "SELECT stock_id FROM stock_price_data_rm GROUP BY stock_id HAVING COUNT(*) >= ?"
    valid_stocks_df = pd.read_sql_query(query, conn, params=(window_size,))
    stock_ids = valid_stocks_df['stock_id'].tolist()
    conn.close()

    print(f"Found {len(stock_ids)} valid stocks.")

    # Step 2: 初始化 LMDB
    map_size = map_size_gb * 1024**3  # 转为字节
    env = lmdb.open(lmdb_path, map_size=map_size, metasync=False, sync=False, meminit=False)

    processor = StockDataProcessor(db_path, window_size, start_date, end_date, stock_ids=stock_ids)
    total_written = 0

    with env.begin(write=True) as txn:
        for stock_id in tqdm(stock_ids, desc="Processing stocks"):
            if total_written >= max_total_samples:
                break
            data = processor.generate_window(stock_id, samples_per_stock)
            if len(data) == 0:  
                continue
            for window_df in data:
                if len(window_df) < window_size:
                    continue
                if total_written >= max_total_samples:
                    break
                X, y = processor.process(window_df, means, stds)
                if X.empty or y.empty:
                    continue

                # 序列化并写入 LMDB
                key = f"{total_written:08d}".encode("ascii")  # '00000000', '00000001', ...
                value = serialize_sample(X.values.astype(np.float32), 
                                         y.values.astype(np.float32))
                txn.put(key, value)
                total_written += 1

        # 保存元信息
        meta = {
            "total_samples": total_written,
            "window_size": window_size,
            "n_features": X.shape[1] if total_written > 0 else 0,
            "sample_keys": [f"{i:08d}" for i in range(total_written)],
            "means": means,
            "stds": stds
        }
        txn.put(b"__meta__", pickle.dumps(meta))

    processor.close()
    env.close()
    print(f"✅ Successfully wrote {total_written} samples to {lmdb_path}")

if __name__ == "__main__":
    stats = compute_global_stats(
        db_path="./stock_data.db",
        start_date="2001-01-01",
        end_date="2020-01-01",
    )

    precompute_samples_to_lmdb(
        db_path="./stock_data.db",
        lmdb_path="./train_data.lmdb",
        window_size=30,
        start_date="2001-01-01",
        end_date="2020-01-01",
        means=stats["means"],
        stds = stats["stds"],
        samples_per_stock=200,
        max_total_samples=300_000,
        map_size_gb=5
    )
    precompute_samples_to_lmdb(
        db_path="./stock_data.db",
        lmdb_path="./valid_data.lmdb",
        window_size=30,
        start_date="2021-01-01",
        end_date="2025-10-30",
        means=stats["means"],
        stds = stats["stds"],
        samples_per_stock=20,
        max_total_samples=60_000,
        map_size_gb=2
    )
