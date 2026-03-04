import numpy as np
import lmdb, pickle
from torch.nn import Identity
from torch.utils.data import Dataset
from .log import logger
from typing import Callable

class StockDataset(Dataset):
    def __init__(self, lmdb_path: str, 
                 transform: Callable = Identity(),
                ):
        self.lmdb_path = lmdb_path
        self._env = None
        with lmdb.open(lmdb_path, readonly = True, lock = False, ) as env:
            with env.begin() as txn:
                meta = pickle.loads(txn.get(b"__meta__"))
                self.total_samples = meta["total_samples"]
                self.window_size = meta["window_size"]
                self.n_features = meta["n_features"]
                self.means = meta["means"]
                self.stds = meta["stds"]
                # 注意：不再保存 keys 列表（避免大列表 pickle 开销）
        logger.info(f"Loaded LMDB: {self.total_samples} samples, shape=({self.window_size}, {self.n_features})")

        self.transform = transform

    def _get_env(self):
        if not hasattr(self, "_env") or self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly = True, 
                                  lock = False, readahead = False, meminit = False)
        return self._env

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        env = self._get_env()
        key = f"{index:08d}".encode("ascii")
        with env.begin(write = False, buffers = True) as txn:
            data = txn.get(key)
        if data is None:
            raise IndexError(f"Sample {index} not found")
        X, y = pickle.loads(data)
        return self.transform(X), self.transform(y)
    
    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


## CNN only predict the lastest day profit.
class CNNDataset(Dataset):
    def __init__(self):
        super().__init__()
        
class LSTMDataset(Dataset):
    def __init__(self):
        super().__init__()

class TransformerDataset(Dataset):
    def __init__(self):
        super().__init__()

class RealTimeDataset(Dataset):
    def __init__(self):
        super().__init__()
