import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import wfdb

class PTBXLQualityDataset(Dataset):
    def __init__(self, db, y, indices, cfg, augment=False):
        self.db = db.iloc[indices].reset_index(drop=True)
        self.y = torch.tensor(y[indices], dtype=torch.float32)
        self.data_root = cfg.data_root
        self.target_len = cfg.target_fs * cfg.target_sec
        self.augment = augment

    def __len__(self):
        return len(self.db)

    def _load_waveform(self, filename_hr):
        rel = str(filename_hr).replace("\\", "/").split(".")[0]
        candidates = [
            self.data_root / rel,
            self.data_root / "records500" / rel,
            self.data_root / "records100" / rel
        ]
        path = next((c for c in candidates if c.with_suffix(".hea").exists()), None)
        if not path: raise FileNotFoundError(f"Missing WFDB for {filename_hr}")

        sig, _ = wfdb.rdsamp(str(path))
        x = sig[:, :12].T.astype(np.float32)

        if x.shape[1] > self.target_len:
            x = x[:, :self.target_len]
        else:
            x = np.pad(x, ((0,0), (0, self.target_len - x.shape[1])), mode='constant')

        mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-6
        return (x - mu) / sd

    def __getitem__(self, i):
        x = self._load_waveform(self.db.iloc[i]["filename_hr"])
        label = self.y[i]
        if self.augment:
            if np.random.rand() < 0.5: x *= np.random.uniform(0.9, 1.1)
            if np.random.rand() < 0.3: x += np.random.normal(0, 0.01, x.shape)
            if np.random.rand() < 0.2: x = np.roll(x, np.random.randint(-25, 25), axis=1) 
        return torch.from_numpy(x), label
