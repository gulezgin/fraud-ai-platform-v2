"""Entity profil deposu (Repository): geçmiş işlemleri uid ve cihaz anahtarına göre indeksli tutar.

Gerçek zamanlı skorlamada tek bir işlem gelir; entity / velocity / cihaz paylaşımı feature'ları geçmişe ihtiyaç duyar.
Depo, kullanıcının ve cihazın önceki işlemlerini sadece pipeline'ın okuduğu ham kolonlarla saklar.
Yeni işlem bu geçmişin sonuna eklenip AYNI feature pipeline'dan geçirilir (bkz. features/online.py);
böylece eğitimdeki ve API'deki feature hesabı tasarım gereği birebir aynıdır.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["uid", "device_key"]


class EntityProfileStore:
    def __init__(self, history: pd.DataFrame, time_col: str, id_col: str, amount_col: str):
        self.time_col, self.id_col, self.amount_col = time_col, id_col, amount_col
        self.history = history.sort_values([time_col, id_col], kind="stable").reset_index(drop=True)
        self._reindex()

    def _reindex(self) -> None:
        self._by_uid = self.history.groupby("uid", sort=False).indices
        self._by_device = self.history.groupby("device_key", sort=False).indices   # boş anahtar indekslenmez

    @classmethod
    def from_features(cls, df: pd.DataFrame, columns: list[str], time_col: str, id_col: str, amount_col: str):
        cols = list(dict.fromkeys([*columns, *KEYS]))
        return cls(df[cols].copy(), time_col, id_col, amount_col)

    def __len__(self) -> int:
        return len(self.history)

    def next_id(self) -> int:
        return int(self.history[self.id_col].max()) + 1 if len(self) else 0

    def history_for(self, uid: str, device_key: str | None, before_time: int, before_id: int | None = None) -> pd.DataFrame:
        """Kullanıcının ve cihazın bu işlemden ÖNCEKİ kayıtları."""
        empty = np.array([], dtype=int)
        by_device = self._by_device.get(device_key, empty) if pd.notna(device_key) else empty
        idx = np.union1d(self._by_uid.get(uid, empty), by_device)
        h = self.history.iloc[idx]
        t = h[self.time_col]
        before = t < before_time
        if before_id is not None:
            before |= (t == before_time) & (h[self.id_col] < before_id)
        return h[before]

    def profile(self, uid: str) -> dict:
        """Agent'ların ve /explain'in kullandığı okunabilir kullanıcı özeti."""
        idx = self._by_uid.get(uid)
        if idx is None:
            return {"uid": uid, "known": False, "n_tx": 0}
        h = self.history.iloc[idx]
        amt, t = h[self.amount_col], h[self.time_col]
        return {
            "uid": uid,
            "known": True,
            "n_tx": len(h),
            "avg_amount": round(float(amt.mean()), 2),
            "max_amount": round(float(amt.max()), 2),
            "first_seen_day": round(float(t.min()) / 86400, 2),
            "last_seen_day": round(float(t.max()) / 86400, 2),
            "n_devices": int(h["device_key"].nunique()),
            "devices": h["device_key"].dropna().value_counts().head(5).index.tolist(),
        }

    def add(self, rows: pd.DataFrame) -> None:
        """Skorlanan işlemleri depoya ekler (online öğrenme olmadan sadece geçmişi büyütür)."""
        cols = self.history.columns
        self.history = pd.concat([self.history, rows.reindex(columns=cols)], ignore_index=True)
        self._reindex()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"time_col": self.time_col, "id_col": self.id_col, "amount_col": self.amount_col}
        self.history.to_parquet(path, index=False)
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EntityProfileStore:
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return cls(pd.read_parquet(path), meta["time_col"], meta["id_col"], meta["amount_col"])
