"""Gerçek zamanlı (tek işlem) feature hesabı.

Ayrı bir "online" formül yazılmaz: işlemin uid/cihaz geçmişi depodan çekilir, işlem sona eklenir ve eğitimde
kullanılan pipeline.transform aynen çalıştırılır. tests/test_online.py offline ve online sonuçların aynı olduğunu doğrular.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_platform.features.pipeline import FeaturePipeline
from fraud_platform.store.profile_store import EntityProfileStore


class OnlineFeatureBuilder:
    def __init__(self, pipeline: FeaturePipeline, store: EntityProfileStore):
        self.pipeline = pipeline
        self.store = store
        self.time_col, self.id_col = pipeline.sort_by

    def build(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Ham işlem(ler) -> feature satır(lar)ı, girişteki sırayla."""
        # API'den eksik alanla gelebilir; pipeline'ın okuduğu kolonlar boş olarak tamamlanır.
        # np.nan (pd.NA değil): pd.NA kolonu object tipine çevirip sayısal dönüşümü bozuyor
        missing_cols = [c for c in self.pipeline.required_columns() if c not in tx]
        tx = tx.assign(**{c: np.nan for c in missing_cols}) if missing_cols else tx.copy()
        if self.id_col not in tx or tx[self.id_col].isna().any():
            missing = tx[self.id_col].isna() if self.id_col in tx else pd.Series(True, index=tx.index)
            start = self.store.next_id()
            tx.loc[missing, self.id_col] = range(start, start + int(missing.sum()))
        tx[self.id_col] = tx[self.id_col].astype("int64")

        keys = self.pipeline.transform(tx)[[self.id_col, "uid", "device_key"]]
        rows = []
        for _, k in keys.iterrows():
            one = tx[tx[self.id_col] == k[self.id_col]]
            history = self.store.history_for(
                k["uid"], k["device_key"], before_time=int(one[self.time_col].iloc[0]), before_id=int(k[self.id_col])
            )
            frame = pd.concat([history, one], ignore_index=True)
            out = self.pipeline.transform(frame)
            rows.append(out[out[self.id_col] == k[self.id_col]])
        result = pd.concat(rows, ignore_index=True)
        return result.set_index(self.id_col).loc[tx[self.id_col]].reset_index()
