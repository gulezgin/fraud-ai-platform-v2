"""Ham CSV'leri okuyup birleştirir, belleği küçültür ve parquet olarak saklar."""
from __future__ import annotations

import logging

import pandas as pd

from fraud_platform.config import Settings

logger = logging.getLogger(__name__)


def memory_gb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e9


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> sığan en küçük int tipi.

    float32'nin TransactionAmt'te 3 ondalık hanede kayıp yaratmadığı kontrol edildi (docs/notes.md).
    """
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


class DataLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.id_col = settings.data.id_column
        self.time_col = settings.data.time_column

    def read_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        tx = pd.read_csv(self.settings.transaction_path)
        idn = pd.read_csv(self.settings.identity_path)
        logger.info("transaction %s, identity %s okundu", tx.shape, idn.shape)
        return tx, idn

    def merge(self, tx: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
        # left join: identity sadece işlemlerin ~%25'inde var, inner join geri kalanını atardı.
        # bayrak join'den önce eklendi; indicator=True 435 kolonlu tabloyu parçalıyor (PerformanceWarning)
        idn = idn.assign(has_identity=1)
        df = tx.merge(idn, on=self.id_col, how="left", validate="one_to_one")
        df["has_identity"] = df["has_identity"].fillna(0).astype("int8")

        if len(df) != len(tx):
            raise ValueError(f"birleştirme satır sayısını değiştirdi: {len(tx)} -> {len(df)}")
        logger.info("birleşik tablo %s, identity oranı %.3f", df.shape, df["has_identity"].mean())
        return df

    def build(self) -> pd.DataFrame:
        """CSV -> birleştirme -> bellek küçültme -> merged.parquet"""
        tx, idn = self.read_raw()
        df = self.merge(tx, idn)

        before = memory_gb(df)
        df = reduce_memory(df)
        logger.info("bellek: %.2f GB -> %.2f GB", before, memory_gb(df))

        out = self.settings.merged_path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        logger.info("kaydedildi: %s", out)
        return df

    def load(self, sample: bool = False, columns: list[str] | None = None) -> pd.DataFrame:
        """merged.parquet'i okur. sample=True ise settings'teki dev_sample_size kadar örneklem döner."""
        path = self.settings.merged_path
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadı, önce `python scripts/prepare_data.py` çalıştırın")

        df = pd.read_parquet(path, columns=columns)
        n = self.settings.data.dev_sample_size
        if sample and n and n < len(df):
            df = df.sample(n, random_state=self.settings.data.random_state)
            if self.time_col in df.columns:
                df = df.sort_values(self.time_col)
            df = df.reset_index(drop=True)
        return df
