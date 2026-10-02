"""Entity katmanı: bu işlem bu kullanıcı / bu cihaz için tuhaf mı?

Kullanıcı tarafı: tutarın kişisel geçmişten sapması, yeni cihaz, kullanıcının cihaz sayısı.
Cihaz tarafı: cihazın daha önce kaç farklı kullanıcı ve kartla görüldüğü (cihaz paylaşımı).
Kullanıcının ilk işleminde kişisel bileşenler 0 (nötr); "yeni kullanıcı" bilgisi context motoruna bırakılır.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from fraud_platform.detection.base import WeightedComponentDetector, saturate


class EntityDetector(WeightedComponentDetector):
    name = "entity"

    TEXTS: ClassVar[dict[str, str]] = {
        "amount_deviation": "Tutar kullanıcının geçmiş ortalamasının {ratio:.1f} katı ({amount:.2f} vs {avg:.2f} USD)",
        "new_device": "Geçmişi olan kullanıcı bu cihazı ilk kez kullanıyor",
        "user_devices": "Kullanıcı daha önce {n_devices:.0f} farklı cihaz kullanmış",
        "device_shared_uids": "Bu cihazda daha önce {device_uids:.0f} farklı kullanıcı görülmüş",
        "device_shared_cards": "Bu cihazda daha önce {device_cards:.0f} farklı kart görülmüş",
    }
    VALUE_OF: ClassVar[dict[str, str]] = {
        "amount_deviation": "ratio", "new_device": "new_device", "user_devices": "n_devices",
        "device_shared_uids": "device_uids", "device_shared_cards": "device_cards",
    }

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.caps
        all_components = {
            "amount_deviation": (df["amount_user_zscore"].abs().clip(upper=c["amount_z"]) / c["amount_z"]).fillna(0.0),
            "new_device": df["is_new_device_for_user"].astype("float64"),
            "user_devices": saturate(df["user_n_devices"], c["user_devices"]),
            "device_shared_uids": saturate(df["device_n_uids"], c["device_shared"]),
            "device_shared_cards": saturate(df["device_n_cards"], c["device_shared"]),
        }
        return pd.DataFrame({k: all_components[k] for k in self.weights}, index=df.index)

    def component_values(self, row: pd.Series) -> dict[str, object]:
        avg = row["user_avg_amount"]
        return {
            "amount": row[self.amount_col],
            "avg": avg if pd.notna(avg) else np.nan,
            "ratio": row["amount_to_user_avg"] if pd.notna(row["amount_to_user_avg"]) else np.nan,
            "new_device": int(row["is_new_device_for_user"]),
            "n_devices": row["user_n_devices"],
            "device_uids": row["device_n_uids"],
            "device_cards": row["device_n_cards"],
        }
