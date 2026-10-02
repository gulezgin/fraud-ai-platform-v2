"""FeatureAgent: işlemin feature'larını profil deposundaki geçmişle üretir; kullanıcı profilini paylaşır (Adım 3)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from fraud_platform.agents.base import BaseAgent, Capability
from fraud_platform.agents.message import Message
from fraud_platform.features.online import OnlineFeatureBuilder

KEY_FEATURES = ["user_tx_count", "amount_to_user_avg", "tx_count_1h", "tx_count_24h", "device_n_uids",
                "is_new_device_for_user", "is_first_tx", "local_hour"]


class FeatureAgent(BaseAgent):
    name = "feature_agent"
    description = "Feature üretir (geçmiş profil deposundan), kullanıcı profilini paylaşır"

    def __init__(self, online: OnlineFeatureBuilder):
        super().__init__()
        self.online = online

    def capabilities(self) -> dict[str, Capability]:
        return {
            "build_features": Capability("İşlemin 55 feature'ını kullanıcı ve cihaz geçmişiyle üretir",
                                         requires=("validate_transaction",), inputs=("transaction",),
                                         provides=("features", "uid", "key_features"), needs_transaction=True),
            "get_entity_profile": Capability("Bir kullanıcının (uid) geçmiş davranış özetini verir", inputs=("uid",),
                                             provides=("profile",)),
        }

    def on_build_features(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        feats = self.online.build(pd.DataFrame([payload["transaction"]]))
        row = feats.iloc[0]
        return {"features": feats, "uid": row["uid"],
                "key_features": {k: (None if pd.isna(row[k]) else round(float(row[k]), 3)) for k in KEY_FEATURES}}

    def on_get_entity_profile(self, payload: dict[str, Any], msg: Message) -> dict[str, Any]:
        return {"profile": self.online.store.profile(payload["uid"])}
