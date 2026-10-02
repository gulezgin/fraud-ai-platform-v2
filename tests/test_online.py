import pandas as pd
from helpers import make_transactions

from fraud_platform.features.online import OnlineFeatureBuilder
from fraud_platform.features.pipeline import build_pipeline
from fraud_platform.store.profile_store import EntityProfileStore


def test_online_features_match_offline(settings):
    """Train/serve tutarlılığı: depodan geçmişle hesaplanan feature'lar toplu hesapla birebir aynı olmalı."""
    raw = make_transactions()
    pipe = build_pipeline(settings)
    offline = pipe.fit_transform(raw).set_index("TransactionID")

    store = EntityProfileStore.from_features(
        offline.reset_index(), pipe.required_columns(), "TransactionDT", "TransactionID", "TransactionAmt"
    )
    online = OnlineFeatureBuilder(pipe, store)

    feats = [f for f in pipe.feature_names if f != "null_count"]
    for tx_id in [1000, 1040, 1079]:
        got = online.build(raw[raw["TransactionID"] == tx_id]).set_index("TransactionID")
        expected = offline.loc[[tx_id], feats]
        pd.testing.assert_frame_equal(got[feats], expected, check_dtype=False)


def test_online_handles_missing_id_and_columns(settings):
    raw = make_transactions()
    pipe = build_pipeline(settings)
    offline = pipe.fit_transform(raw)
    store = EntityProfileStore.from_features(offline, pipe.required_columns(), "TransactionDT", "TransactionID", "TransactionAmt")

    # 5. gün, D1=4 -> başlangıç günü 1; geçmişteki 1. gün işlemleri (D1=0) ile aynı uid
    tx = pd.DataFrame([{"TransactionDT": 86400 * 5, "TransactionAmt": 120.0, "ProductCD": "W", "card1": 1, "addr1": 315.0, "D1": 4.0}])
    out = OnlineFeatureBuilder(pipe, store).build(tx)

    assert out["TransactionID"].iloc[0] == store.next_id()
    assert out["user_tx_count"].iloc[0] > 0   # aynı uid'nin geçmişi bulundu


def test_profile_summary(settings):
    raw = make_transactions()
    pipe = build_pipeline(settings)
    offline = pipe.fit_transform(raw)
    store = EntityProfileStore.from_features(offline, pipe.required_columns(), "TransactionDT", "TransactionID", "TransactionAmt")

    uid = offline["uid"].iloc[0]
    p = store.profile(uid)
    assert p["known"] and p["n_tx"] == int((offline["uid"] == uid).sum())
    assert store.profile("yok")["known"] is False
