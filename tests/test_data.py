import numpy as np
import pandas as pd

from fraud_platform.data.loader import DataLoader, reduce_memory
from fraud_platform.data.quality import domain_variants, duplicate_columns, null_blocks
from fraud_platform.data.schema import SchemaInferer, SemanticType


def test_merge_keeps_all_transactions(settings):
    tx = pd.DataFrame({"TransactionID": [1, 2, 3], "TransactionAmt": [10.0, 20.0, 30.0]})
    idn = pd.DataFrame({"TransactionID": [2], "DeviceType": ["mobile"]})

    df = DataLoader(settings).merge(tx, idn)

    assert len(df) == 3
    assert df["has_identity"].tolist() == [0, 1, 0]
    assert df.loc[df["TransactionID"] == 2, "DeviceType"].item() == "mobile"


def test_reduce_memory_downcasts():
    df = pd.DataFrame({"a": np.array([1.5, 2.5]), "b": np.array([1, 2], dtype="int64")})
    out = reduce_memory(df)
    assert out["a"].dtype == "float32"
    assert out["b"].dtype == "int8"


def test_schema_rules(settings):
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "isFraud": [0, 1, 0, 0],
        "TransactionDT": [100, 200, 300, 400],
        "card1": [1001, 1002, 1003, 1001],   # sayısal ama override ile kategorik
        "C1": [1.0, 5.0, 2.0, 1.0],          # sayaç -> numeric
        "M1": ["T", "F", "T", None],         # 2 değer -> binary
        "P_emaildomain": ["gmail.com", "yahoo.com", "aol.com", None],
        "flat": [7, 7, 7, 7],
    })
    schema = SchemaInferer(settings.data, settings.schema_).infer(df)
    types = {name: c.semantic_type for name, c in schema.columns.items()}

    assert types["TransactionID"] == SemanticType.IDENTIFIER
    assert types["isFraud"] == SemanticType.TARGET
    assert types["TransactionDT"] == SemanticType.DATETIME
    assert types["card1"] == SemanticType.CATEGORICAL
    assert schema.columns["card1"].reason.startswith("override")
    assert types["C1"] == SemanticType.NUMERIC
    assert types["M1"] == SemanticType.BINARY
    assert types["P_emaildomain"] == SemanticType.CATEGORICAL
    assert types["flat"] == SemanticType.CONSTANT


def test_null_blocks_groups_identical_masks():
    df = pd.DataFrame({
        "V1": [1, None, 3, None],
        "V2": [5, None, 7, None],   # V1 ile aynı maske
        "V3": [None, 1, 2, 3],
    })
    blocks = null_blocks(df)
    assert len(blocks) == 1
    assert blocks[0]["columns"] == ["V1", "V2"]


def test_duplicate_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 2, 1]})
    assert duplicate_columns(df) == [["a", "b"]]


def test_domain_variants():
    s = pd.Series(["gmail.com", "gmail", "yahoo.com", "aol.com", None])
    assert domain_variants(s) == {"gmail": ["gmail", "gmail.com"]}
