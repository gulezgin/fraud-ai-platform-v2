import numpy as np
import pandas as pd


def make_transactions(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Pipeline'ın okuduğu kolonları içeren küçük sentetik işlem tablosu."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(1000, 1000 + n), "isFraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 4, n)),
        "TransactionAmt": rng.uniform(5, 500, n).round(2).astype("float32"),
        "ProductCD": rng.choice(["W", "C"], n), "card1": rng.choice([1, 2, 3], n), "card2": 100.0,
        "card4": "visa", "card5": 226.0, "card6": "debit", "addr1": rng.choice([315.0, 204.0], n), "addr2": 87.0,
        "dist1": np.nan, "D1": 0.0, "D4": 0.0, "D11": 0.0, "D15": 0.0,
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n), "R_emaildomain": None,
        "M1": "T", "M4": "M0", "M6": "F", "M7": None, "DeviceType": rng.choice(["mobile", None], n),
        "DeviceInfo": rng.choice(["ios device", "windows", None], n), "id_19": 100.0, "id_20": 200.0,
        "id_31": "chrome", "id_33": None, "C1": rng.integers(1, 5, n).astype(float),
    })
