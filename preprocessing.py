import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "amount", "hour_of_day", "day_of_week", "merchant_category",
    "transaction_frequency_24h", "avg_transaction_amount",
    "amount_deviation", "is_foreign", "is_new_merchant",
    "account_age_days", "failed_attempts_24h",
]

TARGET_COLUMN = "is_fraud"


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["transaction_time"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour_of_day"] = df["transaction_time"].dt.hour
    df["day_of_week"] = df["transaction_time"].dt.dayofweek

    # Rolling stats per account
    df = df.sort_values(["account_id", "transaction_time"])
    df["transaction_frequency_24h"] = (
        df.groupby("account_id")["transaction_time"]
        .transform(lambda x: x.expanding().count())
    )
    df["avg_transaction_amount"] = (
        df.groupby("account_id")["amount"]
        .transform(lambda x: x.expanding().mean())
    )
    df["amount_deviation"] = df["amount"] - df["avg_transaction_amount"]

    # Binary flags
    df["is_foreign"] = (df["card_country"] != df["merchant_country"]).astype(int)
    df["is_new_merchant"] = df["merchant_first_seen"].astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    df["merchant_category"] = le.fit_transform(df["merchant_category"].astype(str))
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_class_weights(y: pd.Series) -> dict:
    fraud_ratio = y.value_counts(normalize=True)
    return {0: fraud_ratio[1], 1: fraud_ratio[0]}


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 10000

    demo_df = pd.DataFrame({
        "transaction_time": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "account_id": np.random.randint(1, 500, n),
        "amount": np.random.exponential(scale=200, size=n).round(2),
        "merchant_category": np.random.choice(["retail", "food", "travel", "crypto", "online"], n),
        "card_country": np.random.choice(["US", "UK", "IN"], n),
        "merchant_country": np.random.choice(["US", "UK", "IN", "RU", "CN"], n),
        "merchant_first_seen": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "failed_attempts_24h": np.random.poisson(0.2, n),
        "account_age_days": np.random.randint(1, 3650, n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })

    demo_df = engineer_features(demo_df)
    demo_df = encode_categoricals(demo_df)

    X_train, X_test, y_train, y_test = split_data(demo_df)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Fraud rate       : {demo_df['is_fraud'].mean():.2%}")
    print(f"Features         : {FEATURE_COLUMNS}")
