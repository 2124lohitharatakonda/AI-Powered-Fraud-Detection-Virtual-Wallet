import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from preprocessing import (
    load_data, engineer_features, encode_categoricals,
    split_data, scale_features, get_class_weights, FEATURE_COLUMNS,
)


MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": 42,
}


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **XGBOOST_PARAMS)


def build_ensemble(scale_pos_weight: float) -> VotingClassifier:
    xgb = build_xgboost(scale_pos_weight)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    return VotingClassifier(estimators=[("xgb", xgb), ("rf", rf)], voting="soft")


def train(X_train, y_train, X_val, y_val):
    weights = get_class_weights(y_train)
    scale_pos_weight = weights[1] / weights[0]

    model = build_xgboost(scale_pos_weight)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": (y_pred == y_test).mean(),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }

    print("\n=== Model Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def score_transaction(model, scaler, transaction: dict) -> dict:
    """Score a single transaction and return fraud probability."""
    row = pd.DataFrame([transaction])[FEATURE_COLUMNS]
    row_scaled = scaler.transform(row)
    prob = model.predict_proba(row_scaled)[0][1]
    label = "FRAUD" if prob >= 0.5 else "LEGIT"
    risk = "HIGH" if prob >= 0.75 else ("MEDIUM" if prob >= 0.4 else "LOW")
    return {"fraud_probability": round(float(prob), 4), "label": label, "risk_level": risk}


def detect_anomalies(X: np.ndarray, contamination: float = 0.03) -> np.ndarray:
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    flags = iso.fit_predict(X)
    return (flags == -1).astype(int)


def save_model(model, scaler) -> None:
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved → {MODEL_PATH}")


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def get_feature_importance(model) -> pd.DataFrame:
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    import os
    np.random.seed(42)
    n = 10000

    # Generate synthetic dataset
    from preprocessing import engineer_features, encode_categoricals
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

    print("Training XGBoost fraud detector...")
    model = train(X_train_s, y_train, X_test_s, y_test)
    metrics = evaluate(model, X_test_s, y_test)

    print("\nTop Feature Importances:")
    print(get_feature_importance(model).head(6).to_string(index=False))

    # Sample inference
    sample_txn = {
        "amount": 4500.0, "hour_of_day": 3, "day_of_week": 6,
        "merchant_category": 2, "transaction_frequency_24h": 12,
        "avg_transaction_amount": 180.0, "amount_deviation": 4320.0,
        "is_foreign": 1, "is_new_merchant": 1,
        "account_age_days": 45, "failed_attempts_24h": 3,
    }
    result = score_transaction(model, scaler, sample_txn)
    print(f"\nSample transaction score: {result}")

    save_model(model, scaler)
