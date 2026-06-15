"""
REST API — Virtual Wallet & Fraud Detection
Endpoints: wallet balance, transaction submission, fraud score, history
"""

from flask import Flask, jsonify, request, abort
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import uuid
import os

from fraud_detection import load_model, score_transaction

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///wallet.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Account(db.Model):
    __tablename__ = "accounts"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(64), unique=True, nullable=False)
    balance = db.Column(db.Float, default=0.0)
    currency = db.Column(db.String(3), default="USD")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_verified = db.Column(db.Boolean, default=False)


class Transaction(db.Model):
    __tablename__ = "transactions"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = db.Column(db.Integer, db.ForeignKey("accounts.id"), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    merchant = db.Column(db.String(128))
    merchant_category = db.Column(db.String(64))
    card_country = db.Column(db.String(2), default="US")
    merchant_country = db.Column(db.String(2), default="US")
    fraud_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(10), default="LOW")
    status = db.Column(db.String(16), default="PENDING")  # CLEARED | BLOCKED | REVIEW
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feature_vector(txn_data: dict, account: Account) -> dict:
    """Map raw request fields to model feature vector."""
    now = datetime.utcnow()
    recent_count = Transaction.query.filter(
        Transaction.account_id == account.id,
        Transaction.created_at >= now - timedelta(hours=24),
    ).count()

    recent_txns = Transaction.query.filter_by(account_id=account.id).all()
    avg_amount = (
        sum(t.amount for t in recent_txns) / len(recent_txns) if recent_txns else txn_data["amount"]
    )

    return {
        "amount": txn_data["amount"],
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "merchant_category": hash(txn_data.get("merchant_category", "retail")) % 5,
        "transaction_frequency_24h": recent_count,
        "avg_transaction_amount": avg_amount,
        "amount_deviation": txn_data["amount"] - avg_amount,
        "is_foreign": int(txn_data.get("card_country", "US") != txn_data.get("merchant_country", "US")),
        "is_new_merchant": int(txn_data.get("is_new_merchant", False)),
        "account_age_days": (now - account.created_at).days,
        "failed_attempts_24h": txn_data.get("failed_attempts_24h", 0),
    }


def _resolve_status(risk_level: str) -> str:
    return {"LOW": "CLEARED", "MEDIUM": "REVIEW", "HIGH": "BLOCKED"}.get(risk_level, "REVIEW")


# ---------------------------------------------------------------------------
# Routes — Wallet
# ---------------------------------------------------------------------------

@app.route("/api/v1/wallet/<user_id>", methods=["GET"])
def get_wallet(user_id):
    account = Account.query.filter_by(user_id=user_id).first_or_404()
    return jsonify({
        "user_id": account.user_id,
        "balance": account.balance,
        "currency": account.currency,
        "is_verified": account.is_verified,
        "created_at": account.created_at.isoformat(),
    })


@app.route("/api/v1/wallet", methods=["POST"])
def create_wallet():
    data = request.get_json(force=True)
    if not data or "user_id" not in data:
        abort(400, description="user_id is required")

    if Account.query.filter_by(user_id=data["user_id"]).first():
        abort(409, description="Wallet already exists")

    account = Account(
        user_id=data["user_id"],
        balance=float(data.get("initial_balance", 0.0)),
        currency=data.get("currency", "USD"),
    )
    db.session.add(account)
    db.session.commit()
    return jsonify({"message": "Wallet created", "user_id": account.user_id}), 201


@app.route("/api/v1/wallet/<user_id>/deposit", methods=["POST"])
def deposit(user_id):
    data = request.get_json(force=True)
    amount = float(data.get("amount", 0))
    if amount <= 0:
        abort(400, description="Amount must be positive")

    account = Account.query.filter_by(user_id=user_id).first_or_404()
    account.balance += amount
    db.session.commit()
    return jsonify({"balance": account.balance, "deposited": amount})


# ---------------------------------------------------------------------------
# Routes — Transactions
# ---------------------------------------------------------------------------

@app.route("/api/v1/transactions", methods=["POST"])
def submit_transaction():
    data = request.get_json(force=True)
    required = ["user_id", "amount", "merchant"]
    for field in required:
        if field not in data:
            abort(400, description=f"Missing required field: {field}")

    account = Account.query.filter_by(user_id=data["user_id"]).first_or_404()

    # Fraud scoring
    try:
        model, scaler = load_model()
        features = _build_feature_vector(data, account)
        result = score_transaction(model, scaler, features)
    except FileNotFoundError:
        # Model not yet trained — allow transaction with LOW risk
        result = {"fraud_probability": 0.05, "label": "LEGIT", "risk_level": "LOW"}

    status = _resolve_status(result["risk_level"])

    txn = Transaction(
        account_id=account.id,
        amount=data["amount"],
        merchant=data["merchant"],
        merchant_category=data.get("merchant_category", "retail"),
        card_country=data.get("card_country", "US"),
        merchant_country=data.get("merchant_country", "US"),
        fraud_score=result["fraud_probability"],
        risk_level=result["risk_level"],
        status=status,
    )

    if status == "CLEARED":
        if account.balance < data["amount"]:
            abort(402, description="Insufficient balance")
        account.balance -= data["amount"]

    db.session.add(txn)
    db.session.commit()

    return jsonify({
        "transaction_id": txn.id,
        "status": txn.status,
        "fraud_score": txn.fraud_score,
        "risk_level": txn.risk_level,
        "amount": txn.amount,
        "merchant": txn.merchant,
        "timestamp": txn.created_at.isoformat(),
    }), 201


@app.route("/api/v1/transactions/<user_id>", methods=["GET"])
def get_transactions(user_id):
    account = Account.query.filter_by(user_id=user_id).first_or_404()
    limit = int(request.args.get("limit", 50))
    txns = (
        Transaction.query
        .filter_by(account_id=account.id)
        .order_by(Transaction.created_at.desc())
        .limit(limit)
        .all()
    )
    return jsonify([
        {
            "transaction_id": t.id,
            "amount": t.amount,
            "merchant": t.merchant,
            "fraud_score": t.fraud_score,
            "risk_level": t.risk_level,
            "status": t.status,
            "timestamp": t.created_at.isoformat(),
        }
        for t in txns
    ])


@app.route("/api/v1/transactions/<user_id>/stats", methods=["GET"])
def transaction_stats(user_id):
    account = Account.query.filter_by(user_id=user_id).first_or_404()
    txns = Transaction.query.filter_by(account_id=account.id).all()
    total = len(txns)
    if total == 0:
        return jsonify({"total": 0})

    return jsonify({
        "total": total,
        "blocked": sum(1 for t in txns if t.status == "BLOCKED"),
        "cleared": sum(1 for t in txns if t.status == "CLEARED"),
        "review": sum(1 for t in txns if t.status == "REVIEW"),
        "avg_fraud_score": round(sum(t.fraud_score for t in txns) / total, 4),
        "total_spent": round(sum(t.amount for t in txns if t.status == "CLEARED"), 2),
    })


# ---------------------------------------------------------------------------
# Routes — Health
# ---------------------------------------------------------------------------

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "FraudShield Wallet API", "version": "1.0.0"})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e.description)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(409)
def conflict(e):
    return jsonify({"error": str(e.description)}), 409


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)
