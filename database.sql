-- ============================================================
-- FraudShield — Schema & Analytical Queries
-- ============================================================

-- ------------------------------------------------------------
-- Schema
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS accounts (
    id            SERIAL PRIMARY KEY,
    user_id       VARCHAR(64)  UNIQUE NOT NULL,
    balance       NUMERIC(15,2) DEFAULT 0.00,
    currency      CHAR(3)       DEFAULT 'USD',
    is_verified   BOOLEAN       DEFAULT FALSE,
    created_at    TIMESTAMP     DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    id                VARCHAR(36)   PRIMARY KEY,
    account_id        INT           REFERENCES accounts(id) ON DELETE CASCADE,
    amount            NUMERIC(15,2) NOT NULL,
    merchant          VARCHAR(128),
    merchant_category VARCHAR(64),
    card_country      CHAR(2)       DEFAULT 'US',
    merchant_country  CHAR(2)       DEFAULT 'US',
    fraud_score       NUMERIC(5,4)  DEFAULT 0.0,
    risk_level        VARCHAR(10)   DEFAULT 'LOW',
    status            VARCHAR(16)   DEFAULT 'PENDING',
    created_at        TIMESTAMP     DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fraud_alerts (
    id             SERIAL PRIMARY KEY,
    transaction_id VARCHAR(36) REFERENCES transactions(id),
    alert_type     VARCHAR(64),
    description    TEXT,
    resolved       BOOLEAN   DEFAULT FALSE,
    created_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_txn_account  ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_txn_status   ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_txn_score    ON transactions(fraud_score DESC);
CREATE INDEX IF NOT EXISTS idx_txn_created  ON transactions(created_at DESC);


-- ------------------------------------------------------------
-- Query 1: High-risk transactions in the last 24 hours
-- ------------------------------------------------------------

SELECT
    t.id              AS transaction_id,
    a.user_id,
    t.amount,
    t.merchant,
    t.merchant_category,
    t.card_country,
    t.merchant_country,
    t.fraud_score,
    t.risk_level,
    t.status,
    t.created_at
FROM transactions t
JOIN accounts a ON a.id = t.account_id
WHERE t.created_at >= NOW() - INTERVAL '24 hours'
  AND t.fraud_score >= 0.75
ORDER BY t.fraud_score DESC
LIMIT 100;


-- ------------------------------------------------------------
-- Query 2: Daily fraud summary — blocked vs cleared vs review
-- ------------------------------------------------------------

SELECT
    DATE(created_at)                                     AS txn_date,
    COUNT(*)                                             AS total_transactions,
    SUM(CASE WHEN status = 'BLOCKED' THEN 1 ELSE 0 END) AS blocked,
    SUM(CASE WHEN status = 'CLEARED' THEN 1 ELSE 0 END) AS cleared,
    SUM(CASE WHEN status = 'REVIEW'  THEN 1 ELSE 0 END) AS under_review,
    ROUND(AVG(fraud_score) * 100, 2)                    AS avg_fraud_score_pct,
    ROUND(
        SUM(CASE WHEN status = 'BLOCKED' THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 100, 2
    )                                                    AS fraud_rate_pct
FROM transactions
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY txn_date DESC;


-- ------------------------------------------------------------
-- Query 3: Accounts with highest fraud activity
-- ------------------------------------------------------------

SELECT
    a.user_id,
    COUNT(t.id)                                          AS total_txns,
    SUM(CASE WHEN t.status = 'BLOCKED' THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(AVG(t.fraud_score), 4)                         AS avg_risk_score,
    MAX(t.fraud_score)                                   AS max_risk_score,
    SUM(t.amount)                                        AS total_attempted
FROM accounts a
JOIN transactions t ON t.account_id = a.id
WHERE t.created_at >= NOW() - INTERVAL '7 days'
GROUP BY a.user_id
HAVING SUM(CASE WHEN t.status = 'BLOCKED' THEN 1 ELSE 0 END) > 2
ORDER BY fraud_count DESC
LIMIT 20;


-- ------------------------------------------------------------
-- Query 4: Fraud rate by merchant category
-- ------------------------------------------------------------

SELECT
    merchant_category,
    COUNT(*)                                              AS total,
    SUM(CASE WHEN status = 'BLOCKED' THEN 1 ELSE 0 END)  AS fraud_count,
    ROUND(
        SUM(CASE WHEN status = 'BLOCKED' THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 100, 2
    )                                                     AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                                 AS avg_amount
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;


-- ------------------------------------------------------------
-- Query 5: Cross-border transaction anomalies
-- ------------------------------------------------------------

SELECT
    t.id,
    a.user_id,
    t.amount,
    t.card_country,
    t.merchant_country,
    t.fraud_score,
    t.created_at
FROM transactions t
JOIN accounts a ON a.id = t.account_id
WHERE t.card_country <> t.merchant_country
  AND t.fraud_score >= 0.5
  AND t.created_at >= NOW() - INTERVAL '7 days'
ORDER BY t.fraud_score DESC;


-- ------------------------------------------------------------
-- Query 6: Spending pattern — rolling 7-day avg per account
-- ------------------------------------------------------------

SELECT
    account_id,
    created_at::DATE                    AS txn_date,
    SUM(amount)                         AS daily_total,
    AVG(SUM(amount)) OVER (
        PARTITION BY account_id
        ORDER BY created_at::DATE
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                   AS rolling_7d_avg,
    SUM(amount) - AVG(SUM(amount)) OVER (
        PARTITION BY account_id
        ORDER BY created_at::DATE
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                   AS deviation
FROM transactions
WHERE status = 'CLEARED'
GROUP BY account_id, created_at::DATE
ORDER BY account_id, txn_date;
