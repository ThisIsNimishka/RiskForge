"""
generate_data.py
----------------
Generates realistic synthetic credit application data
mimicking real-world lending portfolios.

Features engineered to reflect actual credit bureau data patterns
including realistic correlations and class imbalance (~20% default rate).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from loguru import logger

SEED = 42
np.random.seed(SEED)


def generate_credit_data(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Generate synthetic credit application dataset with realistic
    feature correlations and a ~20% default rate.
    """
    logger.info(f"Generating {n_samples:,} synthetic credit records...")

    # ── Core applicant demographics ──────────────────────────────────
    age = np.random.normal(40, 12, n_samples).clip(18, 75).astype(int)
    employment_years = np.random.exponential(7, n_samples).clip(0, 45)

    # Income correlated with age & employment
    base_income = 30_000 + age * 800 + employment_years * 1_500
    annual_income = np.random.lognormal(
        np.log(base_income), 0.5, n_samples
    ).clip(15_000, 500_000)

    # ── Credit profile ────────────────────────────────────────────────
    # Credit score correlated with income and employment stability
    credit_score_base = 550 + (annual_income / 500_000) * 200 + employment_years * 2
    credit_score = np.random.normal(credit_score_base, 50, n_samples).clip(300, 850).astype(int)

    num_open_accounts = np.random.poisson(6, n_samples).clip(0, 30)
    num_delinquencies_2yr = np.random.poisson(0.3, n_samples).clip(0, 10)
    num_credit_inquiries = np.random.poisson(1.5, n_samples).clip(0, 15)
    months_since_last_delinq = np.where(
        num_delinquencies_2yr > 0,
        np.random.randint(1, 24, n_samples),
        np.nan
    )

    # Revolving utilization (inversely correlated with credit score)
    revolving_util = np.random.beta(2, 5, n_samples)
    revolving_util = revolving_util * (1 + (800 - credit_score) / 800 * 0.5)
    revolving_util = revolving_util.clip(0, 1)

    # ── Loan characteristics ──────────────────────────────────────────
    loan_purpose_choices = [
        "debt_consolidation", "home_improvement", "major_purchase",
        "medical", "small_business", "vacation", "car", "education"
    ]
    loan_purpose_probs = [0.35, 0.18, 0.12, 0.10, 0.10, 0.06, 0.05, 0.04]
    loan_purpose = np.random.choice(loan_purpose_choices, n_samples, p=loan_purpose_probs)

    loan_amount = np.random.lognormal(9.5, 0.8, n_samples).clip(1_000, 100_000)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.05, 0.15, 0.45, 0.15, 0.20])
    interest_rate = 5 + (800 - credit_score) / 50 + np.random.normal(0, 1, n_samples)
    interest_rate = interest_rate.clip(3, 35)

    # ── Derived ratios ────────────────────────────────────────────────
    debt_to_income = loan_amount / annual_income
    monthly_payment = (loan_amount * (interest_rate / 1200)) / (1 - (1 + interest_rate / 1200) ** (-loan_term_months))
    payment_to_income = monthly_payment / (annual_income / 12)

    home_ownership = np.random.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        n_samples, p=[0.38, 0.42, 0.17, 0.03]
    )

    # ── Churn / Default label ─────────────────────────────────────────
    # Logistic model for default probability
    log_odds = (
        -4.5
        + 0.008 * (800 - credit_score)
        + 1.8 * revolving_util
        + 2.0 * debt_to_income
        + 1.5 * payment_to_income
        + 0.3 * num_delinquencies_2yr
        + 0.05 * num_credit_inquiries
        - 0.02 * employment_years
        - 0.5 * (home_ownership == "OWN").astype(float)
        + 0.3 * (loan_purpose == "small_business").astype(float)
        + 0.2 * (loan_purpose == "vacation").astype(float)
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    default = (np.random.uniform(0, 1, n_samples) < default_prob).astype(int)

    logger.info(f"Default rate: {default.mean():.2%}")

    df = pd.DataFrame({
        "applicant_id": [f"APP_{i:06d}" for i in range(n_samples)],
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_years": employment_years.round(1),
        "home_ownership": home_ownership,
        "loan_amount": loan_amount.round(2),
        "loan_term_months": loan_term_months,
        "loan_purpose": loan_purpose,
        "interest_rate": interest_rate.round(2),
        "credit_score": credit_score,
        "num_open_accounts": num_open_accounts,
        "num_delinquencies_2yr": num_delinquencies_2yr,
        "num_credit_inquiries": num_credit_inquiries,
        "months_since_last_delinq": months_since_last_delinq,
        "revolving_utilization": revolving_util.round(4),
        "debt_to_income_ratio": debt_to_income.round(4),
        "payment_to_income_ratio": payment_to_income.round(4),
        "monthly_payment": monthly_payment.round(2),
        "default": default,
    })

    return df


def split_and_save(df: pd.DataFrame, output_dir: Path) -> None:
    """Split into train/validation/test and save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chronological-style split (80/10/10)
    n = len(df)
    train = df.iloc[:int(0.8 * n)]
    val = df.iloc[int(0.8 * n):int(0.9 * n)]
    test = df.iloc[int(0.9 * n):]

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Saved: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    logger.info(f"Train default rate: {train['default'].mean():.2%}")
    logger.info(f"Test default rate:  {test['default'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit data")
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()

    df = generate_credit_data(args.n_samples)
    split_and_save(df, Path(args.output_dir))
    logger.success("Data generation complete!")
