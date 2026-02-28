"""
tests/test_features.py + test_models.py + test_api.py combined
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from src.data.generate_data import generate_credit_data
from src.features.build_features import (
    WOEEncoder, CreditFeatureEngineer, Winsorizer, build_feature_pipeline
)


# ── Data Tests ─────────────────────────────────────────────────────────────
class TestDataGeneration:
    def test_generates_correct_shape(self):
        df = generate_credit_data(n_samples=1000)
        assert len(df) == 1000

    def test_default_rate_realistic(self):
        df = generate_credit_data(n_samples=5000)
        default_rate = df["default"].mean()
        assert 0.10 <= default_rate <= 0.35, f"Unrealistic default rate: {default_rate:.2%}"

    def test_no_null_required_fields(self):
        df = generate_credit_data(n_samples=500)
        required = ["credit_score", "annual_income", "loan_amount", "default"]
        for col in required:
            assert df[col].isna().sum() == 0, f"Null values in {col}"

    def test_credit_score_in_range(self):
        df = generate_credit_data(n_samples=1000)
        assert df["credit_score"].between(300, 850).all()

    def test_age_in_range(self):
        df = generate_credit_data(n_samples=1000)
        assert df["age"].between(18, 75).all()


# ── Feature Engineering Tests ───────────────────────────────────────────────
class TestFeatureEngineering:
    @pytest.fixture
    def sample_data(self):
        df = generate_credit_data(n_samples=500)
        return df

    def test_woe_encoder_fit_transform(self, sample_data):
        y = sample_data["default"]
        X_cat = sample_data[["home_ownership", "loan_purpose"]]
        encoder = WOEEncoder()
        X_woe = encoder.fit_transform(X_cat, y)
        assert X_woe.shape == X_cat.shape
        assert X_woe.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()

    def test_woe_iv_scores_positive(self, sample_data):
        y = sample_data["default"]
        X_cat = sample_data[["home_ownership"]]
        encoder = WOEEncoder()
        encoder.fit(X_cat, y)
        assert all(iv >= 0 for iv in encoder.iv_scores_.values())

    def test_feature_engineer_creates_new_features(self, sample_data):
        engineer = CreditFeatureEngineer()
        y = sample_data["default"]
        X = sample_data.drop(columns=["default", "applicant_id"])
        X_eng = engineer.fit_transform(X, y)
        assert "credit_health_score" in X_eng.columns
        assert "dti_x_revolving" in X_eng.columns
        assert "log_annual_income" in X_eng.columns

    def test_winsorizer_clips_outliers(self):
        df = pd.DataFrame({"a": [1, 2, 3, 100, -50, 5, 6, 7, 8, 9]})
        winsorizer = Winsorizer(lower=0.1, upper=0.9)
        winsorizer.fit(df)
        df_out = winsorizer.transform(df)
        assert df_out["a"].max() < 100
        assert df_out["a"].min() > -50

    def test_pipeline_output_no_nulls(self, sample_data):
        artifacts = build_feature_pipeline(sample_data)
        assert artifacts["X_processed"].isna().sum().sum() == 0


# ── Model Tests ─────────────────────────────────────────────────────────────
class TestModelEvaluation:
    def test_auc_above_random(self):
        """Simple sanity check: any trained model should beat random."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        df = generate_credit_data(n_samples=2000)
        artifacts = build_feature_pipeline(df)
        X, y = artifacts["X_processed"], artifacts["y"]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        assert auc > 0.65, f"AUC too low: {auc:.3f}"


# ── API Schema Tests ─────────────────────────────────────────────────────────
class TestAPISchemas:
    def test_valid_request(self):
        from src.api.schemas import CreditApplicationRequest
        req = CreditApplicationRequest(
            applicant_id="TEST_001",
            age=35, annual_income=75000, employment_years=8.0,
            home_ownership="MORTGAGE", loan_amount=15000,
            loan_term_months=36, loan_purpose="debt_consolidation",
            credit_score=720, num_open_accounts=5,
            num_delinquencies_2yr=0, num_credit_inquiries=2,
            months_since_last_delinq=None,
            revolving_utilization=0.28, debt_to_income_ratio=0.22,
        )
        assert req.applicant_id == "TEST_001"

    def test_invalid_credit_score_rejected(self):
        from pydantic import ValidationError
        from src.api.schemas import CreditApplicationRequest
        with pytest.raises(ValidationError):
            CreditApplicationRequest(
                applicant_id="TEST_002", age=35, annual_income=75000,
                employment_years=8.0, home_ownership="MORTGAGE",
                loan_amount=15000, loan_term_months=36,
                loan_purpose="debt_consolidation",
                credit_score=999,  # Invalid: > 850
                num_open_accounts=5, num_delinquencies_2yr=0,
                num_credit_inquiries=2, revolving_utilization=0.28,
                debt_to_income_ratio=0.22,
            )

    def test_underage_applicant_rejected(self):
        from pydantic import ValidationError
        from src.api.schemas import CreditApplicationRequest
        with pytest.raises(ValidationError):
            CreditApplicationRequest(
                applicant_id="TEST_003", age=16,  # Invalid: < 18
                annual_income=30000, employment_years=1.0,
                home_ownership="RENT", loan_amount=5000,
                loan_term_months=24, loan_purpose="car",
                credit_score=600, num_open_accounts=1,
                num_delinquencies_2yr=0, num_credit_inquiries=1,
                revolving_utilization=0.10, debt_to_income_ratio=0.15,
            )
