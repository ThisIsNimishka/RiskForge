"""
schemas.py
----------
Pydantic v2 request/response models with field-level validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum


class HomeOwnership(str, Enum):
    RENT = "RENT"
    MORTGAGE = "MORTGAGE"
    OWN = "OWN"
    OTHER = "OTHER"


class LoanPurpose(str, Enum):
    debt_consolidation = "debt_consolidation"
    home_improvement = "home_improvement"
    major_purchase = "major_purchase"
    medical = "medical"
    small_business = "small_business"
    vacation = "vacation"
    car = "car"
    education = "education"


class CreditApplicationRequest(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier")
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., gt=0, description="Annual gross income in USD")
    employment_years: float = Field(..., ge=0, le=50, description="Years at current employer")
    home_ownership: HomeOwnership
    loan_amount: float = Field(..., gt=0, le=500_000, description="Requested loan amount in USD")
    loan_term_months: Literal[12, 24, 36, 48, 60] = Field(..., description="Loan term in months")
    loan_purpose: LoanPurpose
    credit_score: int = Field(..., ge=300, le=850, description="FICO credit score")
    num_open_accounts: int = Field(..., ge=0, le=100)
    num_delinquencies_2yr: int = Field(..., ge=0, le=50)
    num_credit_inquiries: int = Field(..., ge=0, le=30)
    months_since_last_delinq: Optional[float] = Field(None, ge=0)
    revolving_utilization: float = Field(..., ge=0.0, le=1.0)
    debt_to_income_ratio: float = Field(..., ge=0.0, le=5.0)

    model_config = {"json_schema_extra": {
        "example": {
            "applicant_id": "APP_001",
            "age": 35,
            "annual_income": 75000,
            "employment_years": 8.0,
            "home_ownership": "MORTGAGE",
            "loan_amount": 15000,
            "loan_term_months": 36,
            "loan_purpose": "debt_consolidation",
            "credit_score": 720,
            "num_open_accounts": 5,
            "num_delinquencies_2yr": 0,
            "num_credit_inquiries": 2,
            "months_since_last_delinq": None,
            "revolving_utilization": 0.28,
            "debt_to_income_ratio": 0.22,
        }
    }}


class ShapFactor(BaseModel):
    feature: str
    impact: float
    label: str


class ShapExplanation(BaseModel):
    top_positive_factors: list[ShapFactor]
    top_negative_factors: list[ShapFactor]


class CreditRiskResponse(BaseModel):
    applicant_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Default probability (0=safe, 1=high risk)")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    shap_explanation: dict
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cache_connected: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    auc_roc: float
    gini: float
    ks_statistic: float
    training_samples: int
    feature_count: int
