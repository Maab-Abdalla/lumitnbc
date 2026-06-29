"""
LumiTNBC: Database Models
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    password_hash = db.Column(db.String(512), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="patient")  # patient | provider | admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Optional patient-specific fields
    phone = db.Column(db.String(50))
    diagnosis_date = db.Column(db.String(50))
    cancer_type = db.Column(db.String(100), default="Triple-Negative Breast Cancer")
    cancer_stage = db.Column(db.String(20))

    analyses = db.relationship("Analysis", backref="user", lazy="dynamic",
                               foreign_keys="Analysis.user_id")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "phone": self.phone,
            "diagnosis_date": self.diagnosis_date,
            "cancer_type": self.cancer_type,
            "cancer_stage": self.cancer_stage,
        }


class Analysis(db.Model):
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(30), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # Classification
    subtype = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    method = db.Column(db.String(30))  # xgboost_ml | clinical_rules | hybrid_ml

    # Input details
    filename = db.Column(db.String(255))
    input_mode = db.Column(db.String(20))  # gene_only | clinical_only | hybrid

    # Full result stored as JSON
    result_json = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        import json
        base = {
            "analysis_id": self.analysis_id,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "method": self.method,
            "input_mode": self.input_mode,
            "filename": self.filename,
            "timestamp": self.created_at.isoformat(),
        }
        if self.result_json:
            try:
                base.update(json.loads(self.result_json))
            except Exception:
                pass
        return base


def patient_code(user_id):
    """Privacy-safe patient identifier shown to providers (no name/email).
    Deterministic so the same patient always maps to the same code."""
    return f"PT-{int(user_id):05d}"


class ReviewRequest(db.Model):
    """A patient's request for a provider to review one of their analyses.

    Patients opt in per-analysis. Requests sit in a shared queue that any
    provider can pick up. The provider writes free-text feedback that the
    patient then sees on their results page.
    """
    __tablename__ = "review_requests"

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(30), db.ForeignKey("analyses.analysis_id"),
                            nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"),
                           nullable=False, index=True)

    status = db.Column(db.String(20), default="pending", index=True)  # pending | reviewed

    # Filled in when a provider completes the review.
    reviewer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    review_text = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    analysis = db.relationship("Analysis", backref="review_requests",
                               foreign_keys=[analysis_id])
    patient = db.relationship("User", foreign_keys=[patient_id])
    reviewer = db.relationship("User", foreign_keys=[reviewer_id])

    def to_dict(self, include_name=False):
        d = {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "patient_code": patient_code(self.patient_id),
            "subtype": self.analysis.subtype if self.analysis else None,
            "status": self.status,
            "review_text": self.review_text,
            "requested_at": self.created_at.isoformat() if self.created_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer_name": self.reviewer.name if self.reviewer else None,
        }
        # Names are deliberately omitted for providers; only enabled where the
        # viewer is the patient themselves.
        if include_name and self.patient:
            d["patient_name"] = self.patient.name
        return d
