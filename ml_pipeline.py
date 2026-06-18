"""
LumiTNBC: ML Pipeline
Handles three input modes:
  1. gene_only     → Hybrid XGBoost on gene features; clin_* filled with population medians
  2. clinical_only → Rule-based scoring (FUSCC surrogates): handled in app_utils.py
  3. hybrid        → Hybrid XGBoost on full 163-feature space (150 genes + 13 clin_*)
"""
import os
import io
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    import xgboost  # noqa: needed for joblib unpickling
except ImportError:
    pass

_model = None
_label_encoder = None
_all_features = []       # 163 ordered features from hybrid_feature_list.csv
_gene_features = []      # 150 gene-only features
_clinical_features = []  # 13 clin_* features
_shap_explainer = None

# Population medians used to fill clin_* when only genes are available
CLINICAL_MEDIANS = {
    "clin_age": 52.0,
    "clin_stage": 2.0,
    "clin_grade": 3.0,
    "clin_tumor_size": 2.5,
    "clin_node_status": 0.0,
    "clin_tmb": 1.0,
    "clin_menopausal": 0.0,
    "clin_cellularity": 2.0,
    "clin_hist_ductal": 1.0,
    "clin_hist_lobular": 0.0,
    "clin_hist_metaplastic": 0.0,
    "clin_hist_medullary": 0.0,
    "clin_hist_other": 0.0,
}


def load_assets():
    global _model, _label_encoder, _all_features, _gene_features, _clinical_features, _shap_explainer

    # hybrid_feature_list.csv contains ALL 163 features in training order
    feat_path = os.path.join(MODEL_DIR, "hybrid_feature_list.csv")
    feat_df = pd.read_csv(feat_path, header=None)
    _all_features = feat_df.iloc[1:, 0].tolist()

    # Split into gene vs clinical subsets
    _clinical_features = [f for f in _all_features if f.startswith("clin_")]
    _gene_features = [f for f in _all_features if not f.startswith("clin_")]
    print(f"[✓] Features: {len(_all_features)} total ({len(_gene_features)} genes, {len(_clinical_features)} clinical)")

    # Label encoder
    le_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    _label_encoder = joblib.load(le_path)
    print(f"[✓] Label encoder: {list(_label_encoder.classes_)}")

    # Hybrid model
    model_path = os.path.join(MODEL_DIR, "hybrid_model.joblib")
    try:
        _model = joblib.load(model_path)
        print("[✓] Hybrid XGBoost model loaded")
    except Exception as e:
        print(f"[!] Could not load model: {e}: demo mode")
        _model = None

    # SHAP
    if _model is not None:
        try:
            import shap
            _shap_explainer = shap.TreeExplainer(_model)
            print("[✓] SHAP TreeExplainer ready")
        except Exception as e:
            print(f"[!] SHAP unavailable: {e}")
            _shap_explainer = None


def get_gene_list():
    return list(_gene_features)


def get_classes():
    if _label_encoder is not None:
        return list(_label_encoder.classes_)
    return ["BL1", "BL2", "LAR", "M"]


# ── Input parsing ─────────────────────────────────────────────────────────────

def parse_gene_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV/TSV gene expression file → DataFrame (samples × genes)."""
    sep = "\t" if filename.lower().endswith(".tsv") else None
    if sep is None:
        first_line = file_bytes[:500].decode("utf-8", errors="ignore")
        sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, index_col=0)
    # Handle genes-as-rows orientation
    if len(df.columns) < 50 and len(df.index) > 50:
        df = df.T
    return df


def build_clinical_vector(clinical_data: dict) -> dict:
    """Map form clinical_data to the 13 clin_* feature columns."""
    hist = clinical_data.get("histologic_type", "")
    node_map = {"n0": 0, "n1": 1, "n2": 2, "n3": 3}
    menopausal_map = {"premenopausal": 0, "perimenopausal": 1, "postmenopausal": 2}
    return {
        "clin_age":             _safe_float(clinical_data.get("age"), CLINICAL_MEDIANS["clin_age"]),
        "clin_stage":           _safe_float(clinical_data.get("stage"), CLINICAL_MEDIANS["clin_stage"]),
        "clin_grade":           _safe_float(clinical_data.get("grade"), CLINICAL_MEDIANS["clin_grade"]),
        "clin_tumor_size":      _safe_float(clinical_data.get("tumor_size"), CLINICAL_MEDIANS["clin_tumor_size"]),
        "clin_node_status":     float(node_map.get(clinical_data.get("lymph_node", ""), 0)),
        "clin_tmb":             _safe_float(clinical_data.get("tmb"), CLINICAL_MEDIANS["clin_tmb"]),
        "clin_menopausal":      float(menopausal_map.get(clinical_data.get("menopausal_status", ""), 0)),
        "clin_cellularity":     _safe_float(clinical_data.get("cellularity"), CLINICAL_MEDIANS["clin_cellularity"]),
        "clin_hist_ductal":     1.0 if hist in ["ductal_nst", "ductal"] else 0.0,
        "clin_hist_lobular":    1.0 if hist == "lobular" else 0.0,
        "clin_hist_metaplastic": 1.0 if hist == "metaplastic" else 0.0,
        "clin_hist_medullary":  1.0 if hist == "medullary" else 0.0,
        "clin_hist_other":      1.0 if hist in ["other", "apocrine"] else 0.0,
    }


# ── Classification entry points ───────────────────────────────────────────────

def classify_gene_only(gene_df: pd.DataFrame) -> dict:
    """Mode 1: Gene file only: clin_* filled with population medians."""
    missing = [g for g in _gene_features if g not in gene_df.columns]
    coverage = len(_gene_features) - len(missing)

    if len(missing) > len(_gene_features) * 0.5:
        return {
            "error": True,
            "message": (
                f"Too many missing genes ({len(missing)}/{len(_gene_features)}). "
                f"Ensure your file contains the required gene expression values."
            )
        }

    # Build full 163-feature matrix
    X = _build_feature_matrix(gene_df, clin_values=None)
    result = _run_model(X, sample_idx=0)
    result.update({
        "input_mode": "gene_only",
        "method": "xgboost_ml",
        "method_note": "Hybrid XGBoost: 150-gene expression (clinical filled with population medians)",
        "missing_genes": len(missing),
        "genes_matched": coverage,
        "samples_processed": len(gene_df),
    })
    return result


def classify_hybrid(gene_df: pd.DataFrame, clinical_data: dict) -> dict:
    """Mode 3: Gene expression + clinical data: full hybrid model."""
    missing = [g for g in _gene_features if g not in gene_df.columns]

    clin_values = build_clinical_vector(clinical_data)
    X = _build_feature_matrix(gene_df, clin_values=clin_values)
    result = _run_model(X, sample_idx=0)
    result.update({
        "input_mode": "hybrid",
        "method": "hybrid_ml",
        "method_note": "Hybrid XGBoost: 150-gene expression + 13 clinical features",
        "missing_genes": len(missing),
        "genes_matched": len(_gene_features) - len(missing),
        "samples_processed": len(gene_df),
    })
    return result


def _build_feature_matrix(gene_df: pd.DataFrame, clin_values: dict | None) -> np.ndarray:
    """
    Build the (n_samples, 163) matrix in the correct training feature order.
    clin_values=None → use population medians for all clin_* columns.
    """
    n = len(gene_df)
    rows = []
    for feat in _all_features:
        if feat.startswith("clin_"):
            if clin_values is not None:
                val = clin_values.get(feat, CLINICAL_MEDIANS.get(feat, 0.0))
            else:
                val = CLINICAL_MEDIANS.get(feat, 0.0)
            rows.append(np.full(n, val))
        else:
            if feat in gene_df.columns:
                rows.append(gene_df[feat].fillna(0.0).values.astype(float))
            else:
                rows.append(np.zeros(n))

    return np.column_stack(rows)  # (n, 163)


def _run_model(X: np.ndarray, sample_idx: int = 0) -> dict:
    if _model is None:
        return _simulated_prediction()

    probas = _model.predict_proba(X)
    pred_indices = np.argmax(probas, axis=1)
    pred_labels = _label_encoder.inverse_transform(pred_indices)

    subtype = str(pred_labels[sample_idx])
    confidence = round(float(np.max(probas[sample_idx])) * 100, 1)
    all_probas = {
        str(_label_encoder.inverse_transform([i])[0]): round(float(probas[sample_idx][i]) * 100, 1)
        for i in range(len(_label_encoder.classes_))
    }

    shap_features = _compute_shap(X, sample_idx, subtype)

    return {
        "subtype": subtype,
        "confidence": confidence,
        "probabilities": all_probas,
        "shap_features": shap_features,
    }


def _compute_shap(X: np.ndarray, sample_idx: int, predicted_subtype: str) -> list:
    if _shap_explainer is not None:
        try:
            shap_values = _shap_explainer.shap_values(X[sample_idx:sample_idx + 1])
            classes = list(_label_encoder.classes_)
            class_idx = classes.index(predicted_subtype)
            if isinstance(shap_values, list):
                sv = shap_values[class_idx][0]
            elif shap_values.ndim == 3:
                sv = shap_values[0, :, class_idx]
            else:
                sv = shap_values[0]
            top_idx = np.argsort(np.abs(sv))[::-1][:15]
            return [
                {
                    "gene": _all_features[i],
                    "shap_value": round(float(sv[i]), 4),
                    "abs_shap": round(float(abs(sv[i])), 4),
                    "expression": round(float(X[sample_idx][i]), 3),
                    "is_clinical": _all_features[i].startswith("clin_"),
                }
                for i in top_idx
            ]
        except Exception as e:
            print(f"[!] SHAP error: {e}")

    # Fallback: model feature importances
    if _model is not None:
        try:
            imp = _model.feature_importances_
            top_idx = np.argsort(imp)[::-1][:15]
            return [
                {
                    "gene": _all_features[i],
                    "shap_value": round(float(imp[i]), 4),
                    "abs_shap": round(float(imp[i]), 4),
                    "expression": round(float(X[sample_idx][i]), 3),
                    "is_clinical": _all_features[i].startswith("clin_"),
                }
                for i in top_idx
            ]
        except Exception:
            pass

    return _static_shap_fallback()


def _simulated_prediction() -> dict:
    np.random.seed(42)
    subtypes = ["BL1", "BL2", "M", "LAR"]
    probs = np.array([0.48, 0.18, 0.20, 0.14])
    probs = probs / probs.sum()
    predicted = subtypes[int(np.argmax(probs))]
    return {
        "subtype": predicted,
        "confidence": round(float(np.max(probs)) * 100, 1),
        "probabilities": {s: round(float(p) * 100, 1) for s, p in zip(subtypes, probs)},
        "shap_features": _static_shap_fallback(),
        "demo_mode": True,
    }


def _static_shap_fallback() -> list:
    return [
        {"gene": "BMP4",     "shap_value": 0.1572, "abs_shap": 0.1572, "expression": 0.0, "is_clinical": False},
        {"gene": "GBP5",     "shap_value": 0.1406, "abs_shap": 0.1406, "expression": 0.0, "is_clinical": False},
        {"gene": "BATF",     "shap_value": 0.1342, "abs_shap": 0.1342, "expression": 0.0, "is_clinical": False},
        {"gene": "TGFBI",    "shap_value": 0.1286, "abs_shap": 0.1286, "expression": 0.0, "is_clinical": False},
        {"gene": "LAG3",     "shap_value": 0.1277, "abs_shap": 0.1277, "expression": 0.0, "is_clinical": False},
        {"gene": "ADAMDEC1", "shap_value": 0.1171, "abs_shap": 0.1171, "expression": 0.0, "is_clinical": False},
        {"gene": "COL5A1",   "shap_value": 0.1168, "abs_shap": 0.1168, "expression": 0.0, "is_clinical": False},
        {"gene": "FZD7",     "shap_value": 0.0957, "abs_shap": 0.0957, "expression": 0.0, "is_clinical": False},
        {"gene": "CXCL10",   "shap_value": 0.0872, "abs_shap": 0.0872, "expression": 0.0, "is_clinical": False},
        {"gene": "EPHB3",    "shap_value": 0.0842, "abs_shap": 0.0842, "expression": 0.0, "is_clinical": False},
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
