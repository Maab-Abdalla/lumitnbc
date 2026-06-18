"""
LumiTNBC: Clinical Intelligence Module
=========================================
Two-stage use of clinical data:

Stage 1: CONFIDENCE CALIBRATION
  After gene-based subtype prediction, clinical data validates/adjusts
  the confidence score based on known subtype-clinical associations.

Stage 2: PERSONALISED INSIGHTS
  After classification, combines subtype + clinical data to generate
  risk profiles, treatment insights, and smarter trial matching.

Based on:
  - Lehmann et al. (2011, 2016, 2021): TNBC subtype characterisation
  - Jiang et al. (2019): FUSCC clinical surrogates
  - NCCN Guidelines: TNBC treatment pathways
"""

import numpy as np

# ============================================================
# SUBTYPE CLINICAL PROFILES (from literature)
# ============================================================
# Each subtype has expected clinical characteristics.
# Used for confidence calibration: if patient matches profile,
# confidence is reinforced; if atypical, confidence is reduced.

SUBTYPE_PROFILES = {
    "BL1": {
        "description": "Basal-Like 1: High proliferation, DNA damage response",
        "typical_age_range": (30, 60),
        "typical_age_median": 50,
        "typical_grade": 3,         # Almost always grade 3
        "typical_stage": [2, 3],    # Often presents at stage II-III
        "menopausal_bias": "premenopausal",  # More common in younger
        "histology_typical": ["ductal"],
        "histology_atypical": ["lobular"],
        "node_positive_rate": 0.45,  # ~45% node positive
        "expected_tmb": "moderate_high",
        "key_features": [
            "High proliferation (Ki-67 typically >50%)",
            "Enriched for BRCA1 mutations",
            "Best response to platinum-based chemotherapy",
            "High DNA damage response pathway activity",
        ]
    },
    "BL2": {
        "description": "Basal-Like 2: Growth factor signalling, metabolic",
        "typical_age_range": (35, 65),
        "typical_age_median": 53,
        "typical_grade": 3,
        "typical_stage": [2, 3],
        "menopausal_bias": None,  # No strong bias
        "histology_typical": ["ductal"],
        "histology_atypical": ["lobular"],
        "node_positive_rate": 0.40,
        "expected_tmb": "moderate",
        "key_features": [
            "Growth factor receptor signalling enrichment",
            "Metabolic pathway activation (glycolysis)",
            "Myoepithelial marker expression",
            "Lower pCR rate than BL1",
        ]
    },
    "LAR": {
        "description": "Luminal Androgen Receptor: Hormone-driven, luminal-like",
        "typical_age_range": (50, 80),
        "typical_age_median": 62,
        "typical_grade": 2,         # Often grade 2, sometimes grade 1
        "typical_stage": [1, 2],    # Often earlier stage
        "menopausal_bias": "postmenopausal",  # Strongly postmenopausal
        "histology_typical": ["ductal", "apocrine"],
        "histology_atypical": ["metaplastic", "medullary"],
        "node_positive_rate": 0.30,  # Lower node positive rate
        "expected_tmb": "low",
        "key_features": [
            "Androgen receptor overexpression",
            "Luminal-like gene expression pattern",
            "PI3K/AKT pathway activation common",
            "May respond to anti-androgen therapy (enzalutamide)",
        ]
    },
    "M": {
        "description": "Mesenchymal: EMT, cell motility, stem-like",
        "typical_age_range": (40, 70),
        "typical_age_median": 55,
        "typical_grade": 3,
        "typical_stage": [2, 3],
        "menopausal_bias": None,
        "histology_typical": ["ductal", "metaplastic"],
        "histology_atypical": ["lobular"],
        "node_positive_rate": 0.35,
        "expected_tmb": "moderate",
        "key_features": [
            "Epithelial-to-mesenchymal transition (EMT)",
            "Cell motility and invasion pathways",
            "Wnt/TGF-β signalling enrichment",
            "May respond to anti-angiogenic therapy",
        ]
    }
}


# ============================================================
# STAGE 1: CONFIDENCE CALIBRATION
# ============================================================

def calibrate_confidence(predicted_subtype, raw_confidence, raw_probabilities, clinical_data):
    """
    Adjust classification confidence based on clinical-subtype consistency.

    Args:
        predicted_subtype: str: "BL1", "BL2", "LAR", "M"
        raw_confidence: float: model's raw confidence (0-100)
        raw_probabilities: dict: {subtype: probability} for all classes
        clinical_data: dict: clinical features from the form

    Returns:
        dict with adjusted_confidence, consistency_score, flags, explanations
    """
    profile = SUBTYPE_PROFILES.get(predicted_subtype, {})
    if not profile or not clinical_data:
        return {
            "adjusted_confidence": raw_confidence,
            "consistency_score": None,
            "flags": [],
            "explanations": [],
            "clinical_agreement": "unknown"
        }

    agreements = []   # (weight, is_consistent, explanation)
    flags = []

    # --- Age consistency ---
    age = _to_float(clinical_data.get("age"))
    if age is not None:
        age_range = profile["typical_age_range"]
        age_median = profile["typical_age_median"]

        if age_range[0] <= age <= age_range[1]:
            agreements.append((2.0, True, f"Age {int(age)} is within typical range for {predicted_subtype} ({age_range[0]}-{age_range[1]})"))
        elif abs(age - age_median) <= 15:
            agreements.append((1.0, True, f"Age {int(age)} is near typical range for {predicted_subtype}"))
        else:
            agreements.append((2.0, False, f"Age {int(age)} is atypical for {predicted_subtype} (typical: {age_range[0]}-{age_range[1]})"))
            flags.append(f"age_atypical")

            # Specific LAR flag: LAR in young patient is unusual
            if predicted_subtype == "LAR" and age < 40:
                agreements.append((1.5, False, f"LAR is uncommon in patients under 40 (median age ~62)"))
                flags.append("lar_young_patient")

    # --- Grade consistency ---
    grade = _to_float(clinical_data.get("grade"))
    if grade is not None:
        expected = profile["typical_grade"]
        if grade == expected:
            agreements.append((1.5, True, f"Grade {int(grade)} is typical for {predicted_subtype}"))
        elif abs(grade - expected) == 1:
            agreements.append((0.5, True, f"Grade {int(grade)} is acceptable for {predicted_subtype}"))
        else:
            agreements.append((1.5, False, f"Grade {int(grade)} is unusual for {predicted_subtype} (typically grade {expected})"))
            flags.append("grade_atypical")

            # LAR with grade 3 or BL1 with grade 1 are red flags
            if predicted_subtype == "LAR" and grade == 3:
                agreements.append((1.0, False, "Grade 3 is uncommon for LAR subtype (usually grade 1-2)"))
            elif predicted_subtype == "BL1" and grade == 1:
                agreements.append((1.0, False, "Grade 1 is rare for BL1 (almost always grade 3)"))

    # --- Menopausal status consistency ---
    menopause = clinical_data.get("menopausal_status", "")
    bias = profile.get("menopausal_bias")
    if menopause and bias:
        if bias == "postmenopausal" and menopause == "premenopausal":
            agreements.append((1.5, False, f"{predicted_subtype} is predominantly postmenopausal: premenopausal is atypical"))
            flags.append("menopausal_atypical")
        elif bias == "premenopausal" and menopause == "postmenopausal":
            agreements.append((1.0, False, f"{predicted_subtype} is more common in premenopausal patients"))
            flags.append("menopausal_atypical")
        elif bias == menopause or (bias == "postmenopausal" and menopause == "postmenopausal"):
            agreements.append((1.5, True, f"Menopausal status ({menopause}) is consistent with {predicted_subtype}"))

    # --- Histology consistency ---
    histology = clinical_data.get("histologic_type", "")
    if histology:
        hist_lower = histology.lower()
        # Map form values to our categories
        hist_map = {
            "ductal_nst": "ductal", "lobular": "lobular", "apocrine": "apocrine",
            "metaplastic": "metaplastic", "medullary": "medullary"
        }
        hist_cat = hist_map.get(hist_lower, hist_lower)

        if hist_cat in profile.get("histology_typical", []):
            agreements.append((1.0, True, f"Histologic type ({histology}) is typical for {predicted_subtype}"))
        elif hist_cat in profile.get("histology_atypical", []):
            agreements.append((1.5, False, f"Histologic type ({histology}) is atypical for {predicted_subtype}"))
            flags.append("histology_atypical")

            # Strong signal: metaplastic almost always M, apocrine almost always LAR
            if hist_cat == "metaplastic" and predicted_subtype != "M":
                agreements.append((2.0, False, f"Metaplastic histology strongly suggests Mesenchymal subtype, not {predicted_subtype}"))
                flags.append("metaplastic_not_m")
            elif hist_cat == "apocrine" and predicted_subtype != "LAR":
                agreements.append((2.0, False, f"Apocrine histology strongly suggests LAR subtype, not {predicted_subtype}"))
                flags.append("apocrine_not_lar")

    # --- BRCA status ---
    brca = clinical_data.get("brca_status", "")
    if brca == "brca1_positive":
        if predicted_subtype == "BL1":
            agreements.append((1.5, True, "BRCA1 mutation is strongly associated with BL1 subtype"))
        elif predicted_subtype == "LAR":
            agreements.append((1.5, False, "BRCA1 mutation is uncommon in LAR subtype"))
            flags.append("brca1_lar")

    # --- Stage ---
    stage = _to_float(clinical_data.get("stage"))
    if stage is not None:
        typical_stages = profile.get("typical_stage", [])
        if int(stage) in typical_stages:
            agreements.append((0.5, True, f"Stage {int(stage)} is typical for {predicted_subtype}"))

    # --- Compute consistency score ---
    if not agreements:
        return {
            "adjusted_confidence": raw_confidence,
            "consistency_score": None,
            "flags": [],
            "explanations": [],
            "clinical_agreement": "insufficient_data"
        }

    total_weight = sum(w for w, _, _ in agreements)
    consistent_weight = sum(w for w, c, _ in agreements if c)
    consistency_score = consistent_weight / total_weight if total_weight > 0 else 0.5

    # Adjust confidence
    # consistency_score ranges from 0 (all disagreements) to 1 (all agreements)
    # We adjust confidence by up to ±10 percentage points
    adjustment = (consistency_score - 0.5) * 20  # -10 to +10
    adjusted_confidence = max(10.0, min(99.0, raw_confidence + adjustment))

    # Determine overall agreement level
    if consistency_score >= 0.75:
        agreement = "strong"
    elif consistency_score >= 0.5:
        agreement = "moderate"
    elif consistency_score >= 0.25:
        agreement = "weak"
    else:
        agreement = "conflicting"

    explanations = [exp for _, _, exp in agreements]

    return {
        "adjusted_confidence": round(adjusted_confidence, 1),
        "raw_confidence": raw_confidence,
        "consistency_score": round(consistency_score, 3),
        "flags": flags,
        "explanations": explanations,
        "clinical_agreement": agreement,
        "n_checks": len(agreements),
        "n_consistent": sum(1 for _, c, _ in agreements if c),
        "n_inconsistent": sum(1 for _, c, _ in agreements if not c),
    }


# ============================================================
# STAGE 2: PERSONALISED INSIGHTS
# ============================================================

def generate_insights(predicted_subtype, confidence, clinical_data):
    """
    Generate personalised clinical insights based on subtype + clinical data.

    Returns dict with risk_profile, treatment_insights, clinical_context, recommendations.
    """
    profile = SUBTYPE_PROFILES.get(predicted_subtype, {})
    insights = {
        "subtype_overview": profile.get("description", ""),
        "key_features": profile.get("key_features", []),
        "risk_profile": _build_risk_profile(predicted_subtype, clinical_data),
        "treatment_insights": _build_treatment_insights(predicted_subtype, clinical_data),
        "clinical_context": _build_clinical_context(predicted_subtype, clinical_data),
        "patient_vs_typical": _build_comparison(predicted_subtype, clinical_data),
    }
    return insights


def _build_risk_profile(subtype, clinical_data):
    """Build a personalised risk assessment."""
    risk_factors = []
    protective_factors = []

    age = _to_float(clinical_data.get("age"))
    grade = _to_float(clinical_data.get("grade"))
    stage = _to_float(clinical_data.get("stage"))
    tumor_size = _to_float(clinical_data.get("tumor_size"))
    node_status = clinical_data.get("lymph_node", "")
    lvi = clinical_data.get("lvi", "")
    ki67 = _to_float(clinical_data.get("ki67"))

    # Age-based risk
    if age is not None:
        if age < 35:
            risk_factors.append({"factor": "Young age at diagnosis", "detail": f"Age {int(age)}: younger patients may have more aggressive disease", "severity": "moderate"})
        elif age > 70:
            risk_factors.append({"factor": "Older age", "detail": f"Age {int(age)}: treatment tolerance considerations", "severity": "low"})

    # Grade
    if grade is not None:
        if grade == 3:
            risk_factors.append({"factor": "High tumour grade", "detail": "Grade 3: poorly differentiated, typically faster growing", "severity": "moderate"})
        elif grade == 1:
            protective_factors.append({"factor": "Low tumour grade", "detail": "Grade 1: well differentiated, generally slower growing"})

    # Stage
    if stage is not None:
        if stage >= 3:
            risk_factors.append({"factor": "Advanced stage", "detail": f"Stage {int(stage)}: locally advanced or metastatic", "severity": "high"})
        elif stage == 1:
            protective_factors.append({"factor": "Early stage", "detail": "Stage I: localised disease with better prognosis"})

    # Tumour size
    if tumor_size is not None:
        try:
            ts = float(tumor_size)
            if ts > 5:
                risk_factors.append({"factor": "Large tumour", "detail": f"{ts:.1f} cm: larger tumours associated with higher recurrence risk", "severity": "moderate"})
            elif ts <= 2:
                protective_factors.append({"factor": "Small tumour", "detail": f"{ts:.1f} cm: smaller tumours have better outcomes"})
        except (ValueError, TypeError):
            pass

    # Node status
    if node_status in ["n2", "n3"]:
        risk_factors.append({"factor": "Significant lymph node involvement", "detail": f"Node status {node_status.upper()}: indicates regional spread", "severity": "high"})
    elif node_status == "n0":
        protective_factors.append({"factor": "Node negative", "detail": "No lymph node involvement: favourable prognostic sign"})

    # LVI
    if lvi == "present":
        risk_factors.append({"factor": "Lymphovascular invasion", "detail": "LVI present: indicates potential for lymphatic spread", "severity": "moderate"})
    elif lvi == "absent":
        protective_factors.append({"factor": "No lymphovascular invasion", "detail": "LVI absent: favourable"})

    # Ki-67
    if ki67 is not None:
        if ki67 > 50:
            risk_factors.append({"factor": "Very high proliferation", "detail": f"Ki-67 {ki67:.0f}%: rapidly dividing tumour, but may respond well to chemotherapy", "severity": "moderate"})
        elif ki67 < 15:
            protective_factors.append({"factor": "Low proliferation", "detail": f"Ki-67 {ki67:.0f}%: slower growing tumour"})

    # Subtype-specific risk
    subtype_risk = {
        "BL1": {"overall": "moderate_high", "note": "BL1 has aggressive biology but responds well to chemotherapy: highest pCR rates among TNBC subtypes"},
        "BL2": {"overall": "moderate_high", "note": "BL2 has moderate chemosensitivity: growth factor pathway targeting may be beneficial"},
        "LAR": {"overall": "moderate", "note": "LAR has less aggressive biology than basal subtypes: anti-androgen therapy may be an option"},
        "M": {"overall": "moderate_high", "note": "Mesenchymal subtype has EMT-driven invasion: anti-angiogenic and EMT-targeted therapies under investigation"},
    }

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "subtype_risk": subtype_risk.get(subtype, {}),
        "overall_risk_level": _calculate_overall_risk(risk_factors, protective_factors, stage),
    }


def _build_treatment_insights(subtype, clinical_data):
    """Subtype-specific treatment pathway information."""
    treatments = {
        "BL1": {
            "primary_approaches": [
                {"name": "Platinum-based chemotherapy", "rationale": "BL1 has high DNA damage response: platinum agents exploit this vulnerability", "evidence": "Strong"},
                {"name": "PARP inhibitors", "rationale": "Especially if BRCA1/2 mutated: synthetic lethality with DNA repair deficiency", "evidence": "Strong (for BRCA+)"},
                {"name": "Immune checkpoint inhibitors", "rationale": "BL1 often has high tumour mutational burden and immune infiltration", "evidence": "Moderate"},
            ],
            "neoadjuvant_note": "BL1 has the highest pathological complete response (pCR) rate to neoadjuvant chemotherapy among TNBC subtypes (~50-52%)",
        },
        "BL2": {
            "primary_approaches": [
                {"name": "Standard chemotherapy", "rationale": "Anthracycline/taxane-based regimens as first-line", "evidence": "Standard of care"},
                {"name": "Growth factor receptor inhibitors", "rationale": "BL2 shows enrichment in EGFR/IGF1R signalling pathways", "evidence": "Investigational"},
                {"name": "Metabolic pathway targeting", "rationale": "BL2 shows glycolytic pathway activation", "evidence": "Early research"},
            ],
            "neoadjuvant_note": "BL2 has a lower pCR rate than BL1 (~18-22%): additional targeted approaches may be needed",
        },
        "LAR": {
            "primary_approaches": [
                {"name": "Anti-androgen therapy (e.g. enzalutamide)", "rationale": "LAR expresses high androgen receptor: direct therapeutic target", "evidence": "Phase II trials"},
                {"name": "PI3K/AKT/mTOR inhibitors", "rationale": "LAR frequently has PIK3CA mutations and AKT pathway activation", "evidence": "Moderate"},
                {"name": "CDK4/6 inhibitors", "rationale": "Luminal-like expression pattern may respond similar to HR+ breast cancer", "evidence": "Investigational"},
            ],
            "neoadjuvant_note": "LAR has the lowest pCR rate to standard chemotherapy (~10%): targeted therapy may be more appropriate",
        },
        "M": {
            "primary_approaches": [
                {"name": "Anti-angiogenic therapy (e.g. bevacizumab)", "rationale": "Mesenchymal tumours show angiogenesis pathway enrichment", "evidence": "Moderate"},
                {"name": "EMT-targeted therapies", "rationale": "High EMT activity is a defining feature of mesenchymal subtype", "evidence": "Investigational"},
                {"name": "TGF-β pathway inhibitors", "rationale": "TGF-β signalling drives EMT and invasion in M subtype", "evidence": "Early research"},
            ],
            "neoadjuvant_note": "Mesenchymal subtype has moderate chemosensitivity (~23-31% pCR): combination approaches under investigation",
        },
    }

    result = treatments.get(subtype, {})

    # Add BRCA-specific treatment note
    brca = clinical_data.get("brca_status", "")
    if brca in ["brca1_positive", "brca2_positive"]:
        result["brca_note"] = f"Patient is {brca.replace('_', ' ').title()}: PARP inhibitors (olaparib, talazoparib) are specifically indicated for BRCA-mutated TNBC regardless of subtype."

    # Add stage-specific note
    stage = _to_float(clinical_data.get("stage"))
    if stage is not None:
        if stage <= 2:
            result["stage_note"] = "Early-stage disease: neoadjuvant chemotherapy may enable breast-conserving surgery and provides valuable prognostic information through pathological response."
        elif stage == 3:
            result["stage_note"] = "Locally advanced disease: neoadjuvant chemotherapy is standard to downstage the tumour before surgery."
        elif stage == 4:
            result["stage_note"] = "Metastatic disease: treatment goals focus on disease control, quality of life, and clinical trial participation."

    return result


def _build_clinical_context(subtype, clinical_data):
    """
    Generate contextual clinical notes combining subtype + clinical picture.
    """
    notes = []
    profile = SUBTYPE_PROFILES.get(subtype, {})

    age = _to_float(clinical_data.get("age"))
    grade = _to_float(clinical_data.get("grade"))
    ki67 = _to_float(clinical_data.get("ki67"))
    ar = clinical_data.get("ar_status", "")
    tils = _to_float(clinical_data.get("tils"))

    # Subtype + age context
    if age is not None:
        median = profile.get("typical_age_median", 55)
        if abs(age - median) <= 5:
            notes.append(f"Patient age ({int(age)}) is close to the median age for {subtype} patients (~{median})")
        elif age < median - 10:
            notes.append(f"Patient is younger ({int(age)}) than typical {subtype} patients (median ~{median}): may indicate more aggressive biology")
        elif age > median + 10:
            notes.append(f"Patient is older ({int(age)}) than typical {subtype} patients (median ~{median})")

    # Ki-67 + subtype context
    if ki67 is not None:
        if subtype == "BL1" and ki67 > 40:
            notes.append(f"High Ki-67 ({ki67:.0f}%) is consistent with BL1's high-proliferation signature: good chemosensitivity expected")
        elif subtype == "LAR" and ki67 > 40:
            notes.append(f"Ki-67 {ki67:.0f}% is unusually high for LAR subtype: consider whether classification needs clinical review")
        elif subtype == "LAR" and ki67 < 20:
            notes.append(f"Low Ki-67 ({ki67:.0f}%) is consistent with LAR's lower proliferation rate")

    # AR + subtype context
    if ar:
        if ar in ["positive_high", "positive_low"] and subtype == "LAR":
            notes.append("AR positivity is consistent with LAR classification: anti-androgen therapy may be considered")
        elif ar in ["positive_high", "positive_low"] and subtype != "LAR":
            notes.append(f"AR positivity is detected but gene expression classifies as {subtype}: the molecular subtype may differ from IHC surrogate")
        elif ar == "negative" and subtype == "LAR":
            notes.append("AR negative by IHC but gene expression classifies as LAR: gene-level androgen pathway activation may not be captured by IHC")

    # TILs context
    if tils is not None:
        if tils >= 30:
            notes.append(f"High stromal TILs ({tils:.0f}%): associated with better prognosis and potential immunotherapy benefit")
            if subtype in ["BL1", "BL2"]:
                notes.append("High immune infiltration in basal-like tumour supports potential checkpoint inhibitor response")
        elif tils < 10:
            notes.append(f"Low TILs ({tils:.0f}%): immune-cold tumour, immunotherapy less likely to benefit as monotherapy")

    return notes


def _build_comparison(subtype, clinical_data):
    """
    Compare patient's clinical features against typical subtype profile.
    Returns a list of comparison items for display.
    """
    profile = SUBTYPE_PROFILES.get(subtype, {})
    comparisons = []

    age = _to_float(clinical_data.get("age"))
    if age is not None:
        typical = profile.get("typical_age_range", (35, 70))
        comparisons.append({
            "feature": "Age at diagnosis",
            "patient_value": f"{int(age)} years",
            "typical_value": f"{typical[0]}-{typical[1]} years (median ~{profile.get('typical_age_median', 'N/A')})",
            "status": "typical" if typical[0] <= age <= typical[1] else "atypical"
        })

    grade = _to_float(clinical_data.get("grade"))
    if grade is not None:
        typical_grade = profile.get("typical_grade", "N/A")
        comparisons.append({
            "feature": "Tumour grade",
            "patient_value": f"Grade {int(grade)}",
            "typical_value": f"Grade {typical_grade}",
            "status": "typical" if grade == typical_grade else "atypical"
        })

    menopause = clinical_data.get("menopausal_status", "")
    bias = profile.get("menopausal_bias")
    if menopause and bias:
        comparisons.append({
            "feature": "Menopausal status",
            "patient_value": menopause.replace("_", " ").title(),
            "typical_value": f"Predominantly {bias}",
            "status": "typical" if menopause == bias else "atypical"
        })

    stage = _to_float(clinical_data.get("stage"))
    if stage is not None:
        typical_stages = profile.get("typical_stage", [])
        comparisons.append({
            "feature": "Clinical stage",
            "patient_value": f"Stage {int(stage)}",
            "typical_value": f"Typically Stage {'-'.join(str(s) for s in typical_stages)}",
            "status": "typical" if int(stage) in typical_stages else "atypical"
        })

    return comparisons


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _to_float(value):
    """Safely convert to float."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _calculate_overall_risk(risk_factors, protective_factors, stage):
    """Calculate a simple overall risk level."""
    high_risks = sum(1 for r in risk_factors if r.get("severity") == "high")
    mod_risks = sum(1 for r in risk_factors if r.get("severity") == "moderate")
    protectives = len(protective_factors)

    score = high_risks * 3 + mod_risks * 2 - protectives * 1.5

    if stage is not None:
        stage = float(stage)
        if stage >= 4:
            return "high"
        elif stage >= 3:
            score += 2

    if score >= 5:
        return "high"
    elif score >= 2:
        return "moderate"
    else:
        return "low"
