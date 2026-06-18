"""
LumiTNBC: Clinical rule-based classification utility.
Extracted from app.py to avoid circular imports.
"""


def clinical_rule_based_classification(clinical_data: dict) -> dict:
    """
    Rule-based TNBC subtype approximation from IHC / clinical markers.
    Based on FUSCC clinical surrogates (Jiang et al. 2019).
    """
    scores = {"BL1": 0, "BL2": 0, "M": 0, "LAR": 0}
    reasons = {"BL1": [], "BL2": [], "M": [], "LAR": []}

    # AR status
    ar = clinical_data.get("ar_status", "")
    if ar == "positive_high":
        scores["LAR"] += 5
        reasons["LAR"].append("Strong AR positivity (key LAR marker)")
    elif ar == "positive_low":
        scores["LAR"] += 3
        reasons["LAR"].append("Weak AR positivity")
    elif ar == "negative":
        for s in ["BL1", "BL2", "M"]:
            scores[s] += 1
        reasons["BL1"].append("AR negative (consistent with basal)")

    # Ki-67
    ki67 = _to_float(clinical_data.get("ki67"))
    if ki67 is not None:
        if ki67 >= 50:
            scores["BL1"] += 3
            reasons["BL1"].append(f"Very high Ki-67 ({ki67}%): high proliferation")
        elif ki67 >= 30:
            scores["BL1"] += 2; scores["BL2"] += 1
            reasons["BL1"].append(f"High Ki-67 ({ki67}%)")
        elif ki67 < 20:
            scores["LAR"] += 2
            reasons["LAR"].append(f"Low Ki-67 ({ki67}%): typical of LAR")

    # EGFR
    if clinical_data.get("egfr_status") == "positive":
        scores["BL1"] += 2; scores["BL2"] += 2
        reasons["BL1"].append("EGFR positive (basal-like marker)")
        reasons["BL2"].append("EGFR positive (basal-like marker)")

    # CK5/6
    if clinical_data.get("ck56_status") == "positive":
        scores["BL1"] += 2; scores["BL2"] += 2
        reasons["BL1"].append("CK5/6 positive (basal cytokeratin)")
        reasons["BL2"].append("CK5/6 positive (basal cytokeratin)")

    # PD-L1
    if clinical_data.get("pdl1_status") == "positive_high":
        scores["BL1"] += 1
        reasons["BL1"].append("PD-L1 positive (immune-active)")

    # TILs
    tils = _to_float(clinical_data.get("tils"))
    if tils is not None:
        if tils >= 30:
            scores["BL1"] += 1
            reasons["BL1"].append(f"High TILs ({tils}%): immune-rich")
        elif tils < 10:
            scores["LAR"] += 1; scores["M"] += 1
            reasons["M"].append(f"Low TILs ({tils}%)")

    # Histology
    hist = clinical_data.get("histologic_type", "")
    if hist == "apocrine":
        scores["LAR"] += 3
        reasons["LAR"].append("Apocrine histology (strongly associated with LAR)")
    elif hist == "metaplastic":
        scores["M"] += 3
        reasons["M"].append("Metaplastic histology (associated with mesenchymal)")
    elif hist == "medullary":
        scores["BL1"] += 2
        reasons["BL1"].append("Medullary-like histology (immune-rich basal)")

    # Grade
    grade = clinical_data.get("grade", "")
    if grade == "3":
        scores["BL1"] += 1
        reasons["BL1"].append("Grade 3 (high-grade, typical of BL1)")
    elif grade == "1":
        scores["LAR"] += 1
        reasons["LAR"].append("Grade 1 (lower grade, more common in LAR)")

    # Menopausal status
    menopause = clinical_data.get("menopausal_status", "")
    if menopause == "postmenopausal":
        scores["LAR"] += 1
        reasons["LAR"].append("Postmenopausal (LAR more common in older patients)")
    elif menopause == "premenopausal":
        scores["BL1"] += 1
        reasons["BL1"].append("Premenopausal (basal-like more common)")

    # BRCA
    if clinical_data.get("brca_status") == "brca1_positive":
        scores["BL1"] += 2
        reasons["BL1"].append("BRCA1 mutation (strong BL1 association)")

    # LVI
    if clinical_data.get("lvi") == "present":
        scores["BL1"] += 1
        reasons["BL1"].append("LVI present (aggressive, basal-like)")

    max_score = max(scores.values())
    if max_score == 0:
        predicted, confidence = "BL1", 35.0
        method_note = "Insufficient clinical data: defaulting to most common subtype"
    else:
        predicted = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = min(round((max_score / total) * 100, 1) if total > 0 else 40.0, 82.0)
        method_note = "Rule-based clinical approximation (IHC/pathology surrogates)"

    return {
        "subtype": predicted,
        "confidence": confidence,
        "method": "clinical_rules",
        "method_note": method_note,
        "scores": scores,
        "top_reasons": reasons[predicted][:5],
    }


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
