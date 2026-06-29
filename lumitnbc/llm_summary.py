"""
LLM layer for LumiTNBC: personalised patient summary.

This module is OPTIONAL and disabled by default. It generates a short,
plain-language summary paragraph tailored to a single patient's result by
calling the Anthropic API.

To enable:
    1. pip install anthropic   (already listed in requirements.txt)
    2. Set the ANTHROPIC_API_KEY environment variable.

When no key is present, is_enabled() returns False and the results page
simply does not show the "Your Personal Summary" card, with no cost and no errors.

The prompt is built ONLY from the model's own output (subtype, confidence,
top SHAP features) plus any clinical values the patient already provided.
No raw gene-level data is sent beyond the top feature labels.
"""

import os

# Load .env if present so the key is available even when imported outside the
# Flask app. Safe to call repeatedly.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Model used for the one-shot summary. Small/fast model is plenty here.
SUMMARY_MODEL = os.environ.get("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 320


def is_enabled():
    """True only when an API key is configured in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _plain_label(gene, gene_label_fn):
    label, _ = gene_label_fn(gene)
    return label


def build_prompt(result, gene_label_fn):
    """Assemble the user prompt from the patient's result + clinical data."""
    subtype = result.get("subtype", "unknown")
    confidence = result.get("confidence", "?")
    full_name = result.get("subtype_full_name") or subtype

    # Top 3 SHAP features as plain-language labels (no raw gene symbols).
    feats = (result.get("shap_features") or [])[:3]
    signals = []
    for f in feats:
        label = _plain_label(f.get("gene", ""), gene_label_fn)
        direction = "supporting" if f.get("shap_value", 0) > 0 else "arguing against"
        signals.append(f"{label} ({direction} the classification)")
    signals_text = "; ".join(signals) if signals else "general molecular patterns"

    # Clinical context, if the patient supplied it.
    clinical = result.get("clinical_data") or {}
    clinical_bits = []
    for key, label in [
        ("age", "age"), ("menopausal", "menopausal status"),
        ("stage", "cancer stage"), ("grade", "tumour grade"),
        ("tumor_size", "tumour size"), ("node_status", "lymph node status"),
    ]:
        if clinical.get(key) not in (None, "", "unknown"):
            clinical_bits.append(f"{label}: {clinical[key]}")
    clinical_text = ("; ".join(clinical_bits) if clinical_bits
                     else "no additional clinical details provided")

    return (
        "You are explaining a triple-negative breast cancer (TNBC) molecular "
        "subtype result to the patient it belongs to. Write 2-3 warm, clear "
        "sentences in plain English (no jargon, no gene names, no statistics "
        "beyond the confidence already given). Be reassuring but accurate, and "
        "remind them gently to discuss it with their care team.\n\n"
        f"Subtype: {full_name} ({subtype})\n"
        f"Model confidence: {confidence}%\n"
        f"Strongest signals: {signals_text}\n"
        f"Patient clinical context: {clinical_text}\n\n"
        "Write only the summary paragraph, nothing else."
    )


def generate_summary(result, gene_label_fn):
    """
    Return a dict: {"enabled": bool, "summary": str|None, "error": str|None}.

    Never raises; on any failure it degrades to enabled=False so the UI hides
    the card rather than showing a broken state.
    """
    if not is_enabled():
        return {"enabled": False, "summary": None, "error": None}

    try:
        import anthropic  # imported lazily so the app runs without the package
    except ImportError:
        return {"enabled": False, "summary": None,
                "error": "anthropic package not installed"}

    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        msg = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.4,
            messages=[{"role": "user",
                       "content": build_prompt(result, gene_label_fn)}],
        )
        text = "".join(block.text for block in msg.content
                       if getattr(block, "type", None) == "text").strip()
        return {"enabled": True, "summary": text or None, "error": None}
    except Exception as exc:  # network, auth, rate-limit, etc.
        return {"enabled": False, "summary": None, "error": str(exc)}
