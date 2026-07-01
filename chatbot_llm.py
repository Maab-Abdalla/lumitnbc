"""
LumiTNBC chatbot: lightweight RAG (retrieval) + Anthropic (Claude) answer.

Flow:
  1. Load the curated knowledge base from chatbot_kb.json (id/category/tags/
     title/content chunks).
  2. retrieve(question) scores chunks by tag/title/content overlap and returns
     the top few.
  3. generate_reply(question, history) injects those chunks into a scoped system
     prompt and calls the Anthropic API with MODERATE medical guardrails:
       - educational TNBC / app information is fine
       - treatment, dosing, prognosis, "what should I do" -> declined, redirect
         to the user's care team
       - no patient-specific medical advice

Degrades gracefully: if no ANTHROPIC_API_KEY is set, or the package/network is
unavailable, is_enabled() is False and the caller falls back to the offline
keyword bot. This module never raises to the caller.
"""

import os
import re
import json

# Load .env if present so ANTHROPIC_API_KEY is available even when this module
# is used outside the Flask app (e.g. scripts, tests). Safe to call repeatedly.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "chatbot_kb.json")

CHAT_MODEL = os.environ.get("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 220
TOP_K = 3  # how many KB chunks to inject


def _load_kb():
    try:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chunks", [])
    except Exception:
        return []


KNOWLEDGE_BASE = _load_kb()


def is_enabled():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def retrieve(question, top_k=TOP_K):
    """Score each KB chunk against the question by tag/title/content overlap.
    Returns the top_k chunks (highest score first), score > 0 only."""
    q_tokens = _tokenize(question)
    q_lower = (question or "").lower()
    if not q_tokens:
        return []
    scored = []
    for chunk in KNOWLEDGE_BASE:
        score = 0
        for tag in chunk.get("tags", []):
            tl = tag.lower()
            if tl in q_lower:
                score += 3
            elif _tokenize(tag) & q_tokens:
                score += 1
        score += len(_tokenize(chunk.get("title", "")) & q_tokens) * 2
        score += len(_tokenize(chunk.get("content", "")) & q_tokens)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [c for _, c in scored[:top_k]]


SYSTEM_PROMPT = (
    "You are Lumi, a friendly assistant inside the LumiTNBC web app, which helps "
    "people understand triple-negative breast cancer (TNBC) molecular subtypes.\n\n"
    "SCOPE: Answer questions about TNBC in general, the four molecular subtypes "
    "(BL1, BL2, LAR, M), how the app works, how to read results, and how "
    "clinical-trial matching works.\n\n"
    "ANSWER STYLE (follow exactly):\n"
    "- Be clear, organised, and straight to the point. Lead with the direct answer.\n"
    "- Keep it to 2-3 short sentences (about 50 words). Stop once the question is "
    "answered; do not pad or over-explain.\n"
    "- Write in plain text ONLY. Do NOT use Markdown or any special formatting: no "
    "asterisks (*), no bold, no italics, no headings, no bullet characters, no "
    "backticks.\n"
    "- Do NOT use em dashes or en dashes (the long dash characters). Use a comma, a "
    "full stop, or the word 'to' for ranges instead.\n"
    "- If you list a few items, separate them with commas in a normal sentence.\n\n"
    "SAFETY RULES (important):\n"
    "- You may give general, educational information.\n"
    "- You must NOT give personalised medical advice, treatment or medication "
    "recommendations, dosing, or prognosis ('how long', 'will I survive', "
    "'what should I do', 'is this treatment right for me'). For those, gently "
    "decline in one sentence and tell the person to discuss it with their own "
    "oncologist or care team.\n"
    "- Do not diagnose. Do not interpret an individual's specific medical results.\n"
    "- If a question is outside TNBC / this app, say briefly it's outside what you "
    "can help with here.\n"
    "- Never invent statistics, study results, or clinical-trial details.\n\n"
    "Ground your answer in the provided CONTEXT when it is relevant. If the "
    "context does not cover the question and it is still in scope, give a brief "
    "general answer without fabricating specifics. LumiTNBC is a decision-support "
    "tool, not a substitute for professional medical care."
)


# Audience guidance appended per user role. This is what fixes answers being
# pitched at the wrong level: a patient asking "what data do I need?" should
# hear "gene expression data from your tumour sample; your care team handles
# the technical preparation", NOT log2-FPKM/TPM/normalisation/feature-matching.
ROLE_GUIDANCE = {
    "patient": (
        "\n\nAUDIENCE: You are talking to a PATIENT, not a clinician.\n"
        "- Answer in everyday language a non-scientist understands.\n"
        "- Do NOT include technical data-handling detail (file formats, "
        "log2/FPKM/TPM, normalisation, raw read counts, matching training "
        "features, preprocessing pipelines). These are handled by the patient's "
        "care team or laboratory, not the patient.\n"
        "- If the CONTEXT contains such technical detail, do NOT repeat it. "
        "Instead, summarise at a high level and say their doctor, care team, or "
        "lab takes care of preparing and formatting the data.\n"
        "- Keep it reassuring and simple. Example: if asked what data is needed, "
        "say it is gene expression information from their tumour sample, and that "
        "their care team handles the technical side."
    ),
    "provider": (
        "\n\nAUDIENCE: You are talking to a HEALTHCARE PROVIDER or researcher.\n"
        "- Precise clinical and technical detail from the CONTEXT is appropriate "
        "(input formats, normalisation, features), but stay concise."
    ),
    "admin": (
        "\n\nAUDIENCE: You are talking to a system ADMINISTRATOR.\n"
        "- Focus on how the app and model work; clinical specifics are secondary."
    ),
}


def _clean_reply(text):
    """Safety net: strip Markdown and long dashes in case the model slips.
    Keeps the answer as plain, readable text regardless of prompt adherence."""
    if not text:
        return text
    # Remove bold/italic markers (**x**, __x__, *x*, _x_) keeping inner text.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Strip leading heading hashes and list bullets at line starts.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*\u2022]\s+", "", text)
    # Remove stray backticks.
    text = text.replace("`", "")
    # Replace em/en dashes with a comma-space (or plain space if already spaced).
    text = re.sub(r"\s*[\u2014\u2013]\s*", ", ", text)
    # Any leftover lone asterisks.
    text = text.replace("*", "")
    # Collapse excess whitespace/blank lines.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_reply(question, history=None, role="patient"):
    """Return {"enabled": bool, "reply": str|None, "error": str|None}.
    `role` ("patient" | "provider" | "admin") tailors the answer's depth and
    audience. Never raises; on any failure returns enabled=False so the caller
    can fall back to the offline keyword bot."""
    if not is_enabled():
        return {"enabled": False, "reply": None, "error": None}

    try:
        import anthropic
    except ImportError:
        return {"enabled": False, "reply": None,
                "error": "anthropic package not installed"}

    chunks = retrieve(question)
    if chunks:
        context = "\n\n".join(
            f"[{c.get('title', c.get('id', 'info'))}]\n{c.get('content', '')}"
            for c in chunks
        )
    else:
        context = "(No specific knowledge-base entry matched this question.)"

    # Anthropic takes the system prompt as a separate parameter; messages hold
    # only user/assistant turns.
    messages = []
    for turn in (history or [])[-4:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": (f"CONTEXT (trusted knowledge base):\n{context}\n\n"
                    f"USER QUESTION:\n{question}")
    })

    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        system_prompt = SYSTEM_PROMPT + ROLE_GUIDANCE.get(role, ROLE_GUIDANCE["patient"])
        msg = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            system=system_prompt,
            messages=messages,
        )
        text = "".join(b.text for b in msg.content
                       if getattr(b, "type", None) == "text").strip()
        text = _clean_reply(text)
        return {"enabled": True, "reply": text or None, "error": None}
    except Exception as exc:
        return {"enabled": False, "reply": None, "error": str(exc)}
