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
MAX_TOKENS = 400
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
    "clinical-trial matching works. Keep answers short (2-4 sentences), warm, and "
    "in plain language.\n\n"
    "SAFETY RULES (important):\n"
    "- You may give general, educational information.\n"
    "- You must NOT give personalised medical advice, treatment or medication "
    "recommendations, dosing, or prognosis ('how long', 'will I survive', "
    "'what should I do', 'is this treatment right for me'). For those, gently "
    "decline and tell the person to discuss it with their own oncologist or care "
    "team.\n"
    "- Do not diagnose. Do not interpret an individual's specific medical results.\n"
    "- If a question is outside TNBC / this app, say it's outside what you can "
    "help with here.\n"
    "- Never invent statistics, study results, or clinical-trial details.\n\n"
    "Ground your answer in the provided CONTEXT when it is relevant. If the "
    "context does not cover the question and it is still in scope, you may give a "
    "brief general answer, but do not fabricate specifics. Always be clear that "
    "LumiTNBC is a decision-support tool, not a substitute for professional "
    "medical care."
)


def generate_reply(question, history=None):
    """Return {"enabled": bool, "reply": str|None, "error": str|None}.
    Never raises; on any failure returns enabled=False so the caller can fall
    back to the offline keyword bot."""
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
        msg = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        text = "".join(b.text for b in msg.content
                       if getattr(b, "type", None) == "text").strip()
        return {"enabled": True, "reply": text or None, "error": None}
    except Exception as exc:
        return {"enabled": False, "reply": None, "error": str(exc)}
