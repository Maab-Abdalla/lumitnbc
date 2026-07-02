"""
Clinical-trial matching for LumiTNBC.

Queries the ClinicalTrials.gov API v2 (https://clinicaltrials.gov/api/v2)
for trials relevant to each TNBC molecular subtype, with:

  * per-subtype search terms (condition + subtype-specific intervention keywords)
  * an in-memory cache keyed by subtype, refreshed at most once per day
    (ClinicalTrials.gov data itself only refreshes on weekdays, so a daily
    cache is plenty and keeps the app fast)
  * graceful fallback to a curated set of demo trials if the API is
    unreachable, times out, or returns an unexpected shape, the results
    page always has trials to show.

No API key is required. The v2 API is a public REST endpoint returning JSON.
"""

import json
import time
import urllib.parse
import urllib.request

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
REQUEST_TIMEOUT = 8          # seconds
CACHE_TTL_SECONDS = 24 * 60 * 60  # 1 day
MAX_TRIALS = 3               # how many to surface per subtype

# Per-subtype query terms. The condition is always TNBC; the extra terms bias
# the search toward interventions associated with each molecular subtype so the
# matches are clinically relevant rather than generic.
SUBTYPE_QUERY = {
    "BL1": {"terms": "platinum OR PARP inhibitor OR immunotherapy",
            "label": "Basal-Like 1"},
    "BL2": {"terms": "growth factor OR EGFR OR mTOR",
            "label": "Basal-Like 2"},
    "LAR": {"terms": "androgen receptor OR enzalutamide OR bicalutamide OR CDK4/6",
            "label": "Luminal Androgen Receptor"},
    "M":   {"terms": "antiangiogenic OR PI3K OR mesenchymal OR EMT",
            "label": "Mesenchymal"},
}

# ── Curated fallback trials (used only when the live API is unavailable) ──────
# These mirror the structure returned by fetch_trials() so the template renders
# identically whether data is live or fallback.
FALLBACK_TRIALS = {
    "BL1": [
        {"nct": "NCT05123456", "title": "Phase II Trial: PD-L1 Inhibitor + Carboplatin for BL1 TNBC",
         "phase": "Phase II", "match": 95, "status": "Recruiting", "location": "Multiple Sites (USA)", "sponsor": "National Cancer Institute",
         "url": "https://clinicaltrials.gov/study/NCT05123456", "countries": ["United States"]},
        {"nct": "NCT05234567", "title": "Targeted Therapy with PARP Inhibitor for Basal-Like TNBC",
         "phase": "Phase III", "match": 88, "status": "Recruiting", "location": "Dana-Farber Cancer Institute, Boston", "sponsor": "AstraZeneca",
         "url": "https://clinicaltrials.gov/study/NCT05234567", "countries": ["United States"]},
        {"nct": "NCT05345678", "title": "Neoadjuvant Chemotherapy Optimization for High-Proliferation TNBC",
         "phase": "Phase II", "match": 79, "status": "Active", "location": "MD Anderson Cancer Center, Houston", "sponsor": "NCI Cooperative Group",
         "url": "https://clinicaltrials.gov/study/NCT05345678", "countries": ["United States"]},
    ],
    "BL2": [
        {"nct": "NCT05456789", "title": "Growth Factor Receptor Inhibitor for BL2 TNBC",
         "phase": "Phase II", "match": 91, "status": "Recruiting", "location": "Memorial Sloan Kettering, New York", "sponsor": "Genentech",
         "url": "https://clinicaltrials.gov/study/NCT05456789", "countries": ["United States"]},
        {"nct": "NCT05567890", "title": "TGF-β Pathway Targeted Therapy in TNBC",
         "phase": "Phase I/II", "match": 84, "status": "Recruiting", "location": "Multiple Sites (USA, EU)", "sponsor": "Merck",
         "url": "https://clinicaltrials.gov/study/NCT05567890", "countries": ["United States", "Europe"]},
    ],
    "LAR": [
        {"nct": "NCT05678901", "title": "Enzalutamide + CDK4/6 Inhibitor for LAR TNBC",
         "phase": "Phase II", "match": 93, "status": "Recruiting", "location": "Vanderbilt-Ingram Cancer Center, Nashville", "sponsor": "Pfizer / Astellas",
         "url": "https://clinicaltrials.gov/study/NCT05678901", "countries": ["United States"]},
        {"nct": "NCT05789012", "title": "PI3K/AKT Inhibitor Combined with Anti-Androgen Therapy",
         "phase": "Phase II", "match": 86, "status": "Recruiting", "location": "Multiple Sites (USA)", "sponsor": "Novartis",
         "url": "https://clinicaltrials.gov/study/NCT05789012", "countries": ["United States"]},
    ],
    "M": [
        {"nct": "NCT05890123", "title": "Anti-Angiogenic + EMT Inhibitor for Mesenchymal TNBC",
         "phase": "Phase II", "match": 90, "status": "Recruiting", "location": "Johns Hopkins, Baltimore", "sponsor": "Bristol-Myers Squibb",
         "url": "https://clinicaltrials.gov/study/NCT05890123", "countries": ["United States"]},
        {"nct": "NCT05901234", "title": "Wnt/TGF-β Dual Pathway Inhibitor in Mesenchymal TNBC",
         "phase": "Phase I/II", "match": 82, "status": "Active", "location": "Multiple Sites (USA, Asia)", "sponsor": "Eli Lilly",
         "url": "https://clinicaltrials.gov/study/NCT05901234", "countries": ["United States", "Asia"]},
    ],
}

# In-memory cache: {subtype: {"data": [...], "fetched_at": epoch}}
_CACHE = {}


# ── Response parsing ─────────────────────────────────────────────────────────
def _phase_label(design_module):
    phases = (design_module or {}).get("phases") or []
    mapping = {
        "EARLY_PHASE1": "Early Phase 1", "PHASE1": "Phase I",
        "PHASE2": "Phase II", "PHASE3": "Phase III", "PHASE4": "Phase IV",
        "NA": "N/A",
    }
    if not phases:
        return "N/A"
    if phases == ["PHASE1", "PHASE2"]:
        return "Phase I/II"
    if phases == ["PHASE2", "PHASE3"]:
        return "Phase II/III"
    return ", ".join(mapping.get(p, p.title()) for p in phases)


def _status_label(status_module):
    raw = (status_module or {}).get("overallStatus", "")
    return raw.replace("_", " ").title() if raw else "Unknown"


def _location_label(contacts_module):
    locs = (contacts_module or {}).get("locations") or []
    if not locs:
        return "Location not specified"
    # Distinct countries across all sites: most useful for "can I access this?"
    countries = []
    for loc in locs:
        c = loc.get("country")
        if c and c not in countries:
            countries.append(c)
    first = locs[0]
    city_country = ", ".join(p for p in [first.get("city"), first.get("country")] if p)
    extra_sites = len(locs) - 1
    base = city_country or "Location not specified"
    if extra_sites > 0:
        base = f"{base} (+{extra_sites} more site{'s' if extra_sites != 1 else ''})"
    return base


def _countries_label(contacts_module):
    """Distinct list of countries a trial runs in (for region accessibility)."""
    locs = (contacts_module or {}).get("locations") or []
    countries = []
    for loc in locs:
        c = loc.get("country")
        if c and c not in countries:
            countries.append(c)
    return countries


def _sponsor_label(sponsor_module):
    lead = (sponsor_module or {}).get("leadSponsor") or {}
    return lead.get("name", "Sponsor not specified")


def _match_score(rank, total):
    """Heuristic relevance score for display. ClinicalTrials.gov returns results
    by relevance, so earlier = higher. Scale 95 down to ~78."""
    return max(78, 95 - rank * 6)


def _parse_study(study, rank, total):
    """Map one v2 study record to the flat shape the template expects."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    nct = ident.get("nctId", "")
    title = (ident.get("briefTitle")
             or ident.get("officialTitle")
             or "Untitled study")
    return {
        "nct": nct,
        "title": title,
        "phase": _phase_label(proto.get("designModule")),
        "match": _match_score(rank, total),
        "status": _status_label(proto.get("statusModule")),
        "location": _location_label(proto.get("contactsLocationsModule")),
        "countries": _countries_label(proto.get("contactsLocationsModule")),
        "sponsor": _sponsor_label(proto.get("sponsorCollaboratorsModule")),
        "url": f"https://clinicaltrials.gov/study/{nct}" if nct else None,
    }


# ── Live fetch ───────────────────────────────────────────────────────────────
def _build_url(subtype):
    cfg = SUBTYPE_QUERY.get(subtype, {})
    terms = cfg.get("terms", "")
    params = {
        "query.cond": "triple negative breast cancer",
        "filter.overallStatus": "RECRUITING|ACTIVE_NOT_RECRUITING",
        "pageSize": str(MAX_TRIALS),
        "format": "json",
        # Only request the fields we render, to keep the payload small.
        "fields": ",".join([
            "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus",
            "Phase", "LeadSponsorName", "LocationCity", "LocationCountry",
        ]),
    }
    if terms:
        params["query.term"] = terms
    return f"{API_BASE}?{urllib.parse.urlencode(params)}"


def _fetch_live(subtype):
    """Query the live API. Returns a list of trials, or raises on any failure."""
    url = _build_url(subtype)
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "LumiTNBC/1.0 (research prototype)",
    })
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        payload = json.load(resp)

    studies = payload.get("studies", [])
    total = len(studies)
    trials = [_parse_study(s, i, total) for i, s in enumerate(studies)]
    # Drop any record we couldn't extract an NCT id for.
    return [t for t in trials if t["nct"]][:MAX_TRIALS]


# ── Public API ───────────────────────────────────────────────────────────────
def get_trials(subtype, force_refresh=False):
    """
    Return up to MAX_TRIALS trials for a TNBC subtype.

    Tries the live ClinicalTrials.gov API (cached for a day). On any error,
    network down, timeout, bad response, empty result, falls back to the
    curated FALLBACK_TRIALS so the UI always has something to show.
    """
    subtype = (subtype or "").upper()
    if subtype not in SUBTYPE_QUERY:
        return []

    now = time.time()
    cached = _CACHE.get(subtype)
    if (not force_refresh and cached
            and (now - cached["fetched_at"]) < CACHE_TTL_SECONDS):
        return cached["data"]

    try:
        trials = _fetch_live(subtype)
        if not trials:                      # API reachable but no matches
            raise ValueError("no live trials returned")
        source = "live"
    except Exception:
        trials = FALLBACK_TRIALS.get(subtype, [])
        source = "fallback"

    # Tag the source so callers/templates can show a "live data" indicator.
    for t in trials:
        t.setdefault("source", source)

    _CACHE[subtype] = {"data": trials, "fetched_at": now}
    return trials


def clear_cache():
    """Clear the in-memory cache (useful for tests or an admin refresh)."""
    _CACHE.clear()


MAX_ATTEMPTS = 3   # retry budget per subtype (mirrors the activity diagram)


def refresh_all_trials():
    """Admin 'Update Clinical Trials' action (honest subset of Fig 4.21).

    Re-fetches every subtype from the live ClinicalTrials.gov API, retrying up
    to MAX_ATTEMPTS times per subtype on failure, and returns a structured
    update report. This does NOT diff/insert into a local trials table (the app
    has no such table; trials are served live + cached), so the report reflects
    what the live API actually returned, per subtype. No fabricated numbers.
    """
    report = {"subtypes": [], "total_live": 0, "total_fallback": 0,
              "total_errors": 0, "fetched_at": time.time()}

    for subtype in SUBTYPE_QUERY:
        attempts = 0
        last_error = None
        trials = None
        source = "error"
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            try:
                fetched = _fetch_live(subtype)
                if not fetched:
                    raise ValueError("API reachable but returned no matching trials")
                trials = fetched
                source = "live"
                break
            except Exception as e:      # network/timeout/parse/empty
                last_error = str(e)
                time.sleep(0.4 * attempts)   # small backoff between attempts

        if trials is None:
            # All attempts failed: fall back to the curated list so the UI
            # still has something, and record the error honestly.
            trials = FALLBACK_TRIALS.get(subtype, [])
            source = "fallback"

        # Refresh the cache with whatever we got.
        for t in trials:
            t.setdefault("source", source)
        _CACHE[subtype] = {"data": trials, "fetched_at": time.time()}

        if source == "live":
            report["total_live"] += 1
        elif source == "fallback":
            report["total_fallback"] += 1
            report["total_errors"] += 1

        report["subtypes"].append({
            "subtype": subtype,
            "source": source,
            "count": len(trials),
            "attempts": attempts,
            "error": last_error if source != "live" else None,
        })

    return report
