"""
LumiTNBC: Flask Application
Production-ready: SQLAlchemy DB, hashed auth, multi-user, hybrid ML pipeline.
"""
import os
import json
import uuid
from datetime import datetime
from functools import wraps

# Load variables from a local .env file if present (so ANTHROPIC_API_KEY, SECRET_KEY,
# etc. work in any terminal, PowerShell included). Harmless in production: Railway
# sets real env vars and there is no .env file there. No-op if python-dotenv
# isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, abort, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy

from config import config
from models_db import db, User, Analysis, ReviewRequest, patient_code
from ml_pipeline import (
    load_assets, get_gene_list, get_classes,
    parse_gene_file, classify_gene_only, classify_hybrid,
    build_clinical_vector,
)
from clinical_intelligence import calibrate_confidence, generate_insights
import llm_summary
import clinical_trials


# ── App factory ───────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app(config_name="default"):
    app = Flask(__name__,
                static_folder="static",
                template_folder="templates")
    app.config.from_object(config[config_name])

    # Ensure instance/ directory exists before SQLAlchemy tries to open the DB
    instance_dir = os.path.join(_APP_DIR, "instance")
    os.makedirs(instance_dir, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        db.create_all()
        _seed_demo_users()
        load_assets()

    _register_routes(app)
    return app


def _seed_demo_users():
    """Ensure the demo accounts exist. Seeds each one individually if missing,
    so the admin/provider are always present even after patients have
    registered (avoids the all-or-nothing 'count==0' gap)."""
    demos = [
        dict(email="sarah@email.com", name="Sarah Johnson", role="patient",
             phone="+60 123456789", diagnosis_date="December 2024",
             cancer_stage="Stage IIA", password="password123"),
        dict(email="dr.brown@hospital.com", name="Dr. M. Brown", role="provider",
             password="password123"),
        dict(email="admin@lumitnbc.com", name="System Admin", role="admin",
             password="admin123"),
    ]
    created = []
    for d in demos:
        if User.query.filter_by(email=d["email"]).first():
            continue
        pw = d.pop("password")
        u = User(**d)
        u.set_password(pw)
        db.session.add(u)
        created.append(d["email"])
    if created:
        db.session.commit()
        print(f"[seed] Created demo accounts: {', '.join(created)}")


# ── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("index"))
            if session.get("role") not in roles:
                abort(403)
            return f(*args, **kwargs)
        return decorated
    return decorator


def current_user():
    if "user_id" in session:
        return db.session.get(User, session["user_id"])
    return None


# ── Static data ───────────────────────────────────────────────────────────────

SUBTYPE_INFO = {
    "BL1": {"full_name": "Basal-Like 1", "color": "#D75DA6",
             "key_pathways": "Cell cycle, DNA damage response, proliferation",
             "clinical_relevance": "Highest pCR to neoadjuvant chemotherapy",
             "description": "BL1 tumours are characterised by high proliferation rates and active DNA damage response pathways. They typically show the best response to platinum-based chemotherapy and PARP inhibitors.",
             # Plain-language content shown to patients
             "patient_summary": "Your tumour belongs to the Basal-Like 1 group. These tumours tend to grow quickly, but that same fast growth often makes them respond very well to chemotherapy.",
             "patient_meaning": "Because BL1 tumours are sensitive to common chemotherapy drugs (especially platinum-based ones), this subtype usually has one of the best responses to treatment given before surgery.",
             "patient_good_news": "BL1 has the highest rate of tumour shrinkage with standard chemotherapy compared with the other TNBC subtypes."},
    "BL2": {"full_name": "Basal-Like 2", "color": "#ECA3BF",
             "key_pathways": "Growth factor signalling, glycolysis, myoepithelial markers",
             "clinical_relevance": "Growth factor pathway targets; lower pCR than BL1",
             "description": "BL2 tumours show enrichment in growth factor signalling and metabolic pathways. They may respond to therapies targeting growth factor receptors and metabolic pathways.",
             "patient_summary": "Your tumour belongs to the Basal-Like 2 group. These tumours rely on growth-signal and energy pathways that researchers are actively learning to target.",
             "patient_meaning": "BL2 tumours use specific 'growth signals' to grow. This opens the door to treatments that aim to block those signals, often alongside standard chemotherapy.",
             "patient_good_news": "Knowing your tumour is BL2 helps your care team consider therapies matched to its growth pathways."},
    "LAR": {"full_name": "Luminal Androgen Receptor", "color": "#4A8EC2",
             "key_pathways": "Androgen receptor, luminal-like, PI3K/AKT",
             "clinical_relevance": "Anti-androgen therapy, CDK4/6 inhibitors",
             "description": "LAR tumours express high levels of androgen receptor and show luminal-like gene expression. They may benefit from anti-androgen therapies (e.g., enzalutamide) and PI3K/AKT pathway inhibitors.",
             "patient_summary": "Your tumour belongs to the Luminal Androgen Receptor group. These tumours respond to a hormone signal (the androgen receptor), which gives extra treatment options beyond chemotherapy.",
             "patient_meaning": "LAR tumours are driven partly by a hormone pathway. This means medicines that block that hormone signal (similar in idea to hormone therapies) may be useful for you.",
             "patient_good_news": "This subtype often grows more slowly than other TNBC types and has additional targeted treatment options."},
    "M":   {"full_name": "Mesenchymal", "color": "#28A745",
             "key_pathways": "EMT, cell motility, Wnt/TGF-β signalling",
             "clinical_relevance": "EMT-targeted therapies, anti-angiogenics",
             "description": "Mesenchymal tumours are enriched for epithelial-to-mesenchymal transition (EMT) and cell motility pathways. They may respond to anti-angiogenic therapies and EMT-targeted approaches.",
             "patient_summary": "Your tumour belongs to the Mesenchymal group. These tumours show patterns linked to cell movement and tissue structure, which researchers are studying for targeted treatments.",
             "patient_meaning": "Mesenchymal tumours use pathways tied to how cells move and reshape tissue. Treatments that target blood-vessel growth and these movement pathways are being explored for this subtype.",
             "patient_good_news": "Identifying the Mesenchymal subtype helps match you with clinical trials studying treatments designed for these specific pathways."},
}


# Curated YouTube videos for the Resources page. Fill this list with vetted
# videos (see RESOURCE_VIDEOS below). Each card shows the real YouTube
# thumbnail and opens the video in a new tab when clicked.
def youtube_thumb(video_id):
    """Public YouTube thumbnail URL for a given video ID (no API needed)."""
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def youtube_watch_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


def favicon_url(page_url, size=128):
    """Brand favicon for a resource link, via Google's favicon service.
    Derives the domain from the link's own URL, so nothing extra to maintain."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(page_url).netloc or page_url
    except Exception:
        domain = page_url
    return f"https://www.google.com/s2/favicons?domain={domain}&sz={size}"


# Add entries here. Each: {"id": "<youtube_video_id>", "title": "...",
# "source": "...", "desc": "..."}. The id is the part after v= in the URL.
RESOURCE_VIDEOS = [
    {"id": "_UqT2846QoU",
     "title": "Triple Negative Breast Cancer (TNBC): Treatment Options and What to Expect",
     "source": "Yerbba – Breast Cancer",
     "desc": "An overview of TNBC treatment options and what patients can expect."},
    {"id": "hgrr690u7H0",
     "title": "Triple Negative Breast Cancer (TNBC) | Dr Rachel Dear",
     "source": "St Vincent's Hospital, Sydney",
     "desc": "Dr Rachel Dear explains triple-negative breast cancer."},
    {"id": "YsCNHXWKRZo",
     "title": "The challenges of treating early-stage TNBC",
     "source": "VJOncology",
     "desc": "A look at the clinical challenges of treating early-stage TNBC."},
    {"id": "TkhMA0EbGgU",
     "title": "Triple-Negative Breast Cancer (TNBC) Breakthroughs: Hope, Innovation, and New Treatments",
     "source": "The Informed Woman Magazine",
     "desc": "Recent breakthroughs and emerging treatments for TNBC."},
    {"id": "Id93EUfA5HM",
     "title": "Thriving Beyond TNBC: Resilience, Advocacy and Community",
     "source": "CancerCare",
     "desc": "Living well beyond a TNBC diagnosis: resilience, advocacy, and community support."},
]


# Curated, real external resources from reputable organisations. These are
# trusted general TNBC/breast-cancer support sources, not medical advice.
RESOURCE_LINKS = [
    {"category": "Understanding TNBC",
     "icon": "book",
     "items": [
        {"title": "Triple-Negative Breast Cancer Overview",
         "org": "American Cancer Society",
         "url": "https://www.cancer.org/cancer/types/breast-cancer/about/types-of-breast-cancer/triple-negative.html",
         "desc": "What triple-negative breast cancer is and how it differs from other types."},
        {"title": "TNBC Information & Support",
         "org": "Breastcancer.org",
         "url": "https://www.breastcancer.org/types/triple-negative",
         "desc": "Plain-language guide covering diagnosis, treatment options and questions to ask."},
        {"title": "Breast Cancer Treatment (PDQ)",
         "org": "National Cancer Institute",
         "url": "https://www.cancer.gov/types/breast/patient/breast-treatment-pdq",
         "desc": "Detailed, regularly-updated treatment information from the NCI."},
        {"title": "TNBC Resource Library",
         "org": "ACCC",
         "url": "https://www.accc-cancer.org/education-and-resources/cancer-types/breast-cancer/multidisciplinary-breast-cancer-resources/tnbc-resource-library",
         "desc": "A multidisciplinary collection of TNBC education and resources."},
        {"title": "Triple-Negative Breast Cancer",
         "org": "CancerCare",
         "url": "https://www.cancercare.org/diagnosis/triple_negative_breast_cancer",
         "desc": "Overview of a TNBC diagnosis and the support services available."},
     ]},
    {"category": "Support & Community",
     "icon": "community",
     "items": [
        {"title": "TNBC Foundation",
         "org": "Triple Negative Breast Cancer Foundation",
         "url": "https://tnbcfoundation.org/",
         "desc": "Education, support resources and research news focused on TNBC."},
        {"title": "Patient Support Services",
         "org": "Susan G. Komen",
         "url": "https://www.komen.org/support-resources/",
         "desc": "Helpline, financial assistance information and local support."},
        {"title": "Living Beyond Breast Cancer",
         "org": "LBBC",
         "url": "https://www.lbbc.org/",
         "desc": "Programmes and community for people affected by breast cancer."},
        {"title": "Support Groups & Organisations",
         "org": "OurTNBC (Australia)",
         "url": "https://ourtnbc.com.au/support-resources/support-groups-organisations/",
         "desc": "A directory of TNBC support groups and organisations."},
        {"title": "TNBC Support Community",
         "org": "TNBCsupport.com",
         "url": "https://www.tnbcsupport.com/support.html",
         "desc": "Peer support and community resources for people with TNBC."},
        {"title": "Treatment of Breast Cancer (Malaysia)",
         "org": "I Can We Will (Malaysia)",
         "url": "https://www.icanwewill.com.my/be-informed/treatment-of-breast-cancer/",
         "desc": "Malaysian breast cancer information and patient support, including treatment guidance."},
     ]},
]

# General, subtype-tailored FAQ content. Educational only, written in plain
# language; encourages discussion with the patient's own care team.
SUBTYPE_FAQS = {
    "BL1": [
        {"q": "What does Basal-Like 1 mean for my treatment?",
         "a": "BL1 tumours tend to grow quickly and often respond well to chemotherapy, especially platinum-based regimens. Your oncologist will decide the best plan for your specific situation."},
        {"q": "Why is chemotherapy often used for this subtype?",
         "a": "BL1 cells divide rapidly and have active DNA-repair pathways, which makes them sensitive to chemotherapy drugs that target dividing cells. This is why response rates are often high."},
    ],
    "BL2": [
        {"q": "What is different about Basal-Like 2?",
         "a": "BL2 tumours show activity in growth-signal and metabolic pathways. Research into therapies that target these pathways is ongoing, so clinical trials can be especially relevant."},
        {"q": "Should I ask about clinical trials?",
         "a": "It is worth discussing with your care team. Because BL2 biology is an active research area, you may be eligible for studies testing newer targeted approaches."},
    ],
    "LAR": [
        {"q": "What does Luminal Androgen Receptor mean?",
         "a": "LAR tumours express the androgen receptor and behave differently from other TNBC subtypes. This opens the possibility of treatments that target hormone-related pathways, which your oncologist can explain."},
        {"q": "Is LAR treated the same as other TNBC?",
         "a": "Not always. Because LAR tumours use androgen-receptor signalling, some trials study anti-androgen therapies for this subtype. Your care team can advise whether that is relevant for you."},
    ],
    "M": [
        {"q": "What is the Mesenchymal subtype?",
         "a": "Mesenchymal tumours show patterns linked to cell movement and tissue structure. Researchers are studying treatments that target these pathways and blood-vessel growth."},
        {"q": "What treatment research applies to me?",
         "a": "Several trials explore therapies aimed at the pathways active in Mesenchymal tumours. Ask your oncologist whether any current studies fit your situation."},
    ],
}

# General FAQs shown to everyone regardless of subtype.
GENERAL_FAQS = [
    {"q": "Does my subtype change my diagnosis?",
     "a": "Subtype is extra detail about the biology of a triple-negative tumour. It helps guide treatment and trial options but does not replace your full diagnosis and staging from your medical team."},
    {"q": "How accurate is this classification?",
     "a": "The model is a decision-support tool, not a diagnosis. It reports a confidence level with every result. Always confirm findings with your oncologist before making any decisions."},
    {"q": "What should I do with these results?",
     "a": "Bring them to your care team. They can interpret the subtype in the context of your full medical picture and discuss whether it affects your treatment or trial eligibility."},
]



# Plain-language labels for SHAP features, shown to patients instead of raw
# gene symbols. (label, sentence); keep the chart label short (the first item).
GENE_PLAIN_LABELS = {
    "CHI3L2":   ("Inflammatory marker activity",  "Your tumour shows elevated activity of an inflammation-related protein, a pattern commonly seen in this subtype."),
    "SCNN1A":   ("Cell surface channel activity", "A protein involved in cell membrane function was elevated, consistent with basal-like tumour behaviour."),
    "NOSTRIN":  ("Blood vessel signalling",       "A gene that helps regulate blood vessel growth was highly active, which helps characterise your subtype."),
    "HOXB5":    ("Development gene activity",      "A gene that controls cell development showed patterns typical of this subtype."),
    "MUC16":    ("Cell surface protein",          "A protein found on the surface of cells showed activity that helped identify your subtype."),
    "ALCAM":    ("Cell adhesion signalling",      "A protein involved in how cells stick together and communicate was active in your sample."),
    "LIN7A":    ("Cell organisation protein",     "A protein involved in organising cell structure matched the pattern of your subtype."),
    "BMP4":     ("Growth signalling pathway",     "A gene involved in controlling cell growth supported this classification."),
    "GBP5":     ("Immune response activity",      "Your sample showed signs of immune system activation, a hallmark of this subtype."),
    "BATF":     ("Immune cell gene activity",     "A gene that controls immune cell behaviour was highly active, suggesting an immune-rich tumour."),
    "TGFBI":    ("Tissue structure protein",      "A protein involved in how tissue is built and repaired showed characteristic activity."),
    "LAG3":     ("Immune checkpoint marker",      "A marker linked to immune regulation was detected, associated with immune-active tumours."),
    "ADAMDEC1": ("Tissue remodelling enzyme",     "An enzyme involved in remodelling tissue showed activity consistent with your subtype."),
    "COL5A1":   ("Connective tissue protein",     "A structural protein in the tissue around your tumour matched this subtype."),
    "FZD7":     ("Cell growth pathway",           "A receptor involved in controlling cell growth and fate supported the classification."),
    "CXCL10":   ("Immune recruitment signal",     "A signalling molecule that attracts immune cells was highly active."),
    "EPHB3":    ("Cell boundary signalling",      "A gene that helps cells define their boundaries was active in your sample."),
    "UCP2":     ("Cellular energy metabolism",    "A gene involved in how your tumour cells generate energy was elevated."),
    "NDRG2":    ("Stress response gene",          "A gene involved in how cells respond to stress matched your tumour profile."),
    "clin_age":         ("Patient age",            "Your age was one factor that helped refine the prediction."),
    "clin_grade":       ("Tumour grade",           "The grade of your tumour (how different the cells look from normal) was an important factor."),
    "clin_stage":       ("Cancer stage",           "The stage of your cancer was taken into account."),
    "clin_tumor_size":  ("Tumour size",            "The size of your tumour contributed to the prediction."),
    "clin_node_status": ("Lymph node involvement", "Whether cancer has spread to nearby lymph nodes was factored in."),
    "clin_menopausal":  ("Menopausal status",      "Your menopausal status was one of the clinical factors considered."),
}


def gene_label(gene):
    """Return (short_label, sentence) plain-language description for a feature."""
    return GENE_PLAIN_LABELS.get(
        gene, ("Molecular marker activity",
               "This marker showed activity that supported the result."))


def build_patient_shap_chart(result):
    """Top-5 SHAP features as plain-language chart data for the patient view."""
    feats = (result or {}).get("shap_features") or []
    chart = []
    for f in feats[:5]:
        label, _ = gene_label(f.get("gene", ""))
        chart.append({
            "label": label,
            "value": round(abs(f.get("shap_value", 0)), 4),
            "supports": f.get("shap_value", 0) > 0,
        })
    return chart


def build_patient_reasons(result):
    """Top-5 SHAP features as plain-language reason cards for the patient view.
    Includes gene-level detail (gene symbol + signed contribution) so this
    section adds information the quick bar chart does not."""
    feats = (result or {}).get("shap_features") or []
    reasons = []
    for f in feats[:5]:
        gene = f.get("gene", "")
        label, sentence = gene_label(gene)
        shap_val = f.get("shap_value", 0)
        reasons.append({
            "label": label,
            "sentence": sentence,
            "gene": gene,
            "value": round(shap_val, 4),
            "magnitude": round(abs(shap_val), 4),
            "supports": shap_val > 0,
        })
    return reasons


# ── Routes registration ───────────────────────────────────────────────────────
def _register_routes(app):

    # ── Page routes ───────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/register")
    def register():
        if "user_id" in session:
            return _role_redirect(session.get("role"))
        return render_template("register.html")

    @app.route("/dashboard")
    @login_required
    def dashboard():
        user = current_user()
        if user.role == "provider":
            return redirect(url_for("provider_dashboard"))
        if user.role == "admin":
            return redirect(url_for("admin_dashboard"))
        analyses = (Analysis.query
                    .filter_by(user_id=user.id)
                    .order_by(Analysis.created_at.desc())
                    .limit(10).all())
        return render_template("dashboard.html", user=user.to_dict(),
                               analyses=[a.to_dict() for a in analyses])

    @app.route("/resources")
    @login_required
    def resources():
        user = current_user()
        # Patients see content tailored to their most recent subtype, if any.
        latest = (Analysis.query
                  .filter_by(user_id=user.id)
                  .order_by(Analysis.created_at.desc())
                  .first())
        subtype = latest.subtype if latest else None
        return render_template("resources.html",
                               user=user.to_dict(),
                               subtype=subtype,
                               subtype_info=SUBTYPE_INFO,
                               resource_links=RESOURCE_LINKS,
                               resource_videos=RESOURCE_VIDEOS,
                               youtube_thumb=youtube_thumb,
                               youtube_watch_url=youtube_watch_url,
                               favicon_url=favicon_url,
                               subtype_faqs=SUBTYPE_FAQS,
                               general_faqs=GENERAL_FAQS)

    @app.route("/classification")
    def classification():
        # Open to guests too; they can run a one-off analysis without an account.
        return render_template("classification.html", gene_list=get_gene_list())

    @app.route("/results")
    @app.route("/results/<analysis_id>")
    def results(analysis_id=None):
        user = current_user()
        result = None

        if user:
            if analysis_id:
                a = Analysis.query.filter_by(analysis_id=analysis_id).first()
                # Providers may only view analyses they ran themselves; admins see all.
                if a and (a.user_id == user.id or user.role == "admin"):
                    result = a.to_dict()
            else:
                # Most recent for this user
                a = (Analysis.query
                     .filter_by(user_id=user.id)
                     .order_by(Analysis.created_at.desc())
                     .first())
                if a:
                    result = a.to_dict()
                    analysis_id = a.analysis_id
            user_dict = user.to_dict()
        else:
            # Guest: result is held in the session only (not persisted).
            result = session.get("guest_result")
            if result:
                analysis_id = result.get("analysis_id")
                result = dict(result)
                result["subtype_info"] = SUBTYPE_INFO.get(result.get("subtype", ""), {})
            user_dict = {"name": "Guest", "role": "patient", "is_guest": True}

        if not result:
            # Nothing to show: guests go to upload, logged-in users likewise.
            return redirect(url_for("classification"))

        return render_template("results.html", result=result,
                               analysis_id=analysis_id,
                               subtype_info=SUBTYPE_INFO,
                               patient_shap_chart=build_patient_shap_chart(result),
                               patient_reasons=build_patient_reasons(result),
                               llm_summary_enabled=llm_summary.is_enabled(),
                               user=user_dict)

    @app.route("/profile")
    @login_required
    def profile():
        user = current_user()
        analyses = (Analysis.query
                    .filter_by(user_id=user.id)
                    .order_by(Analysis.created_at.desc())
                    .all())
        # Providers no longer run analyses; show their review activity instead.
        reviews_done = 0
        if user.role in ("provider", "admin"):
            reviews_done = ReviewRequest.query.filter_by(
                reviewer_id=user.id, status="reviewed").count()
        return render_template("profile.html", user=user.to_dict(),
                               analyses=[a.to_dict() for a in analyses],
                               reviews_done=reviews_done)

    @app.route("/provider-dashboard")
    @role_required("provider", "admin")
    def provider_dashboard():
        user = current_user()
        # Shared queue: every provider sees all review requests. Patient names
        # are never exposed here, only a privacy-safe patient code.
        pending = (ReviewRequest.query
                   .filter_by(status="pending")
                   .order_by(ReviewRequest.created_at.asc())
                   .all())
        reviewed = (ReviewRequest.query
                    .filter_by(status="reviewed")
                    .order_by(ReviewRequest.reviewed_at.desc())
                    .limit(50).all())
        return render_template("provider-dashboard.html",
                               user=user.to_dict(),
                               pending=[r.to_dict() for r in pending],
                               reviewed=[r.to_dict() for r in reviewed],
                               stats=_population_stats())

    @app.route("/review/<int:request_id>")
    @role_required("provider", "admin")
    def review_detail(request_id):
        user = current_user()
        req = ReviewRequest.query.get_or_404(request_id)
        result = req.analysis.to_dict() if req.analysis else None
        if result:
            result["subtype_info"] = SUBTYPE_INFO.get(result.get("subtype", ""), {})
        return render_template("review-detail.html",
                               user=user.to_dict(),
                               review=req.to_dict(),
                               result=result,
                               subtype_info=SUBTYPE_INFO,
                               patient_shap_chart=build_patient_shap_chart(result or {}))

    # ── Review API ────────────────────────────────────────────────────────────
    @app.route("/api/review/request", methods=["POST"])
    @login_required
    def api_review_request():
        """Patient opts in to have one of their analyses reviewed."""
        user = current_user()
        data = request.get_json(silent=True) or {}
        analysis_id = data.get("analysis_id")
        a = Analysis.query.filter_by(analysis_id=analysis_id).first()
        if not a:
            return jsonify({"success": False, "message": "Analysis not found"}), 404
        # Only the owner of the analysis can request a review of it.
        if a.user_id != user.id:
            abort(403)
        existing = ReviewRequest.query.filter_by(analysis_id=analysis_id).first()
        if existing:
            return jsonify({"success": True, "status": existing.status,
                            "message": "A review has already been requested."})
        req = ReviewRequest(analysis_id=analysis_id, patient_id=user.id,
                            status="pending")
        db.session.add(req)
        db.session.commit()
        return jsonify({"success": True, "status": "pending"})

    @app.route("/api/review/status/<analysis_id>")
    @login_required
    def api_review_status(analysis_id):
        """Patient checks the review status of one of their analyses."""
        user = current_user()
        req = ReviewRequest.query.filter_by(analysis_id=analysis_id).first()
        if not req or req.patient_id != user.id:
            return jsonify({"requested": False})
        return jsonify({"requested": True, **req.to_dict(include_name=True)})

    @app.route("/api/review/<int:request_id>/submit", methods=["POST"])
    @role_required("provider", "admin")
    def api_review_submit(request_id):
        """Provider submits free-text feedback, completing the review."""
        user = current_user()
        req = ReviewRequest.query.get_or_404(request_id)
        data = request.get_json(silent=True) or {}
        text = (data.get("review_text") or "").strip()
        if len(text) < 10:
            return jsonify({"success": False,
                            "message": "Please write at least a sentence of feedback."}), 400
        req.review_text = text
        req.reviewer_id = user.id
        req.status = "reviewed"
        req.reviewed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({"success": True})

    @app.route("/admin-dashboard")
    @role_required("admin")
    def admin_dashboard():
        user = current_user()
        users = User.query.order_by(User.created_at.desc()).all()
        analyses = Analysis.query.order_by(Analysis.created_at.desc()).all()
        stats = {
            "total_users": len(users),
            "total_analyses": len(analyses),
            "subtypes": {s: Analysis.query.filter_by(subtype=s).count()
                         for s in ["BL1", "BL2", "LAR", "M"]},
            "methods": {
                "gene_only": Analysis.query.filter_by(input_mode="gene_only").count(),
                "clinical_only": Analysis.query.filter_by(input_mode="clinical_only").count(),
                "hybrid": Analysis.query.filter_by(input_mode="hybrid").count(),
            },
        }
        return render_template("admin-dashboard.html",
                               user=user.to_dict(),
                               users=[u.to_dict() for u in users],
                               analyses=[a.to_dict() for a in analyses],
                               stats=stats)

    # ── API: Auth ─────────────────────────────────────────────────────────────

    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({"success": False, "message": "Invalid email or password"}), 401

        session.permanent = True
        session["user_id"] = user.id
        session["role"] = user.role
        session["name"] = user.name
        session["email"] = user.email

        redirect_map = {
            "patient": "/dashboard",
            "provider": "/provider-dashboard",
            "admin": "/admin-dashboard",
        }
        return jsonify({"success": True, "redirect": redirect_map.get(user.role, "/dashboard")})

    @app.route("/api/logout", methods=["POST"])
    def api_logout():
        session.clear()
        return jsonify({"success": True})

    @app.route("/api/register", methods=["POST"])
    def api_register():
        data = request.get_json()
        email = (data.get("email") or "").strip().lower()
        name = (data.get("name") or "").strip()
        password = data.get("password") or ""

        if not email or not name or not password:
            return jsonify({"success": False, "message": "All fields are required"}), 400
        if len(password) < 8:
            return jsonify({"success": False, "message": "Password must be at least 8 characters"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"success": False, "message": "Email already registered"}), 400

        user = User(email=email, name=name, role="patient")
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        session.permanent = True
        session["user_id"] = user.id
        session["role"] = user.role
        session["name"] = user.name
        session["email"] = user.email
        return jsonify({"success": True, "redirect": "/dashboard"})

    @app.route("/api/profile", methods=["PATCH"])
    @login_required
    def api_profile_update():
        user = current_user()
        data = request.get_json()
        allowed = ["name", "phone", "diagnosis_date", "cancer_type", "cancer_stage"]
        for field in allowed:
            if field in data:
                setattr(user, field, data[field])
        if "new_password" in data and data["new_password"]:
            if not user.check_password(data.get("current_password", "")):
                return jsonify({"success": False, "message": "Current password incorrect"}), 400
            user.set_password(data["new_password"])
        db.session.commit()
        session["name"] = user.name
        return jsonify({"success": True})

    # ── API: Classification ───────────────────────────────────────────────────

    @app.route("/api/classify/upload", methods=["POST"])
    def api_classify_upload():
        """Gene expression file upload → gene_only or hybrid classification.
        Works for logged-in users (saved to DB) and guests (session only)."""
        user = current_user()

        if "file" not in request.files:
            return jsonify({"error": True, "message": "No file uploaded"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": True, "message": "No file selected"}), 400

        file_bytes = f.read()

        try:
            gene_df = parse_gene_file(file_bytes, f.filename)
        except Exception as e:
            return jsonify({"error": True, "message": f"Could not parse file: {e}"}), 400

        # Check if clinical data also submitted (hybrid mode)
        clinical_json = request.form.get("clinical_data")
        clinical_data = None
        if clinical_json:
            try:
                clinical_data = json.loads(clinical_json)
            except json.JSONDecodeError:
                pass

        if clinical_data:
            result = classify_hybrid(gene_df, clinical_data)
        else:
            result = classify_gene_only(gene_df)

        if result.get("error"):
            return jsonify(result), 400

        result = _enrich_result(result, user, filename=f.filename)

        # Clinical intelligence when clinical data available
        if clinical_data:
            result["clinical_data"] = clinical_data
            calib = calibrate_confidence(
                result["subtype"], result["confidence"],
                result.get("probabilities", {}), clinical_data
            )
            result["calibration"] = calib
            result["confidence"] = calib["adjusted_confidence"]
            result["insights"] = generate_insights(
                result["subtype"], result["confidence"], clinical_data
            )

        if user:
            _save_analysis(result, user)
        else:
            guest = {k: v for k, v in result.items() if k != "subtype_info"}
            session["guest_result"] = guest
        return jsonify(result)

    @app.route("/api/classify/clinical", methods=["POST"])
    def api_classify_clinical():
        """Clinical-only form → rule-based classification.
        Works for logged-in users (saved to DB) and guests (session only)."""
        from app_utils import clinical_rule_based_classification
        user = current_user()
        clinical_data = request.get_json()

        result = clinical_rule_based_classification(clinical_data)
        result = _enrich_result(result, user)
        result["input_mode"] = "clinical_only"
        result["clinical_data"] = clinical_data

        # Clinical intelligence
        calib = calibrate_confidence(
            result["subtype"], result["confidence"],
            result.get("scores", {}), clinical_data
        )
        result["calibration"] = calib
        result["confidence"] = calib["adjusted_confidence"]
        result["insights"] = generate_insights(
            result["subtype"], result["confidence"], clinical_data
        )

        if user:
            _save_analysis(result, user)
        else:
            guest = {k: v for k, v in result.items() if k != "subtype_info"}
            session["guest_result"] = guest
        return jsonify(result)

    # ── API: Data retrieval ───────────────────────────────────────────────────

    @app.route("/api/analyses")
    @login_required
    def api_analyses():
        user = current_user()
        # Admins see all analyses; everyone else (patients and providers)
        # sees only their own.
        if user.role == "admin":
            analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(100).all()
        else:
            analyses = (Analysis.query
                        .filter_by(user_id=user.id)
                        .order_by(Analysis.created_at.desc())
                        .all())
        return jsonify([a.to_dict() for a in analyses])

    @app.route("/api/analyses/<analysis_id>")
    @login_required
    def api_analysis_detail(analysis_id):
        user = current_user()
        a = Analysis.query.filter_by(analysis_id=analysis_id).first_or_404()
        # Providers may only access analyses they ran themselves; admins see all.
        if a.user_id != user.id and user.role != "admin":
            abort(403)
        return jsonify(a.to_dict())

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        """Chatbot endpoint. Tries the LLM+RAG path; if no API key (or any
        failure) it falls back to the offline keyword responder. Scoped to
        general TNBC / app help only, no patient-specific advice."""
        import chatbot_llm
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not message:
            return jsonify({"reply": "Please type a question.", "source": "system"})

        result = chatbot_llm.generate_reply(message, history)
        if result.get("enabled") and result.get("reply"):
            return jsonify({"reply": result["reply"], "source": "llm"})

        # Log WHY we fell back so it's visible in the deploy logs. This prints
        # the reason (no key / package missing / Anthropic error) but never the key.
        if not chatbot_llm.is_enabled():
            app.logger.warning("[chat] LLM disabled: ANTHROPIC_API_KEY not set in this environment.")
        elif result.get("error"):
            app.logger.warning("[chat] LLM call failed, using keyword fallback. Reason: %s",
                               result["error"])
        else:
            app.logger.warning("[chat] LLM returned no reply; using keyword fallback.")

        # Fallback: offline keyword responder (no API key or LLM unavailable).
        return jsonify({"reply": _keyword_fallback(message), "source": "keyword"})

    @app.route("/api/summary/<analysis_id>")
    @login_required
    def api_summary(analysis_id):
        """
        LLM layer. Returns a personalised plain-language summary
        when ANTHROPIC_API_KEY is set; otherwise {"enabled": false} so the UI
        hides the card. Same access control as the results page.
        """
        user = current_user()
        a = Analysis.query.filter_by(analysis_id=analysis_id).first_or_404()
        if a.user_id != user.id and user.role != "admin":
            abort(403)

        if not llm_summary.is_enabled():
            return jsonify({"enabled": False, "summary": None})

        result = a.to_dict()
        result["subtype_full_name"] = SUBTYPE_INFO.get(
            result.get("subtype"), {}).get("full_name")
        out = llm_summary.generate_summary(result, gene_label)
        return jsonify(out)

    @app.route("/api/gene-list")
    def api_gene_list():
        genes = get_gene_list()
        return jsonify({"genes": genes, "count": len(genes)})

    # Demo sample files users can download to try the app.
    _SAMPLE_FILES = {
        "bl1": "tcga_single_bl1.csv",
        "bl2": "tcga_single_bl2.csv",
        "lar": "tcga_single_lar.csv",
        "m":   "tcga_single_m.csv",
        "all": "tcga_all_subtypes_8samples.csv",
    }

    @app.route("/sample-data/<key>")
    def sample_data(key):
        """Serve a demo gene-expression CSV for users to try the app."""
        filename = _SAMPLE_FILES.get(key)
        if not filename:
            abort(404)
        test_dir = os.path.join(_APP_DIR, "test_data")
        return send_from_directory(test_dir, filename, as_attachment=True)

    @app.route("/api/stats")
    @role_required("admin")
    def api_stats():
        return jsonify({
            "total_users": User.query.count(),
            "total_analyses": Analysis.query.count(),
            "subtypes": {s: Analysis.query.filter_by(subtype=s).count()
                         for s in get_classes()},
        })

    @app.route("/api/admin/refresh-trials", methods=["POST"])
    @role_required("admin")
    def api_refresh_trials():
        """Clear the clinical-trials cache so the next lookup re-queries the
        live ClinicalTrials.gov API. Supports the admin 'Update Clinical Trials
        Database' use case."""
        clinical_trials.clear_cache()
        return jsonify({"success": True, "message": "Clinical-trials cache cleared."})

    @app.route("/api/admin/clear-data", methods=["POST"])
    @role_required("admin")
    def api_clear_data():
        """Wipe all analyses and review requests for a clean demo, keeping
        user accounts intact."""
        n_reviews = ReviewRequest.query.delete()
        n_analyses = Analysis.query.delete()
        db.session.commit()
        return jsonify({
            "success": True,
            "message": f"Cleared {n_analyses} analyses and {n_reviews} review requests.",
        })

    @app.errorhandler(403)
    def forbidden(e):
        return render_template("index.html"), 403

    @app.errorhandler(404)
    def not_found(e):
        return render_template("index.html"), 404


# ── Shared helpers ────────────────────────────────────────────────────────────

def _generate_analysis_id():
    return f"LT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


def _keyword_fallback(message: str) -> str:
    """Offline keyword responder used when the LLM chatbot is unavailable.
    Scoped to general TNBC / app help; declines medical advice."""
    m = (message or "").lower()

    def has(*words):
        return any(w in m for w in words)

    # Decline medical-advice style questions even in fallback mode.
    if has("should i", "what should i do", "will i survive", "how long",
           "prognosis", "cure", "is it curable", "my treatment", "best treatment"):
        return ("I can't give medical advice or discuss your personal treatment "
                "or prognosis. Please talk this through with your oncologist or "
                "care team, they know your full situation.")

    if has("what is tnbc", "triple negative", "triple-negative"):
        return ("Triple-negative breast cancer (TNBC) lacks oestrogen receptors, "
                "progesterone receptors, and excess HER2, so it is treated "
                "differently from other breast cancers. Molecular subtyping helps "
                "understand its biology.")
    if has("subtype", "bl1", "bl2", "lar", "mesenchymal"):
        return ("LumiTNBC classifies TNBC into four molecular subtypes: BL1, BL2, "
                "LAR, and M. Each reflects different tumour biology and is an area "
                "of active research.")
    if has("how", "start", "upload", "use", "begin"):
        return ("Go to 'New Analysis', upload a gene-expression CSV (or try a "
                "sample file), and click Start Analysis. Results appear in about a "
                "minute with the subtype, confidence, factors, and matched trials.")
    if has("shap", "explain", "why", "factor"):
        return ("The model uses SHAP to show which gene signals most influenced "
                "the result, each with a direction and strength, so the decision "
                "is transparent rather than a black box.")
    if has("confidence", "accurate", "sure", "reliable"):
        return ("Each result shows a confidence level: how strongly the model "
                "favours the predicted subtype over the others. It reflects the "
                "model's certainty, not a guarantee, always confirm with your "
                "care team.")
    if has("trial", "clinical", "study"):
        return ("After classification, the app lists clinical trials that may fit "
                "the subtype, drawn from ClinicalTrials.gov, with location, status "
                "and a link to each official trial page.")
    if has("privacy", "secure", "safe", "data"):
        return ("Uploaded data is used only to produce your result. Guest "
                "analyses are not saved, and providers only ever see a "
                "privacy-safe patient code, never your name.")
    if has("hi", "hello", "hey", "help"):
        return ("Hi! I'm Lumi. I can answer general questions about TNBC, the four "
                "subtypes, how this app works, and how to read your results. What "
                "would you like to know?")
    return ("I can help with general questions about TNBC, the molecular "
            "subtypes, how LumiTNBC works, and how to read your results. I can't "
            "give medical advice, for anything about your personal care, please "
            "speak with your oncologist.")


def _population_stats() -> dict:
    """Aggregate, de-identified statistics across all analyses in the system.
    Used on the provider dashboard. Contains no patient-identifying data."""
    from sqlalchemy import func
    total = Analysis.query.count()
    order = ["BL1", "BL2", "LAR", "M"]
    names = {"BL1": "Basal-Like 1", "BL2": "Basal-Like 2",
             "LAR": "Luminal AR", "M": "Mesenchymal"}

    # Count per subtype
    rows = (db.session.query(Analysis.subtype, func.count(Analysis.id))
            .group_by(Analysis.subtype).all())
    counts = {s: 0 for s in order}
    for subtype, n in rows:
        if subtype in counts:
            counts[subtype] = n

    dist = []
    for s in order:
        pct = round(counts[s] / total * 100, 1) if total else 0.0
        dist.append({"subtype": s, "name": names[s], "count": counts[s], "pct": pct})

    # Most common subtype
    most_common = max(dist, key=lambda d: d["count"]) if total else None

    # Average confidence (overall)
    avg_conf = db.session.query(func.avg(Analysis.confidence)).scalar()
    avg_conf = round(avg_conf, 1) if avg_conf is not None else None

    reviews_total = ReviewRequest.query.count()
    reviews_done = ReviewRequest.query.filter_by(status="reviewed").count()

    return {
        "total": total,
        "distribution": dist,
        "most_common": most_common,
        "avg_confidence": avg_conf,
        "reviews_total": reviews_total,
        "reviews_done": reviews_done,
    }


def _enrich_result(result: dict, user: User, filename: str = None) -> dict:
    result["analysis_id"] = _generate_analysis_id()
    result["timestamp"] = datetime.utcnow().isoformat()
    result["filename"] = filename
    result["trials"] = clinical_trials.get_trials(result.get("subtype", ""))
    result["subtype_info"] = SUBTYPE_INFO.get(result.get("subtype", ""), {})
    # Guests (no account) are treated as the patient view; results are not saved.
    result["user_name"] = user.name if user else "Guest"
    result["user_role"] = user.role if user else "patient"
    return result


def _save_analysis(result: dict, user: User):
    # Strip large Jinja-unfriendly objects before JSON serialisation
    storable = {k: v for k, v in result.items()
                if k not in ("subtype_info",)}
    a = Analysis(
        analysis_id=result["analysis_id"],
        user_id=user.id,
        subtype=result.get("subtype"),
        confidence=result.get("confidence"),
        method=result.get("method"),
        input_mode=result.get("input_mode", "unknown"),
        filename=result.get("filename"),
        result_json=json.dumps(storable, default=str),
    )
    db.session.add(a)
    db.session.commit()


def _role_redirect(role):
    return redirect({
        "patient": url_for("dashboard"),
        "provider": url_for("provider_dashboard"),
        "admin": url_for("admin_dashboard"),
    }.get(role, url_for("dashboard")))


# ── Entry point ───────────────────────────────────────────────────────────────
# On a real deployment (Railway/Heroku) DATABASE_URL is set, so default to the
# production config there; default to development when running locally.
_env = os.environ.get("FLASK_ENV")
if not _env:
    _env = "production" if os.environ.get("DATABASE_URL") else "development"
app = create_app(_env if _env in config else "default")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LumiTNBC: TNBC Subtype Classification")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=(_env == "development"), host="0.0.0.0", port=port)
