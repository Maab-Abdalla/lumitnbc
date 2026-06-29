# LumiTNBC — Production

TNBC molecular subtype classification with XGBoost + SHAP explainability.

## Quick start (local dev — Windows, Mac, Linux)

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

> **Note:** Dev mode always uses SQLite (`instance/lumitnbc.db`), regardless of any
> `DATABASE_URL` environment variable. No PostgreSQL needed locally.

Demo accounts (auto-created on first run):

| Email | Password | Role |
|---|---|---|
| sarah@email.com | password123 | patient |
| dr.brown@hospital.com | password123 | provider |
| admin@lumitnbc.com | admin123 | admin |

## Docker (production)

```bash
cp .env.example .env
# Edit .env with real SECRET_KEY and DB_PASSWORD
pip install -r requirements-prod.txt   # adds gunicorn + psycopg2
docker-compose up -d
# → http://localhost
```

## Input modes

| Mode | How | Model |
|---|---|---|
| Gene-only | Upload CSV/TSV | Hybrid XGBoost (150 genes; clinical filled with population medians) |
| Hybrid | Upload CSV/TSV + fill clinical form | Hybrid XGBoost (163 features: 150 genes + 13 clinical) |
| Clinical-only | Fill clinical form only | Rule-based scoring (FUSCC surrogates, capped at 82% confidence) |

## Model files (required in `models/`)

- `hybrid_model.joblib` — XGBoost classifier (163 features)
- `label_encoder.joblib` — BL1, BL2, LAR, M
- `hybrid_feature_list.csv` — 163 ordered feature names
- `clinical_features.csv` — 13 clin_* feature names

## Architecture

```
app.py                    Flask app factory, routes, auth decorators
config.py                 Dev (SQLite) / Prod (PostgreSQL) config
models_db.py              SQLAlchemy ORM (User, Analysis)
ml_pipeline.py            ML — parse, feature matrix, predict, SHAP
app_utils.py              Rule-based clinical classification
clinical_intelligence.py  Confidence calibration + personalised insights
```

## Troubleshooting

**`OperationalError: could not connect to server`** — You have `DATABASE_URL` set as a system environment variable pointing to PostgreSQL. Dev mode now ignores it and always uses SQLite, so this should not appear after the latest update. If it does, unset the variable: `set DATABASE_URL=` (Windows) or `unset DATABASE_URL` (Mac/Linux).

**`ModuleNotFoundError: psycopg2`** — Only needed for production PostgreSQL. Don't install `requirements-prod.txt` for local dev.
