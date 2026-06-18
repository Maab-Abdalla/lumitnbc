# Deploying LumiTNBC to Railway

This app is ready to deploy to [Railway](https://railway.app) using the included
`Dockerfile` and `railway.json`.

## Steps

1. **Push the code to GitHub** (Railway deploys from a repo).

2. **Create a new Railway project** → "Deploy from GitHub repo" → pick this repo.
   Railway detects the `Dockerfile` and `railway.json` automatically.

3. **Add a Postgres database**: in the project, click **New → Database → PostgreSQL**.
   Railway automatically exposes a `DATABASE_URL` variable to the app service.
   (The app rewrites the legacy `postgres://` scheme to `postgresql://` for you.)

4. **Set environment variables** on the app service (Variables tab):

   | Variable          | Value                                  | Required |
   |-------------------|----------------------------------------|----------|
   | `SECRET_KEY`      | a long random string                   | yes      |
   | `FLASK_ENV`       | `production`                           | optional* |
   | `ANTHROPIC_API_KEY` | your key, to enable the LLM summary  | optional |

   *If unset, the app defaults to `production` automatically whenever
   `DATABASE_URL` is present (i.e. on Railway).

5. **Deploy.** Railway builds the image and starts it with:
   `gunicorn -c gunicorn.conf.py app:app` (the port is read from `$PORT` in `gunicorn.conf.py`)

6. **Open the generated URL.** On first boot the app creates the tables and
   seeds the three demo accounts.

## Demo accounts (seeded automatically)

| Role     | Email                   | Password      |
|----------|-------------------------|---------------|
| Patient  | sarah@email.com         | password123   |
| Provider | dr.brown@hospital.com   | password123   |
| Admin    | admin@lumitnbc.com      | admin123      |

## Trying the app

On the **New Analysis** page there are **Download sample file** buttons.
Download one and upload it to run a classification end to end, no real data needed.

## Resetting between demos

Log in as the admin account → **Dashboard → Demo Tools → "Clear all analyses &
reviews"**. This wipes all analyses and review requests but keeps the user
accounts, giving you a clean slate without redeploying.

## Notes

- **Database persistence**: with the Postgres add-on, data persists across
  redeploys. (Without it, the app falls back to SQLite, which resets on each
  redeploy since Railway's container filesystem is ephemeral.)
- **Clinical trials**: live results come from the public ClinicalTrials.gov API.
  If Railway's network blocks it, the app falls back to curated example trials.
- The LLM "Your Personal Summary" card only appears when `ANTHROPIC_API_KEY`
  is set; otherwise it is hidden with no errors.
