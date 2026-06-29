"""
LumiTNBC: Configuration
"""
import os
from datetime import timedelta

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_INSTANCE_DIR = os.path.join(_BASE_DIR, "instance")


def _database_uri():
    """Use Railway/Heroku-style DATABASE_URL when present (Postgres),
    otherwise fall back to a local SQLite file. SQLAlchemy needs the
    'postgresql://' scheme, but Railway sometimes supplies 'postgres://'."""
    url = os.environ.get("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    return "sqlite:///" + os.path.join(_INSTANCE_DIR, "lumitnbc.db")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
    MODEL_DIR = os.path.join(_BASE_DIR, "models")
    SQLALCHEMY_DATABASE_URI = _database_uri()


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
