"""Gunicorn configuration for LumiTNBC.

Reads the port from the environment in Python, so it works regardless of
whether the start command is run through a shell (avoids the literal
'$PORT is not a valid port number' error when the platform does not expand
shell variables)."""
import os

# Railway/Heroku inject PORT; default to 5000 locally.
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")
workers = 2
timeout = 120
