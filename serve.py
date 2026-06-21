"""serve.py — production WSGI entrypoint via waitress.

Run (production):  "/mnt/c/Program Files/Python311/python.exe" serve.py
Run (dev):         python app.py   (Flask dev server, auto-opens browser)
"""
import config
from waitress import serve
from app import app

if __name__ == "__main__":
    host = getattr(config, "WEB_HOST", "127.0.0.1")
    port = getattr(config, "WEB_PORT", 5000)
    print(f"[serve] waitress serving on http://{host}:{port}")
    serve(app, host=host, port=port, threads=8)
