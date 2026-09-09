"""WSGI entry point for production servers (gunicorn in the Docker image).

Local development keeps `python app.py` (Flask dev server on localhost).
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
