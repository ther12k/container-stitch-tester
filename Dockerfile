FROM python:3.12-slim

# Hosted-demo image for the container stitch tester. Synthetic demo samples
# are GENERATED at build time (demo_seed.py) — no photographs ship in git
# or in the image layers beyond these generated ones.
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CST_PUBLIC=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate the deterministic demo samples + their configs for this build.
RUN python demo_seed.py --force \
    && mkdir -p web_jobs web_uploads

EXPOSE 8000
CMD ["gunicorn", "--workers", "2", "--threads", "8", "--timeout", "300", \
     "--bind", "0.0.0.0:8000", "wsgi:app"]
