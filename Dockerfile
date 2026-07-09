FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
# Gunakan default-timeout tinggi untuk mengatasi koneksi lambat
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements-api.txt

COPY . .

# Buat folder models jika belum ada
RUN mkdir -p models

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
