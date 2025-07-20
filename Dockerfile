FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copier le projet
COPY . /app

# Installer dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Configurer le cache pour Hugging Face
ENV TRANSFORMERS_CACHE=/app/hf_cache
RUN mkdir -p /app/hf_cache

# Exposer un port par défaut (optionnel, utile pour tests locaux)
EXPOSE 8000

# Lancer l'application avec le port dynamique fourni par Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
