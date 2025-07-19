# Étape 1 : Utiliser une image légère de Python 3.10
FROM python:3.10-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Étape 4 : Copier les fichiers de ton projet
COPY . /app

# Étape 5 : Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Étape 6 : Créer un répertoire pour le cache Hugging Face (obligatoire pour Render)
ENV TRANSFORMERS_CACHE=/app/hf_cache
RUN mkdir -p /app/hf_cache

# Étape 7 : Exposer le port pour Uvicorn
EXPOSE 8000

# Étape 8 : Commande de lancement
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
