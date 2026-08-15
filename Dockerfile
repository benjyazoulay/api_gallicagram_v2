FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances nécessaires (avec fastmcp dernière version et httpx)
RUN pip install --no-cache-dir --upgrade fastapi uvicorn pandas numpy sqlalchemy nltk matplotlib "fastmcp>=3.0.0" httpx

# Copie du code de l'API V2 et du serveur MCP
COPY app.py .
COPY mcp_server.py .

# Exposition des ports de l'API (8002) et du serveur MCP (8003)
EXPOSE 8002
EXPOSE 8003

# Par défaut, le conteneur démarre l'API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]