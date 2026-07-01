"""API HTTP de l'orchestrateur (FastAPI).

Endpoints :

    GET  /                  → healthcheck + liens utiles
    GET  /models            → liste des modèles spécialisés
    POST /orchestrate       → route une requête et renvoie {model, reduced_prompt}

Lancement :

    uvicorn server:app --reload        # développement (hot reload)
    uvicorn server:app --host 0.0.0.0 --port 8000   # production locale

Documentation interactive auto-générée disponibles sur /docs (Swagger UI)
et /redoc une fois le serveur lancé.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from orchestrator import MODEL_NAMES, Orchestrator

# Instance partagée (sans état mutable → sûre en concurrence).
orchestrator = Orchestrator()

app = FastAPI(
    title="Orchestrateur de modèles de langage",
    description=(
        "Route une requête en langage naturel vers le modèle spécialisé le plus "
        "approprié (code, creative, factual, math, translation, general) et "
        "renvoie un prompt reformulé et optimisé pour ce modèle cible. "
        "Routage déterministe 100% local, sans appel à un LLM."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Schémas (Pydantic)
# ---------------------------------------------------------------------------
class OrchestrateRequest(BaseModel):
    """Corps de la requête POST /orchestrate."""

    query: str = Field(..., min_length=1, description="Requête en langage naturel.")


class OrchestrateResponse(BaseModel):
    """Réponse de POST /orchestrate (conforme à la spec JSON)."""

    model: str = Field(..., description="Identifiant du modèle cible choisi.")
    reduced_prompt: str = Field(..., description="Prompt reformulé, prêt pour le modèle cible.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def healthcheck() -> dict:
    """Healthcheck : confirme que le service tourne."""
    return {
        "status": "ok",
        "service": "orchestrator",
        "version": "0.1.0",
        "endpoints": {
            "orchestrate": "POST /orchestrate",
            "models": "GET /models",
            "docs": "GET /docs",
        },
    }


@app.get("/models")
def list_models() -> dict:
    """Liste les identifiants des modèles spécialisés disponibles."""
    return {"models": list(MODEL_NAMES)}


@app.post("/orchestrate", response_model=OrchestrateResponse)
def orchestrate(request: OrchestrateRequest) -> OrchestrateResponse:
    """Route la requête et renvoie le modèle choisi + le prompt réduit."""
    result = orchestrator.route(request.query)
    return OrchestrateResponse(
        model=result.model,
        reduced_prompt=result.reduced_prompt,
    )
