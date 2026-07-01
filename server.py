"""API HTTP de l'orchestrateur (FastAPI).

Endpoints :

    GET  /                  → healthcheck + liens utiles
    GET  /models            → liste des modèles spécialisés
    POST /orchestrate       → route une requête et renvoie {model, reduced_prompt}
                             (supporte le choix du routeur via ``router_mode``)

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

# Instance partagée heuristique (sans état mutable → sûre en concurrence).
orchestrator_heuristic = Orchestrator(router_mode="heuristic")

# Instance neuronale (créée à la demande, mise en cache).
_orchestrator_neural: Orchestrator | None = None


def _get_neural_orchestrator() -> Orchestrator | None:
    """Retourne l'orchestrateur neuronal (lazy init), ou None si indisponible."""
    global _orchestrator_neural
    if _orchestrator_neural is None:
        try:
            _orchestrator_neural = Orchestrator(router_mode="neural")
        except (ImportError, FileNotFoundError):
            return None  # checkpoints absents → neural non disponible
    return _orchestrator_neural


app = FastAPI(
    title="Orchestrateur de modèles de langage",
    description=(
        "Route une requête en langage naturel vers le modèle spécialisé le plus "
        "approprié (code, creative, factual, math, translation, general) et "
        "renvoie un prompt reformulé et optimisé pour ce modèle cible. "
        "Supporte deux modes de routage : heuristique (défaut) et neuronal (MLP 12 couches)."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Schémas (Pydantic)
# ---------------------------------------------------------------------------
class OrchestrateRequest(BaseModel):
    """Corps de la requête POST /orchestrate."""

    query: str = Field(..., min_length=1, description="Requête en langage naturel.")
    router_mode: str = Field(
        "heuristic",
        description="Routeur à utiliser : 'heuristic' (défaut) ou 'neural'.",
    )


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
    """Route la requête et renvoie le modèle choisi + le prompt réduit.

    Le champ ``router_mode`` permet de choisir entre le routeur heuristique
    (défaut) et le routeur neuronal (nécessite un entraînement préalable).
    Si le mode neural est demandé mais que les checkpoints sont absents,
    le routeur heuristique est utilisé en fallback.
    """
    if request.router_mode == "neural":
        orch = _get_neural_orchestrator()
        if orch is None:
            # Fallback silencieux : le mode neural n'est pas disponible.
            orch = orchestrator_heuristic
    else:
        orch = orchestrator_heuristic

    result = orch.route(request.query)
    return OrchestrateResponse(
        model=result.model,
        reduced_prompt=result.reduced_prompt,
    )
