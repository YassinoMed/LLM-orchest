"""Modèles de données et catalogue des modèles spécialisés.

Ce module centralise les types partagés par l'orchestrateur :
- :data:`MODEL_NAMES` : liste des identifiants de modèles spécialisés disponibles.
- :class:`RoutingResult` : résultat d'un routage (modèle choisi + prompt réduit).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# Catalogue des modèles spécialisés.
# L'ordre est volontairement identique à la spec du projet ; "general" reste
# l'ultime recours (ambiguïté ou demande de conversation générale).
MODEL_NAMES = ("code", "creative", "factual", "math", "translation", "general")


@dataclass(frozen=True)
class RoutingResult:
    """Résultat produit par :class:`orchestrator.orchestrator.Orchestrator`.

    Attributes:
        model: identifiant du modèle cible choisi (l'un des :data:`MODEL_NAMES`).
        reduced_prompt: prompt reformulé, épuré et prêt pour le modèle cible.
    """

    model: str
    reduced_prompt: str

    def to_dict(self) -> dict:
        """Retourne le résultat sous forme de dictionnaire ``{"model", "reduced_prompt"}``."""
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        """Sérialise le résultat en JSON strict.

        Args:
            indent: indentation optionnelle (``None`` pour un JSON compact, conforme
                au format attendu par la spec de l'orchestrateur).
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
