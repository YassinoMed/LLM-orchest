"""Moteur de classification heuristique.

Transforme une requête en langage naturel en identifiant la catégorie de modèle
la plus probable, à partir des signaux pondérés définis dans :mod:`orchestrator.signals`.

Algorithme (déterministe, 100% local) :
    1. Normalisation de la requête (minuscules, espaces normalisés).
    2. Score = somme des poids des mots-clés trouvés + des regex qui matchent,
       pour chaque catégorie.
    3. La catégorie au score maximal l'emporte — *sauf* si :
       - le score maximal est trop faible (< ``min_score``), ou
       - les deux meilleures catégories sont trop proches (écart < ``margin``).
       Dans ces deux cas (ambiguïté), on retombe sur ``"general"``.
"""

from __future__ import annotations

import re

from .models import MODEL_NAMES
from .signals import SIGNALS

# Seuil minimal : en dessous, aucune catégorie n'est suffisamment convaincante.
DEFAULT_MIN_SCORE: float = 3.0
# Marge minimale entre le 1er et le 2ème score pour considérer le choix fiable.
DEFAULT_MARGIN: float = 1.0


def normalize(query: str) -> str:
    """Normalise la requête pour le scoring.

    - passe en minuscules ;
    - réduit les suites d'espaces à une seule espace ;
    - retire les espaces de début/fin.
    Les accents sont *conservés* (la config des signaux gère les variantes
    accentuées/non-accentuées côté français).
    """
    if not query:
        return ""
    text = query.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_category(normalized: str, spec: dict) -> float:
    """Calcule le score d'une catégorie sur une requête déjà normalisée."""
    score = 0.0
    # --- Mots-clés --------------------------------------------------------
    for term, weight in spec["keywords"]:
        if " " in term:
            # Expression multi-mots : recherche en sous-chaîne.
            if term in normalized:
                score += weight
        else:
            # Mot simple : recherche en *mot entier* pour éviter les faux positifs
            # (ex. "pythonesque" ne doit pas déclencher "python").
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                score += weight
    # --- Regex ------------------------------------------------------------
    for pattern, weight in spec["patterns"]:
        if pattern.search(normalized):
            score += weight
    return score


class Router:
    """Classifie une requête vers l'un des identifiants de :data:`MODEL_NAMES`.

    Args:
        min_score: score minimal requis pour qu'une catégorie soit retenue.
            En dessous, on considère la requête ambiguë → ``"general"``.
        margin: écart minimal entre les deux meilleurs scores. Si le 1er et le
            2ème sont trop proches, on considère la requête ambiguë → ``"general"``.
        signals: dictionnaire de signaux (par défaut :data:`SIGNALS`). Permet
            d'injecter une config personnalisée, notamment pour les tests.
    """

    def __init__(
        self,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        margin: float = DEFAULT_MARGIN,
        signals: dict | None = None,
    ) -> None:
        self.min_score = min_score
        self.margin = margin
        self.signals = signals if signals is not None else SIGNALS

    def score(self, query: str) -> dict[str, float]:
        """Retourne le score de chaque catégorie pour la requête donnée.

        Pratique pour le débogage et les tests : on voit exactement pourquoi
        telle catégorie a été choisie (ou rejetée).
        """
        normalized = normalize(query)
        return {
            name: _score_category(normalized, self.signals[name])
            for name in MODEL_NAMES
        }

    def classify(self, query: str) -> str:
        """Identifiant de la catégorie choisie (l'un des :data:`MODEL_NAMES`).

        Renvoie ``"general"`` si la requête est ambiguë ou trop faible en signal.
        Cas limite : requête vide → ``"general"``.
        """
        normalized = normalize(query)
        if not normalized:
            return "general"

        scores = self.score(query)
        # Tri décroissant par score.
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Ambiguïté : signal trop faible OU deux catégories trop proches.
        if best_score < self.min_score:
            return "general"
        if (best_score - second_score) < self.margin:
            return "general"
        return best_name
