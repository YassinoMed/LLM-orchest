"""Tests du moteur de classification :class:`orchestrator.router.Router`.

Vérifie que chaque catégorie est correctement détectée sur des requêtes
représentatives, plus les cas limites (vide, ambigu, normalisation).
"""

import pytest

from orchestrator.router import Router, normalize


# ---------------------------------------------------------------------------
# Cas paramétrés : (requête, catégorie attendue)
# On couvre les 6 catégories + l'ambiguïté + les exemples de la spec.
# ---------------------------------------------------------------------------
ROUTING_CASES = [
    # --- CODE -------------------------------------------------------------
    ("Peux-tu me montrer comment trier une liste en Python ?", "code"),
    ("Écris une fonction qui additionne deux nombres en JavaScript", "code"),
    ("J'ai un bug dans ma fonction def calculer_total():", "code"),
    ("Comment corriger cette erreur : console.log(x)", "code"),
    ("Implémente un tri rapide en C++", "code"),
    ("refactorise cette classe Java", "code"),
    # --- MATH -------------------------------------------------------------
    ("Combien font 345 * 678 ?", "math"),
    ("Calcule 12 + 5", "math"),
    ("Résous l'équation 2x + 3 = 11", "math"),
    ("Quelle est la dérivée de x^2 ?", "math"),
    ("Combien ça fait 15% de 200 ?", "math"),
    # --- CREATIVE ---------------------------------------------------------
    ("Raconte-moi une histoire triste sur un robot", "creative"),
    ("Écris un poème sur l'automne", "creative"),
    ("Invente un scénario pour un court-métrage", "creative"),
    ("Écris-moi un haïku sur la mer", "creative"),
    # --- FACTUAL ----------------------------------------------------------
    ("Explique-moi la photosynthèse", "factual"),
    ("Qu'est-ce que la photosynthèse ?", "factual"),
    ("Définis ce qu'est un algorithme", "factual"),
    ("Quelle est la différence entre TCP et UDP ?", "factual"),
    ("Qui était Albert Einstein ?", "factual"),
    # --- TRANSLATION ------------------------------------------------------
    ("Traduis cette phrase en espagnol : bonjour", "translation"),
    ("Comment dit-on 'merci' en japonais ?", "translation"),
    ("Traduire ce texte en anglais", "translation"),
    # --- GENERAL ----------------------------------------------------------
    ("Bonjour, comment vas-tu ?", "general"),
    ("Donne-moi un conseil pour mieux dormir", "general"),
    ("que penses-tu de la nouvelle mise à jour ?", "general"),
]


@pytest.mark.parametrize("query,expected", ROUTING_CASES)
def test_classify_routes_correctly(query: str, expected: str) -> None:
    router = Router()
    assert router.classify(query) == expected, (
        f"Requête {query!r} aurait dû être routée vers {expected!r}, "
        f"pas {router.classify(query)!r}"
    )


def test_classify_empty_string_returns_general() -> None:
    assert Router().classify("") == "general"


def test_classify_whitespace_only_returns_general() -> None:
    assert Router().classify("    ") == "general"


def test_classify_ambiguous_low_signal_returns_general() -> None:
    """Une requête avec peu de signal distinctif doit tomber sur 'general'."""
    # "xyz" : aucun mot-clé connu → score nul → ambigu → general.
    assert Router().classify("xyz qzzz") == "general"


def test_normalize_lowercases_and_collapses_spaces() -> None:
    assert normalize("  Bonjour   LE   Monde  ") == "bonjour le monde"


def test_normalize_empty_returns_empty() -> None:
    assert normalize("") == ""


def test_normalize_preserves_accents() -> None:
    """Les accents sont conservés (gérés côté signaux), pas supprimés."""
    assert normalize("Café Été") == "café été"


def test_score_returns_all_categories() -> None:
    """``score`` doit retourner une entrée par modèle du catalogue."""
    from orchestrator.models import MODEL_NAMES
    scores = Router().score("Calcule 2 + 2")
    assert set(scores.keys()) == set(MODEL_NAMES)
    # Le math doit dominer.
    assert scores["math"] > 0


def test_math_score_dominates_for_arithmetic() -> None:
    scores = Router().score("Combien font 345 * 678 ?")
    assert scores["math"] == max(scores.values())


def test_router_respects_custom_thresholds() -> None:
    """Avec un seuil délibérément hors de portée, tout devient ambigu."""
    router = Router(min_score=1_000_000)  # impossible à atteindre
    assert router.classify("Calcule 345 * 678") == "general"


def test_code_signal_stronger_than_factual_for_python_snippet() -> None:
    """Un snippet de code doit l'emporter sur le signal factuel « comment »."""
    scores = Router().score("Comment corriger def foo(): pass")
    assert scores["code"] > scores["factual"]
