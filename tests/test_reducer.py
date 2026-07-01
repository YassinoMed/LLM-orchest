"""Tests du module :mod:`orchestrator.reducer`.

Vérifie les transformations de réduction par catégorie + les cas limites.
Les assertions sur le contenu exact sont volontairement ciblées sur les
éléments stables (détection de langage, extraction numérique, structure) plutôt
que sur une chaîne exacte fragile.
"""

import pytest

from orchestrator.reducer import reduce


# ---------------------------------------------------------------------------
# Réduction par catégorie : (requête, modèle, fragments attendus dans le réduit)
# ---------------------------------------------------------------------------
REDUCER_CASES = [
    # --- MATH : extraction de l'expression numérique -----------------------
    ("Combien font 345 * 678 ?", "math", ["Calcule", "345 * 678"]),
    ("Calcule 12 + 5", "math", ["Calcule", "12 + 5"]),
    # --- CODE : détection du langage + formulation -------------------------
    ("Tri d'une liste en Python", "code", ["Python", "code"]),
    ("Comment trier en JavaScript", "code", ["Javascript", "code"]),
    # --- TRANSLATION : langue cible + contenu ------------------------------
    ("Traduis en espagnol : bonjour", "translation", ["espagnol"]),
    ("Comment dit-on merci en japonais ?", "translation", ["japonais"]),
    # --- CREATIVE : type d'œuvre -------------------------------------------
    ("Écris un poème sur l'automne", "creative", ["poème"]),
    ("Raconte une histoire triste", "creative", ["histoire"]),
    # --- FACTUAL : sujet isolé ---------------------------------------------
    ("Explique-moi la photosynthèse", "factual", ["photosynthèse"]),
    # --- GENERAL : clarification ------------------------------------------
    ("Bonjour", "general", ["clarification"]),
]


@pytest.mark.parametrize("query,model,expected_fragments", REDUCER_CASES)
def test_reduce_contains_expected_fragments(
    query: str, model: str, expected_fragments: list[str]
) -> None:
    result = reduce(query, model)
    for fragment in expected_fragments:
        assert fragment.lower() in result.lower(), (
            f"Fragment {fragment!r} absent du prompt réduit {result!r}"
        )


def test_reduce_always_ends_with_dot() -> None:
    """Tous les prompts réduits se terminent par un point (rendu propre)."""
    for query, model, _ in REDUCER_CASES:
        assert reduce(query, model).endswith("."), (
            f"Prompt réduit pour {query!r} devrait finir par un point"
        )


def test_reduce_no_double_trailing_dot() -> None:
    result = reduce("Calcule 2 + 2.", "math")
    assert not result.endswith("..")
    assert result.endswith(".")


def test_reduce_unknown_model_falls_back_to_general() -> None:
    """Un modèle inconnu doit déclencher le réducteur général (pas d'erreur)."""
    result = reduce("n'importe quoi", "modele_inexistant")
    assert "clarification" in result.lower()


def test_reduce_empty_query_general() -> None:
    """Requête vide → message de clarification, sans planter."""
    result = reduce("", "general")
    assert isinstance(result, str)
    assert len(result) > 0


def test_reduce_strips_filler_phrases() -> None:
    """Les formules de politesse sont retirées."""
    result = reduce("Peux-tu expliquer la relativité s'il te plaît", "factual")
    assert "peux-tu" not in result.lower()
    assert "s'il te plaît" not in result.lower()


def test_reduce_math_normalizes_x_operator() -> None:
    """La multiplication « x » / « × » est normalisée en « * »."""
    result = reduce("Combien font 6 x 7 ?", "math")
    assert "*" in result
    assert "6 * 7" in result


def test_reduce_code_removes_language_from_subject() -> None:
    """Le langage ne doit pas apparaître deux fois (évite la redondance)."""
    result = reduce("Tri d'une liste en python", "code")
    # "python" ne doit pas traîner dans la partie sujet (déjà mentionné au début).
    # On accepte une seule occurrence (la mention du langage cible).
    assert result.lower().count("python") <= 1


def test_reduce_translation_strips_redundant_verb_from_content() -> None:
    """Le verbe de traduction de l'utilisateur est retiré du contenu à traduire.

    Le reduced_prompt commence légitimement par le verbe template « Traduis »,
    mais le verbe redondant de la requête ne doit pas réapparaître dans le
    contenu entre guillemets.
    """
    result = reduce("Traduis traduis bonjour en anglais", "translation")
    # Le contenu entre « » ne doit pas contenir un second « traduis ».
    import re
    match = re.search(r"«\s*(.*?)\s*»", result)
    assert match is not None, f"Pas de contenu entre guillemets dans {result!r}"
    assert "traduis" not in match.group(1).lower()


def test_reduce_creative_strips_intro_verb() -> None:
    """Les verbes introductifs créatifs sont retirés du sujet."""
    result = reduce("Raconte-moi une histoire triste sur un robot", "creative")
    assert "raconte" not in result.lower()
