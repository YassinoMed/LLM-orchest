"""Tests d'intégration de :class:`orchestrator.orchestrator.Orchestrator`.

Couvre :
    - Les 3 exemples officiels de la spec (cas de non-régression bloquants).
    - La sérialisation JSON (format strict conforme à la spec).
    - Le contrat de RoutingResult (champs, types, immuabilité).
    - L'usage idiomatique de l'API publique.
"""

import json

import pytest

from orchestrator import MODEL_NAMES, Orchestrator, RoutingResult


@pytest.fixture
def orch() -> Orchestrator:
    return Orchestrator()


# ---------------------------------------------------------------------------
# EXEMPLES OFFICIELS DE LA SPEC — cas de non-régression bloquants.
# Si l'un de ces tests échoue, le projet ne respecte plus sa spec.
# ---------------------------------------------------------------------------
SPEC_EXAMPLES = [
    (
        "Peux-tu me montrer comment trier une liste de dictionnaires par une clé en Python ?",
        "code",
    ),
    (
        "Raconte-moi une histoire triste à propos d'un robot qui apprend à aimer.",
        "creative",
    ),
    (
        "Combien font 345 * 678 ?",
        "math",
    ),
]


@pytest.mark.parametrize("query,expected_model", SPEC_EXAMPLES)
def test_spec_example_routes_to_expected_model(
    orch: Orchestrator, query: str, expected_model: str
) -> None:
    """Chaque exemple de la spec doit être routé vers le bon modèle."""
    result = orch.route(query)
    assert result.model == expected_model


def test_spec_math_example_exact_reduced_prompt(orch: Orchestrator) -> None:
    """L'exemple math doit produire exactement le prompt de la spec."""
    result = orch.route("Combien font 345 * 678 ?")
    assert result.reduced_prompt == "Calcule 345 * 678."


# ---------------------------------------------------------------------------
# Sérialisation JSON conforme à la spec
# ---------------------------------------------------------------------------
def test_to_json_is_valid_json(orch: Orchestrator) -> None:
    raw = orch.to_json("Calcule 2 + 2")
    parsed = json.loads(raw)  # lève si JSON invalide
    assert set(parsed.keys()) == {"model", "reduced_prompt"}


def test_to_json_strict_format_no_extra_keys(orch: Orchestrator) -> None:
    """Le JSON ne doit contenir QUE 'model' et 'reduced_prompt' (spec stricte)."""
    raw = orch.to_json("Traduis bonjour en anglais")
    parsed = json.loads(raw)
    assert sorted(parsed.keys()) == ["model", "reduced_prompt"]


def test_to_json_is_compact_by_default(orch: Orchestrator) -> None:
    """Sans indentation, le JSON doit être compact (une seule ligne)."""
    raw = orch.to_json("Calcule 2 + 2")
    assert "\n" not in raw


def test_to_json_with_indent_is_multiline(orch: Orchestrator) -> None:
    raw = orch.to_json("Calcule 2 + 2", indent=2)
    assert "\n" in raw
    # Toujours du JSON valide.
    json.loads(raw)


def test_to_json_preserves_unicode(orch: Orchestrator) -> None:
    """Les accents doivent être conservés (ensure_ascii=False)."""
    raw = orch.to_json("Raconte une histoire sur l'été")
    # Pas d'échappement \uXXXX pour les caractères accentués.
    assert "é" in raw or "\\u00e9" not in raw


# ---------------------------------------------------------------------------
# Contrat RoutingResult
# ---------------------------------------------------------------------------
def test_routing_result_is_immutable(orch: Orchestrator) -> None:
    """RoutingResult est frozen : on ne peut pas muter ses champs."""
    result = orch.route("Calcule 2 + 2")
    with pytest.raises(Exception):  # FrozenInstanceError (sous-classe d'AttributeError)
        result.model = "autre_chose"  # type: ignore[misc]


def test_routing_result_model_is_known_model(orch: Orchestrator) -> None:
    """Le modèle choisi fait toujours partie du catalogue officiel."""
    result = orch.route("n'importe quelle requête")
    assert result.model in MODEL_NAMES


def test_routing_result_has_nonempty_fields(orch: Orchestrator) -> None:
    result = orch.route("Calcule 2 + 2")
    assert result.model
    assert result.reduced_prompt


def test_routing_result_to_dict_roundtrips(orch: Orchestrator) -> None:
    result = orch.route("Calcule 2 + 2")
    d = result.to_dict()
    assert d == {"model": result.model, "reduced_prompt": result.reduced_prompt}


# ---------------------------------------------------------------------------
# Usage idiomatique de l'API publique
# ---------------------------------------------------------------------------
def test_orchestrator_accepts_custom_router() -> None:
    """On peut injecter un routeur personnalisé (testabilité/paramétrage)."""
    from orchestrator.router import Router
    custom = Orchestrator(router=Router(min_score=0.5, margin=0.5))
    result = custom.route("Calcule 2 + 2")
    assert result.model == "math"


def test_all_spec_models_covered() -> None:
    """Le catalogue contient exactement les 6 modèles de la spec."""
    assert set(MODEL_NAMES) == {"code", "creative", "factual", "math",
                                "translation", "general"}
