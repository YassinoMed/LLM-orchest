"""Tests du routeur neuronal :class:`orchestrator.neural_router.NeuralRouter`.

Ces tests sont automatiquement **ignorés** (skip) si :
    - PyTorch n'est pas installé, OU
    - les checkpoints (checkpoints/model.pt + checkpoints/tfidf.json) sont
      absents (c'est-à-dire que l'entraînement n'a pas encore été lancé).

Ainsi, la suite de tests reste verte dans un environnement minimal, et ces
tests s'activent dès que le modèle a été entraîné.
"""

import pytest

# Détection de la disponibilité de PyTorch et des checkpoints.
torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False

from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_DIR = _ROOT / "checkpoints"
_MODEL_PATH = _CHECKPOINTS_DIR / "model.pt"
_TFIDF_PATH = _CHECKPOINTS_DIR / "tfidf.json"

checkpoints_available = _MODEL_PATH.exists() and _TFIDF_PATH.exists()

# Skip global du fichier si l'environnement n'est pas prêt.
pytestmark = pytest.mark.skipif(
    not (torch_available and checkpoints_available),
    reason=(
        "PyTorch ou les checkpoints sont absents. "
        "Lancez : python -m orchestrator.train --epochs 30"
    ),
)

from orchestrator import MODEL_NAMES
from orchestrator.neural_router import NeuralRouter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_neural_router_loads_from_default_paths() -> None:
    """Le routeur se charge depuis checkpoints/ par défaut."""
    router = NeuralRouter()  # ne doit pas lever
    assert router.model is not None
    assert router.vectorizer is not None


def test_neural_router_param_count() -> None:
    """Le modèle possède bien 12 couches linéaires."""
    router = NeuralRouter()
    n_linear = sum(1 for m in router.model.modules() if m.__class__.__name__ == "Linear")
    assert n_linear == 12


# ---------------------------------------------------------------------------
# Contrat d'interface (identique au routeur heuristique)
# ---------------------------------------------------------------------------
def test_classify_returns_known_model_name() -> None:
    """classify() retourne toujours un identifiant de MODEL_NAMES."""
    router = NeuralRouter()
    result = router.classify("Combien font 345 * 678 ?")
    assert result in MODEL_NAMES


def test_score_returns_all_categories() -> None:
    """score() retourne une entrée par catégorie."""
    router = NeuralRouter()
    scores = router.score("Écris une fonction en Python")
    assert set(scores.keys()) == set(MODEL_NAMES)


def test_score_values_sum_to_one() -> None:
    """Les probabilités (softmax) somment à ~1.0."""
    router = NeuralRouter()
    scores = router.score("Calcule 2 + 2")
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)


def test_score_values_in_zero_one() -> None:
    """Chaque probabilité est dans [0, 1]."""
    router = NeuralRouter()
    scores = router.score("Traduis bonjour en anglais")
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_score_best_matches_classify() -> None:
    """La catégorie argmax de score() doit correspondre à classify() (sauf si ambiguë)."""
    router = NeuralRouter(confidence_threshold=0.0)  # désactive le seuil → pas de "general"
    scores = router.score("Combien font 345 * 678 ?")
    best = max(scores, key=scores.get)
    assert router.classify("Combien font 345 * 678 ?") == best


# ---------------------------------------------------------------------------
# Cas limites
# ---------------------------------------------------------------------------
def test_classify_empty_string() -> None:
    """Requête vide → ne plante pas, retourne un MODEL_NAMES."""
    router = NeuralRouter()
    assert router.classify("") in MODEL_NAMES


def test_classify_low_confidence_returns_general() -> None:
    """Avec un seuil très élevé, toute prédiction est rejetée → 'general'."""
    router = NeuralRouter(confidence_threshold=0.9999)
    result = router.classify("Calcule 2 + 2")
    assert result == "general"


# ---------------------------------------------------------------------------
# Erreur si checkpoints absents (testé via un chemin inexistant)
# ---------------------------------------------------------------------------
def test_neural_router_raises_on_missing_checkpoint(tmp_path) -> None:
    """Une erreur claire est levée si le checkpoint n'existe pas."""
    with pytest.raises(FileNotFoundError, match="Checkpoint"):
        NeuralRouter(
            model_path=str(tmp_path / "absent.pt"),
            tfidf_path=str(tmp_path / "absent.json"),
        )
