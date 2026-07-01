"""Smoke tests du pipeline d'entraînement (sans dépendre de checkpoints existants).

Ces tests valident que **tout le pipeline fonctionne de bout en bout** :
génération de données → TF-IDF → MLP 12 couches → entraînement (1 epoch) →
inférence. Ils n'écrivent pas sur disque et tournent en quelques secondes sur CPU.

Ignorés (skip) si PyTorch ou `datasets` ne sont pas installés.
"""

import numpy as np
import pytest

# Skip global si PyTorch ou datasets absent.
torch_available = True
datasets_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False
try:
    import datasets  # noqa: F401
except ImportError:
    datasets_available = False

pytestmark = pytest.mark.skipif(
    not (torch_available and datasets_available),
    reason="PyTorch et/ou datasets non installés.",
)

from orchestrator.models import MODEL_NAMES
from orchestrator.neural_model import MLPClassifier12
from orchestrator.tfidf import TfidfVectorizer


@pytest.fixture(scope="module")
def small_dataset():
    """Génère un mini-dataset synthétique (rapide)."""
    from orchestrator.data_generation import generate_dataset
    return generate_dataset(seed=0, samples_per_category=30, extra_variations=15)


@pytest.fixture(scope="module")
def trained_pipeline(small_dataset):
    """Pipeline complet entraîné sur 1 epoch (smoke test).

    Retourne un tuple (vectorizer, model, device).
    """
    device = torch.device("cpu")

    # TF-IDF.
    vec = TfidfVectorizer(max_features=1000, min_df=1)
    vec.fit(small_dataset["train"]["text"])
    X = vec.transform(small_dataset["train"]["text"])
    y = np.array(small_dataset["train"]["label"], dtype=np.int64)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    # Modèle 12 couches.
    model = MLPClassifier12(input_dim=vec.vocab_size, dropout=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Entraînement : 1 epoch complet (smoke test, pas de recherche de perf).
    model.train()
    for _ in range(1):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return vec, model, device


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
def test_tfidf_fit_transform_shapes(small_dataset) -> None:
    vec = TfidfVectorizer(max_features=1000, min_df=1)
    vec.fit(small_dataset["train"]["text"])
    X = vec.transform(small_dataset["train"]["text"])
    assert X.shape[0] == len(small_dataset["train"])
    assert X.shape[1] == vec.vocab_size


def test_tfidf_l2_normalized(small_dataset) -> None:
    """Chaque ligne TF-IDF doit être L2-normalisée (norme ~1)."""
    vec = TfidfVectorizer(max_features=1000, min_df=1)
    vec.fit(small_dataset["train"]["text"])
    X = vec.transform(small_dataset["train"]["text"])
    norms = np.linalg.norm(X, axis=1)
    # On exclut les éventuelles lignes toutes nulles (textes vides après tokenisation).
    nonzero = norms[norms > 0]
    assert np.allclose(nonzero, 1.0, atol=1e-5)


def test_tfidf_save_load_roundtrip(tmp_path, small_dataset) -> None:
    """Sauvegarde/rechargement JSON du vectoriseur."""
    vec = TfidfVectorizer(max_features=500, min_df=1)
    vec.fit(small_dataset["train"]["text"])
    path = tmp_path / "tfidf.json"
    vec.save(path)

    vec2 = TfidfVectorizer.load(path)
    assert vec2.vocab_size == vec.vocab_size
    assert vec2.vocabulary_ == vec.vocabulary_
    np.testing.assert_allclose(vec2.idf_, vec.idf_, atol=1e-6)


# ---------------------------------------------------------------------------
# Modèle MLP 12 couches
# ---------------------------------------------------------------------------
def test_model_has_exactly_12_linear_layers() -> None:
    model = MLPClassifier12(input_dim=100)
    n_linear = sum(1 for m in model.modules() if m.__class__.__name__ == "Linear")
    assert n_linear == 12


def test_model_output_shape() -> None:
    model = MLPClassifier12(input_dim=100)
    x = torch.randn(8, 100)
    out = model(x)
    assert out.shape == (8, len(MODEL_NAMES))


def test_model_parameter_count_positive() -> None:
    model = MLPClassifier12(input_dim=100)
    assert model.count_parameters() > 0


def test_model_predict_proba_sums_to_one() -> None:
    model = MLPClassifier12(input_dim=100)
    model.eval()
    x = torch.randn(4, 100)
    probs = model.predict_proba(x)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-4)


# ---------------------------------------------------------------------------
# Pipeline end-to-end (smoke)
# ---------------------------------------------------------------------------
def test_trained_pipeline_predicts_known_label(trained_pipeline) -> None:
    """Le modèle entraîné prédit un label valide sur une requête de test."""
    vec, model, _ = trained_pipeline
    query = "Combien font 345 * 678 ?"
    x = torch.from_numpy(vec.transform_one(query)).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    assert 0 <= pred < len(MODEL_NAMES)


def test_training_reduces_loss(small_dataset) -> None:
    """Vérifie que l'entraînement fait baisser la perte (le modèle apprend)."""
    device = torch.device("cpu")
    vec = TfidfVectorizer(max_features=1000, min_df=1)
    vec.fit(small_dataset["train"]["text"])
    X = torch.from_numpy(vec.transform(small_dataset["train"]["text"]))
    y = torch.from_numpy(np.array(small_dataset["train"]["label"], dtype=np.int64))

    model = MLPClassifier12(input_dim=vec.vocab_size, dropout=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    optimizer.zero_grad()
    loss_before = criterion(model(X), y).item()
    for _ in range(20):  # quelques steps pour voir la perte baisser
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    loss_after = criterion(model(X), y).item()

    assert loss_after < loss_before, (
        f"L'entraînement aurait dû réduire la perte : {loss_before:.4f} → {loss_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Génération de données
# ---------------------------------------------------------------------------
def test_generated_dataset_has_all_splits(small_dataset) -> None:
    assert set(small_dataset.keys()) == {"train", "val", "test"}


def test_generated_dataset_labels_in_range(small_dataset) -> None:
    for split_data in small_dataset.values():
        labels = split_data["label"]
        assert all(0 <= l < len(MODEL_NAMES) for l in labels)


def test_generated_dataset_all_categories_present(small_dataset) -> None:
    """Chaque catégorie doit apparaître au moins une fois dans le train."""
    train_labels = set(small_dataset["train"]["label"])
    assert train_labels == set(range(len(MODEL_NAMES)))
