"""Script d'entraînement du MLP 12 couches pour le routage neuronal.

Usage depuis la racine du projet :

    python -m orchestrator.train [OPTIONS]

Options :

    --epochs N         Nombre d'époques (défaut 30)
    --batch-size N     Taille des mini-batches (défaut 64)
    --lr RATE          Taux d'apprentissage (défaut 0.001)
    --dropout RATE     Taux de dropout (défaut 0.3)
    --weight-decay     Régularisation L2 (défaut 1e-4)
    --max-features     Taille du vocabulaire TF-IDF (défaut 5000)
    --samples-per-cat  Exemples par catégorie (défaut 500)
    --output-dir       Répertoire de sortie (défaut checkpoints/)
    --data-dir         Répertoire des données (défaut data/)
    --seed             Graine aléatoire (défaut 42)
    --device           Device forcé (auto/cpu/mps/cuda, défaut auto)
    --no-save          Ne pas sauvegarder le modèle (pour les tests)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Configuration du logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator.train")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orchestrator.train",
        description="Entraîne le MLP 12 couches pour le routage neuronal.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--samples-per-cat", type=int, default=500)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--data-dir", default="data/routing_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder")
    return parser.parse_args(argv)


def _get_device(device_str: str):
    import torch
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main(argv: list[str] | None = None) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .data_generation import generate_dataset, save_dataset
    from .models import MODEL_NAMES
    from .neural_model import MLPClassifier12
    from .tfidf import TfidfVectorizer

    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _get_device(args.device)
    logger.info("Device : %s", device)
    logger.info("Catégories : %s (%d classes)", list(MODEL_NAMES), len(MODEL_NAMES))

    # =====================================================================
    # 1. Génération / chargement du dataset
    # =====================================================================
    data_path = Path(args.data_dir)
    if data_path.exists():
        logger.info("Chargement du dataset existant : %s", data_path)
        from .data_generation import load_dataset
        dataset = load_dataset(data_path)
    else:
        logger.info("Génération du dataset synthétique (%d ex/cat)...", args.samples_per_cat)
        dataset = generate_dataset(
            seed=args.seed,
            samples_per_category=args.samples_per_cat,
        )
        save_dataset(dataset, data_path)

    logger.info(
        "Dataset : train=%d, val=%d, test=%d",
        len(dataset["train"]),
        len(dataset["val"]),
        len(dataset["test"]),
    )

    # =====================================================================
    # 2. Vectorisation TF-IDF (fit sur train uniquement)
    # =====================================================================
    logger.info("Vectorisation TF-IDF (max_features=%d)...", args.max_features)
    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=2)

    # Fit sur le train.
    train_texts = dataset["train"]["text"]
    vectorizer.fit(train_texts)
    logger.info("Vocabulaire : %d termes", vectorizer.vocab_size)

    # Transform sur tous les splits.
    X_train = vectorizer.transform(dataset["train"]["text"])
    X_val = vectorizer.transform(dataset["val"]["text"])
    X_test = vectorizer.transform(dataset["test"]["text"])

    y_train = np.array(dataset["train"]["label"], dtype=np.int64)
    y_val = np.array(dataset["val"]["label"], dtype=np.int64)
    y_test = np.array(dataset["test"]["label"], dtype=np.int64)

    # Conversion en tenseurs PyTorch.
    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    # DataLoaders.
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # =====================================================================
    # 3. Modèle MLP 12 couches
    # =====================================================================
    input_dim = vectorizer.vocab_size
    model = MLPClassifier12(
        input_dim=input_dim,
        num_classes=len(MODEL_NAMES),
        dropout=args.dropout,
    ).to(device)

    n_params = model.count_parameters()
    logger.info("Modèle MLP 12 couches : %d paramètres, input_dim=%d", n_params, input_dim)

    # =====================================================================
    # 4. Entraînement
    # =====================================================================
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = 7  # early stopping

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
            f"{val_loss:>9.4f}  {val_acc:>7.1%}  {elapsed:>5.1f}s"
        )

        # Early stopping.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping à l'epoch %d (meilleur val_acc=%.1f%%)",
                             epoch, best_val_acc * 100)
                break

    # =====================================================================
    # 5. Évaluation finale sur le test set
    # =====================================================================
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()

    print(f"\n{'='*60}")
    print(f"Meilleur val_acc : {best_val_acc:.1%}")
    print(f"Test accuracy    : {test_acc:.1%}")
    print(f"Nombre de paramètres : {n_params:,}")
    print(f"{'='*60}")

    # =====================================================================
    # 6. Sauvegarde
    # =====================================================================
    if not args.no_save:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        model_path = output / "model.pt"
        tfidf_path = output / "tfidf.json"

        # Sauvegarde modèle (sur CPU pour la portabilité).
        model.cpu()
        torch.save(
            {"model_state_dict": model.state_dict(),
             "input_dim": input_dim,
             "num_classes": len(MODEL_NAMES),
             "best_val_acc": best_val_acc,
             "test_acc": test_acc},
            str(model_path),
        )

        # Sauvegarde vectoriseur.
        vectorizer.save(str(tfidf_path))

        print(f"\nCheckpoints sauvegardés :")
        print(f"  Modèle : {model_path}")
        print(f"  TF-IDF : {tfidf_path}")
        print(f"\nPour utiliser le mode neuronal :")
        print(f"  python cli.py 'Ta requête' --router neural")
    else:
        logger.info("Sauvegarde désactivée (--no-save)")


if __name__ == "__main__":
    main()
